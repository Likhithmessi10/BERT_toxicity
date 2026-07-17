import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from detoxify import Detoxify

app = FastAPI(title="BERT Toxicity & Attention Explainer")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Server initialized. Default device: {device}")

# Global cache for loaded Detoxify models to prevent reloading overhead
models_cache = {}

def get_model(model_name: str):
    """
    Loads and caches the specified Detoxify model.
    Supported types: 'original', 'unbiased', 'multilingual', 'original-small', 'unbiased-small'
    """
    if model_name not in models_cache:
        print(f"Loading model '{model_name}' on {device}...")
        try:
            # Detoxify loads model, tokenizer, and config.
            # Local modifications to detoxify.py ensure attn_implementation="eager" is used.
            models_cache[model_name] = Detoxify(model_name, device=device)
            print(f"Model '{model_name}' loaded successfully on {device}.")
        except Exception as e:
            print(f"Failed to load model '{model_name}' on {device}: {e}. Trying CPU...")
            models_cache[model_name] = Detoxify(model_name, device="cpu")
            print(f"Model '{model_name}' loaded successfully on CPU.")
    return models_cache[model_name]

# Preload default model on startup
try:
    get_model("original")
except Exception as e:
    print(f"Startup preloading failed: {e}")

class AnalysisRequest(BaseModel):
    text: str
    model_name: str = "original"
    layer: int = 11
    head: int = -1 # -1 means average all heads

@torch.no_grad()
def extract_predictions_and_attentions(model_instance, text, layer_idx, head_idx):
    """
    Runs model inference and extracts actual predictions, WordPiece tokens, and real
    attention weights for the specified layer and head.
    """
    # 1. Prepare inputs
    inputs = model_instance.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model_instance.model.device)
    
    # 2. Forward pass with attention weights tracking
    outputs = model_instance.model(**inputs, output_attentions=True)
    logits = outputs[0]
    attentions = outputs.attentions  # Tuple of shape: [batch, heads, seq_len, seq_len]
    
    # 3. Calculate predictions
    scores = torch.sigmoid(logits).cpu().squeeze(0)
    
    # If it is a batch of inputs (should not be here, but just in case), handle shapes
    if len(scores.shape) > 1:
        scores = scores[0]
        
    predictions = {}
    for i, class_name in enumerate(model_instance.class_names):
        predictions[class_name] = float(scores[i])
        
    # 4. Extract token names
    input_ids = inputs["input_ids"][0].tolist()
    tokens = model_instance.tokenizer.convert_ids_to_tokens(input_ids)
    
    # 5. Extract attention matrix
    num_layers = len(attentions)
    layer_idx = max(0, min(layer_idx, num_layers - 1))
    
    # shape: [num_heads, seq_len, seq_len]
    layer_attn = attentions[layer_idx][0]
    num_heads = layer_attn.shape[0]
    
    if head_idx == -1:
        # Average attention weights across all heads
        attn_matrix = layer_attn.mean(dim=0)
    else:
        # Extract a specific head's attention matrix
        head_idx = max(0, min(head_idx, num_heads - 1))
        attn_matrix = layer_attn[head_idx]
        
    attention_matrix = attn_matrix.cpu().tolist()
    
    return predictions, tokens, attention_matrix, num_layers, num_heads

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
        
    try:
        # Load or retrieve model from cache
        model_instance = get_model(request.model_name)
        
        # Extract Jigsaw ratings and actual attention weights
        preds, tokens, attn_matrix, num_layers, num_heads = extract_predictions_and_attentions(
            model_instance,
            request.text,
            request.layer,
            request.head
        )
        
        return {
            "predictions": preds,
            "architecture": {
                "tokens": tokens,
                "attention_matrix": attn_matrix,
                "num_layers": num_layers,
                "num_heads": num_heads
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_index():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template index.html not found")
        
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
