from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Import functions from your AI and DS engines
from ai_engine.card_generator import generate_cards
from ds_engine.predictor import predict_retention

app = FastAPI(
    title="SummQ Python Service",
    description="API for AI Card Generation and DS Retention Prediction",
    version="1.0.0"
)

# --- Health Check Route ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "SmartDeck Python Service is running on Render"}


# --- AI Engine Endpoints ---
class GenerateCardsRequest(BaseModel):
    text_content: str
    num_cards: int = 5

@app.post("/api/ai/generate-cards")
def api_generate_cards(request: GenerateCardsRequest):
    try:
        # Pass the data to the AI dev's function
        cards = generate_cards(request.text_content, request.num_cards)
        return {"success": True, "data": cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- DS Engine Endpoints ---
class RetentionPredictionRequest(BaseModel):
    user_id: str
    card_id: str
    study_history: List[Dict[str, Any]]  # Example: [{"date": "...", "score": 4}]

@app.post("/api/ds/predict-retention")
def api_predict_retention(request: RetentionPredictionRequest):
    try:
        # Pass the data to the DS dev's model prediction function
        prediction = predict_retention(request.user_id, request.card_id, request.study_history)
        return {"success": True, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))