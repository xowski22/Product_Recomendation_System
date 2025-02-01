from fastapi import FastAPI, HTTPException
import yaml
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict
import torch
import numpy as np
import logging
from pathlib import Path
import pickle

from ..models.model import MatrixFactorization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

model = None #init with trained model
user_mapping = None #load user mapping
item_mapping = None #load item mapping

def load_config() -> dict:
    try:
        with open("../config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

def load_model_and_mappings() -> Tuple[MatrixFactorization, Dict, Dict]:
    try:
        config = load_config()

        with open(Path(config['data']['mappings']['user_mapping_path']), "rb") as f:
            user_mapping = pickle.load(f)

        with open(Path(config['data']['mappings']['item_mapping_path']), "rb") as f:
            item_mapping = pickle.load(f)

        model = MatrixFactorization(
            num_users=len(user_mapping),
            n_items=len(item_mapping),
            embedding_dim=config['model']['embedding_dim'],
            reg_lambda=config['model']['reg_lambda']
        )

        model_path = Path(config['model']['checkpoint_path'])
        model.load_state_dict(torch.load(model_path))
        model.eval()

        logger.info("Successfully loaded model and mappings")
        return model, user_mapping, item_mapping
    except Exception as e:
        logger.error(f"Failed to load model and mappings: {str(e)}")
        raise


class RatingRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    item_id: str = Field(..., description="Item ID")

class RatingResponse(BaseModel):
    predicted_rating: float

class RecomendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    n_recommendations: int = Field(default=5, ge=1, le=100, description="Number of recommendations to return")

class RecommendationResponse(BaseModel):
    items: List[str] = Field(..., description="List of recommended item IDs")
    scores: List[float] = Field(..., description="Corresponding prediction scores")

@app.post("predict/rating/", response_model=RatingResponse)
async def predict_rating(request: RatingRequest):
    try:
        if request.user_mapping not in user_mapping:
            raise HTTPException(status_code=404, detail="User mapping not found")
        if item_mapping not in item_mapping:
            raise HTTPException(status_code=404, detail="Item mapping not found")

        user_idx = user_mapping[request.user_id]
        item_idx = item_mapping[request.item_id]

        user_tensor = torch.tensor([user_idx], dtype=torch.long)
        item_tensor = torch.tensor([item_idx], dtype=torch.long)

        with torch.no_grad():
            model.eval()
            prediction = model(user_tensor, item_tensor)
            if isinstance(prediction, tuple):
                prediction = prediction[0]

            predicted_rating = float(prediction.item() * 5.0)
            predicted_rating = max(1.0, min(5.0, predicted_rating))

        return RatingResponse(predicted_rating=predicted_rating)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/", response_model=RecommendationResponse)
async def get_recommendations(request: RecomendationRequest):
    try:

        if request.user_id not in user_mapping:
            raise HTTPException(status_code=404, detail="User ID not found")

        user_idx = user_mapping[request.user_id]

        user_tensor = torch.tensor([user_idx]*len(item_mapping), dtype=torch.long)
        item_tensor = torch.tensor(list(range(len(item_mapping))), dtype=torch.long)

        with torch.no_grad():
            model.eval()
            predictions = model(user_tensor, item_tensor)
            if isinstance(predictions, tuple):
                predictions = predictions[0]

        top_n_indices = np.argsort(-predictions)[:request.n_recommendations]

        reverse_item_mapping = {v: k for k, v in item_mapping.items()}
        recommended_items = [reverse_item_mapping[idx] for idx in top_n_indices]
        recommendation_scores = [float(predictions[idx]) for idx in top_n_indices]

        return RecommendationResponse(
            items=recommended_items,
            scores=recommendation_scores
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/reload_model/")
async def reload_model():
    try:
        global model, user_mapping, item_mapping

        return {"status": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")