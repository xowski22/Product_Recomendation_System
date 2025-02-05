from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
import yaml
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict
import torch
import numpy as np
import logging
from pathlib import Path
import pickle

from src.models.model import MatrixFactorization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        global model, user_mapping, item_mapping
        model, user_mapping, item_mapping = load_model_and_mappings()
        logger.info("Application startup complete - all components loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    finally:
        pass
app = FastAPI(lifespan=lifespan)

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

model = None #init with trained model
user_mapping = None #load user mapping
item_mapping = None #load item mapping

def get_project_root() -> Path:
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent

def load_config() -> dict:
    try:
        project_root = get_project_root()
        config_path = project_root / "config" / "config.yaml"
        logger.info(f"Loading config from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

def load_model_and_mappings() -> Tuple[MatrixFactorization, Dict, Dict]:
    try:
        config = load_config()
        project_root = get_project_root()

        mappings_path = project_root / "data" / "mappings"
        logger.info(f"Loading mapping from: {mappings_path}")

        user_mapping_path = mappings_path / "user_mapping.pkl"
        logger.info(f"Loading mapping from: {user_mapping_path}")
        if not user_mapping_path.exists():
            raise FileNotFoundError(f"User mapping file not found at {user_mapping_path}")

        with open(user_mapping_path, "rb") as f:
            user_mapping = pickle.load(f)
        logger.info("User mapping loaded successfully")

        item_mapping_path = mappings_path / "item_mapping.pkl"
        logger.info(f"Loading mapping from: {item_mapping_path}")
        if not item_mapping_path.exists():
            raise FileNotFoundError(f"Item mapping file not found at {item_mapping_path}")

        with open(item_mapping_path, "rb") as f:
            item_mapping = pickle.load(f)
        logger.info("Item mapping loaded successfully")

        model = MatrixFactorization(
            num_users=len(user_mapping),
            n_items=len(item_mapping),
            embedding_dim=config['model']['embedding_dim'],
            reg_lambda=config['model']['reg_lambda']
        )

        model_path = project_root / "models" / "checkpoints" / "best_model.pt"
        logger.info(f"Loading model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info("Model loaded successfully")

        logger.info("Successfully loaded model and mappings")

        return model, user_mapping, item_mapping
    except Exception as e:
        logger.error(f"Failed to load model and mappings: {str(e)}")
        raise

# @app.on_event("startup")
# async def startup_event():
#
#     try:
#         global model, user_mapping, item_mapping
#         model, user_mapping, item_mapping = load_model_and_mappings()
#         logger.info("Application startup complete - all components loaded successfully")
#     except Exception as e:
#         logger.error(f"Startup failed: {str(e)}")


@app.post("/predict/rating/", response_model=RatingResponse)
async def predict_rating(request: RatingRequest):
    """Predict rating for specific user-item pair"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        user_id = int(request.user_id)
        item_id = int(request.item_id)
        logger.info(f"Converted IDs - user_id: {user_id}, item_id: {item_id}")
    except ValueError:
        logger.info("Failed to convert user ID to integers")
        raise HTTPException(status_code=400, detail="IDs must be valid integers")

    logger.info(f"Checking mappings. User ID exists: {user_id in user_mapping}, Irrelevant: {item_id in item_mapping}")

    if user_id not in user_mapping:
        logger.info(f"User ID {user_id} not found in user_mapping")
        raise HTTPException(status_code=404, detail="User mapping not found")
    if item_id not in item_mapping:
        logger.info(f"Item ID {item_id} not found in item_mapping")
        raise HTTPException(status_code=404, detail="Item mapping not found")

    user_idx = user_mapping[user_id]
    item_idx = item_mapping[item_id]
    logger.info(f"Mapped indices - user_idx: {user_idx}, item_idx: {item_idx}")

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

@app.post("/recommend/", response_model=RecommendationResponse)
async def get_recommendations(request: RecomendationRequest):
    """Get personalized item recommendations for a specific user"""
    try:

        user_id = int(request.user_id)

        if user_id not in user_mapping:
            raise HTTPException(status_code=404, detail="User ID not found")

        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        user_idx = user_mapping[user_id]

        user_tensor = torch.tensor([user_idx]*len(item_mapping), dtype=torch.long)
        item_tensor = torch.tensor(list(range(len(item_mapping))), dtype=torch.long)

        with torch.no_grad():
            model.eval()
            predictions = model(user_tensor, item_tensor)
            if isinstance(predictions, tuple):
                predictions = predictions[0]

        predictions_np = predictions.detach().numpy()
        top_n_indices = np.argsort(-predictions_np)[:request.n_recommendations]

        reverse_item_mapping = {v: k for k, v in item_mapping.items()}
        recommended_items = [str(reverse_item_mapping[idx]) for idx in top_n_indices]
        recommendation_scores = [float(predictions[idx]) * 5.0 for idx in top_n_indices]

        return RecommendationResponse(
            items=recommended_items,
            scores=recommendation_scores
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Checks if every component is healthy, and loaded"""
    try:
        health_status = {
            "status": "healthy",
            "model_loaded": model is not None,
            "user_mapping_loaded": user_mapping is not None,
            "item_mapping_loaded": item_mapping is not None,
            "components": {
                "model": "loaded" if model is not None else "not loaded",
                "user_mapping": "loaded" if user_mapping is not None else "not loaded",
                "item_mapping": "loaded" if item_mapping is not None else "not loaded",
            }
        }
        logger.info(f"Health check performed: {health_status}")
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
@app.post("/reload_model/")
async def reload_model():
    try:
        global model, user_mapping, item_mapping

        return {"status": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.get("/debug/")
async def debug_values():
    return {
            "model_loaded": model is not None,
            "user_mapping_type": str(type(user_mapping)),
            "item_mapping_type": str(type(item_mapping)),
            "user_mapping_size": len(user_mapping) if user_mapping else 0,
            "item_mapping_size": len(item_mapping) if item_mapping else 0
            }
