from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class RecomendationRequest(BaseModel):
    user_id: str
    n_recommendations: int = 5

class RecommendationResponse(BaseModel):
    items: List[int]
    scores: List[float]

@app.post("/recommend/", response_model=RecommendationResponse)
async def get_recommendation(request: RecomendationRequest):
    try:
        items = [1, 2, 3, 4, 5]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        return RecommendationResponse(items=items[:request.n_recommendations],
                                      scores=scores[:request.n_recommendations])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))