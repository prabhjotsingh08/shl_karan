"""FastAPI application exposing recommendation endpoints."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.shl_recommender.data_models import RecommendationRequest, RecommendationResponse
from src.shl_recommender.recommender import RecommendationEngine


app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")
engine = RecommendationEngine()


@app.get("/health", response_model=dict)
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    recommendations = engine.recommend(query)
    return RecommendationResponse(recommended_assessments=recommendations)

