"""FastAPI application exposing recommendation endpoints."""

from __future__ import annotations

import sys
import threading
import typing

from typing_extensions import TypedDict as _TypedDict

if sys.version_info < (3, 12):
    typing.TypedDict = _TypedDict  # type: ignore[attr-defined]

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from src.shl_recommender.agent.chat_agent import ChatAgent
from src.shl_recommender.data_models import (
    ASSESSMENT_TYPE_LABELS,
    ChatRequest,
    ChatResponse,
    ExtractedCategory,
    RecommendationRequest,
    RecommendationResponse,
)
from src.shl_recommender.recommender import RecommendationEngine

_engine: RecommendationEngine | None = None
_chat_agent: ChatAgent | None = None
_engine_lock = threading.Lock()


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RecommendationEngine()
    return _engine


def get_chat_agent() -> ChatAgent:
    global _chat_agent
    if _chat_agent is None:
        with _engine_lock:
            if _chat_agent is None:
                _chat_agent = ChatAgent(engine=get_engine())
    return _chat_agent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    global _engine, _chat_agent
    _engine = None
    _chat_agent = None


app = FastAPI(title="SHL Assessment Recommender", version="2.0.0", lifespan=lifespan)


@app.get("/health")
def health_check() -> dict:
    """Liveness/readiness probe required by the SHL assignment evaluator."""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    result = get_engine().recommend(query)
    categories = [
        ExtractedCategory(code=code, label=ASSESSMENT_TYPE_LABELS.get(code, code))
        for code in result.extracted_types
    ]
    return RecommendationResponse(
        recommended_assessments=result.recommendations,
        extracted_categories=categories,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Stateless conversational endpoint; pass full message history each turn."""
    return get_chat_agent().handle(request.messages)
