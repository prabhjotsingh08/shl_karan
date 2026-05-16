"""Shared pytest fixtures for API and agent tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app, get_engine


@pytest.fixture(scope="module")
def client() -> TestClient:
    """HTTP client with recommendation engine warmed up once per module."""
    get_engine()
    return TestClient(app)
