"""Tests for FastAPI service."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from service.app import app
    return TestClient(app)


def test_healthz(client):
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_with_mocked_retriever_and_generator(client):
    """Test /ask endpoint with mocked retriever and generator."""
    mock_contexts = [
        {"id": 0, "title": "Test Title", "text": "Test context", "score": 0.95},
        {"id": 1, "title": "Test Title 2", "text": "Another context", "score": 0.85}
    ]

    mock_answer_data = {
        "answer": "Test answer",
        "prompt_tokens": 100,
        "output_tokens": 20,
        "latency_ms": 500
    }

    with patch("service.app.get_retriever") as mock_get_retriever:
        with patch("service.app.answer") as mock_answer:
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = mock_contexts
            mock_get_retriever.return_value = mock_retriever

            mock_answer.return_value = mock_answer_data

            response = client.post(
                "/ask",
                json={"question": "What is the capital of France?", "retriever": "hybrid", "k": 2}
            )

            assert response.status_code == 200
            data = response.json()

            assert data["answer"] == "Test answer"
            assert len(data["contexts"]) == 2
            assert data["contexts"][0]["id"] == 0
            assert data["contexts"][0]["title"] == "Test Title"
            assert data["contexts"][0]["score"] == 0.95
            assert "timings_ms" in data
            assert "retrieve" in data["timings_ms"]
            assert "generate" in data["timings_ms"]
            assert "total" in data["timings_ms"]
            assert data["usage"]["prompt_tokens"] == 100
            assert data["usage"]["output_tokens"] == 20


def test_ask_with_default_retriever(client):
    """Test /ask endpoint with default retriever type."""
    mock_contexts = [{"id": 0, "title": "Test", "text": "Context", "score": 0.9}]
    mock_answer_data = {
        "answer": "Answer",
        "prompt_tokens": 50,
        "output_tokens": 10,
        "latency_ms": 300
    }

    with patch("service.app.get_retriever") as mock_get_retriever:
        with patch("service.app.answer") as mock_answer:
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = mock_contexts
            mock_get_retriever.return_value = mock_retriever

            mock_answer.return_value = mock_answer_data

            response = client.post("/ask", json={"question": "Test question"})

            assert response.status_code == 200
            data = response.json()
            # Default k is 5 and retriever is hybrid
            mock_retriever.search.assert_called_once_with("Test question", 5)
            mock_get_retriever.assert_called_once_with("hybrid")
