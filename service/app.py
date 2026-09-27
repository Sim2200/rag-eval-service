"""FastAPI application for RAG service."""

import time
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="RAG Eval Service")

# Module-level cache for retrievers
_retriever_cache = {}


def get_retriever(kind: str):
    """Get a retriever instance by kind, creating and caching if needed.

    This wrapping function allows tests to monkeypatch retriever creation.
    """
    if kind not in _retriever_cache:
        from ragsvc.retrieve import get_retriever as _get_retriever
        _retriever_cache[kind] = _get_retriever(kind)
    return _retriever_cache[kind]


def answer(question: str, contexts: list[dict]) -> dict:
    """Generate an answer from contexts.

    This wrapping function allows tests to monkeypatch answer generation.
    """
    from ragsvc.generate import answer as _answer
    return _answer(question, contexts)


class AskRequest(BaseModel):
    """Request body for /ask endpoint."""
    question: str
    retriever: Literal["dense", "bm25", "hybrid"] = Field(default="hybrid")
    k: int = Field(default=5)


class ContextItem(BaseModel):
    """Context item in response."""
    id: int
    title: str
    score: float


class TimingsMS(BaseModel):
    """Timing measurements in milliseconds."""
    retrieve: float
    generate: float
    total: float


class UsageMetrics(BaseModel):
    """Token usage metrics."""
    prompt_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    """Response body for /ask endpoint."""
    answer: str
    contexts: list[ContextItem]
    timings_ms: TimingsMS
    usage: UsageMetrics


@app.get("/health")  # not /healthz: Cloud Run's front end answers that path itself with a 404
def healthz():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):  # sync on purpose: the retrievers and the Gemini client block, so FastAPI runs this in its thread pool
    """Ask a question and retrieve relevant contexts with generated answer."""
    total_start = time.perf_counter()

    # Retrieve contexts
    retrieve_start = time.perf_counter()
    retriever = get_retriever(request.retriever)
    search_results = retriever.search(request.question, request.k)
    retrieve_time = (time.perf_counter() - retrieve_start) * 1000

    # Generate answer
    generate_start = time.perf_counter()
    answer_data = answer(request.question, search_results)
    generate_time = (time.perf_counter() - generate_start) * 1000

    total_time = (time.perf_counter() - total_start) * 1000

    # Extract context information
    contexts = []
    for ctx in search_results:
        contexts.append(ContextItem(
            id=ctx["id"],
            title=ctx["title"],
            score=ctx["score"]
        ))

    return AskResponse(
        answer=answer_data["answer"],
        contexts=contexts,
        timings_ms=TimingsMS(
            retrieve=retrieve_time,
            generate=generate_time,
            total=total_time
        ),
        usage=UsageMetrics(
            prompt_tokens=answer_data["prompt_tokens"],
            output_tokens=answer_data["output_tokens"]
        )
    )
