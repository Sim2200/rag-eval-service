"""Three retrievers over the same paragraphs, one interface: `search(question, k) -> [{id, title, text, score}]`.

- dense:  the question is embedded with text-embedding-005 (RETRIEVAL_QUERY) and matched against the
          stored document vectors with BigQuery `VECTOR_SEARCH` (cosine distance, brute force: the
          table has ~2,000 rows, so no index is needed).
- dense_local: the same vectors, loaded from BigQuery once and searched in memory (numpy dot
          product on normalised vectors); the shape a serving process wants.
- bm25:   Okapi BM25 over the paragraph tokens, in memory (rank_bm25).
- hybrid: reciprocal rank fusion of the dense_local and BM25 rankings (top 2k from each, fused, top k).

`get_retriever(kind)` builds one lazily and caches it.
"""

from __future__ import annotations

import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from . import config
from .data import load_paragraphs
from .text import rrf

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


class BM25Retriever:
    kind = "bm25"

    def __init__(self):
        self.paragraphs = load_paragraphs().set_index("id")
        self.ids = self.paragraphs.index.to_numpy()
        self.bm25 = BM25Okapi([tokenize(t) for t in self.paragraphs["text"]])

    def search(self, question: str, k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(question))
        top = np.argsort(-scores)[:k]
        return [self._row(int(self.ids[i]), float(scores[i])) for i in top]

    def _row(self, pid: int, score: float) -> dict:
        p = self.paragraphs.loc[pid]
        return {"id": pid, "title": str(p["title"]), "text": str(p["text"]), "score": score}


class DenseRetriever:
    kind = "dense"

    def __init__(self, project: str = config.PROJECT):
        from google import genai
        from google.cloud import bigquery
        self.client = genai.Client(vertexai=True, project=project, location=config.LOCATION)
        self.bq = bigquery.Client(project=project)
        self.table = f"`{project}.{config.DATASET}.{config.TABLE}`"
        self.last_timings: dict[str, float] = {}

    def embed(self, question: str) -> list[float]:
        from google.genai import types
        r = self.client.models.embed_content(model=config.EMBED_MODEL, contents=[question],
                                             config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
        return r.embeddings[0].values

    def search(self, question: str, k: int = 5) -> list[dict]:
        from google.cloud import bigquery
        t0 = time.perf_counter()
        q = self.embed(question)
        t1 = time.perf_counter()
        sql = f"""
            SELECT base.id AS id, base.title AS title, base.text AS text, distance
            FROM VECTOR_SEARCH(TABLE {self.table}, 'embedding', (SELECT @q AS embedding),
                               top_k => @k, distance_type => 'COSINE')
            ORDER BY distance"""
        cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("q", "FLOAT64", q),
                                                        bigquery.ScalarQueryParameter("k", "INT64", k)])
        for attempt in range(2):  # a query job occasionally never returns; give it 60 s, then retry once
            try:
                rows = list(self.bq.query(sql, job_config=cfg).result(timeout=60))
                break
            except Exception:  # noqa: BLE001
                if attempt:
                    raise
        out = [{"id": int(r.id), "title": r.title, "text": r.text, "score": 1.0 - float(r.distance)} for r in rows]
        self.last_timings = {"embed_ms": round((t1 - t0) * 1000, 1), "vector_search_ms": round((time.perf_counter() - t1) * 1000, 1)}
        return out


class DenseLocalRetriever(DenseRetriever):
    kind = "dense_local"

    def __init__(self, project: str = config.PROJECT):
        super().__init__(project)
        rows = list(self.bq.query(f"SELECT id, title, text, embedding FROM {self.table}").result())
        self.ids = np.array([r.id for r in rows])
        self.meta = {int(r.id): (r.title, r.text) for r in rows}
        m = np.array([r.embedding for r in rows], dtype=np.float32)
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)

    def search(self, question: str, k: int = 5) -> list[dict]:
        t0 = time.perf_counter()
        q = np.array(self.embed(question), dtype=np.float32)
        q /= np.linalg.norm(q)
        t1 = time.perf_counter()
        sims = self.matrix @ q
        top = np.argsort(-sims)[:k]
        self.last_timings = {"embed_ms": round((t1 - t0) * 1000, 1), "vector_search_ms": round((time.perf_counter() - t1) * 1000, 2)}
        return [{"id": int(self.ids[i]), "title": self.meta[int(self.ids[i])][0], "text": self.meta[int(self.ids[i])][1],
                 "score": float(sims[i])} for i in top]


class HybridRetriever:
    kind = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever):
        self.dense, self.bm25 = dense, bm25

    def search(self, question: str, k: int = 5) -> list[dict]:
        d = self.dense.search(question, 2 * k)
        b = self.bm25.search(question, 2 * k)
        by_id = {r["id"]: r for r in d + b}
        fused = rrf([[r["id"] for r in d], [r["id"] for r in b]])[:k]
        return [{**by_id[i], "score": 1.0 / (1 + rank)} for rank, i in enumerate(fused)]


_CACHE: dict[str, object] = {}


def get_retriever(kind: str, project: str = config.PROJECT):
    if kind not in _CACHE:
        if kind == "bm25":
            _CACHE[kind] = BM25Retriever()
        elif kind == "dense":
            _CACHE[kind] = DenseRetriever(project)
        elif kind == "dense_local":
            _CACHE[kind] = DenseLocalRetriever(project)
        elif kind == "hybrid":
            _CACHE[kind] = HybridRetriever(get_retriever("dense_local", project), get_retriever("bm25"))
        else:
            raise ValueError(f"unknown retriever {kind!r}")
    return _CACHE[kind]
