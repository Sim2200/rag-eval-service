"""Embed the paragraphs with Vertex AI and load them into BigQuery for vector search.

    python -m ragsvc.index --project <id>

Creates `<project>.rag.paragraphs` (id, title, text, embedding ARRAY<FLOAT64>) with
text-embedding-005 vectors (768 dims, task type RETRIEVAL_DOCUMENT), replacing any
previous table, and writes results/index.json with the embedding time and token count.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
from google import genai
from google.cloud import bigquery
from google.genai import types

from . import config
from .data import load_paragraphs

BATCH = 40  # the embedding endpoint caps a request at 20,000 tokens


def embed(client: genai.Client, texts: list[str], task: str) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        r = client.models.embed_content(model=config.EMBED_MODEL, contents=texts[i:i + BATCH],
                                        config=types.EmbedContentConfig(task_type=task))
        out.extend(e.values for e in r.embeddings)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=config.PROJECT)
    a = ap.parse_args()
    client = genai.Client(vertexai=True, project=a.project, location=config.LOCATION)
    bq = bigquery.Client(project=a.project)
    paragraphs = load_paragraphs()
    t0 = time.perf_counter()
    vectors = embed(client, paragraphs["text"].tolist(), "RETRIEVAL_DOCUMENT")
    embed_seconds = time.perf_counter() - t0
    df = pd.DataFrame({"id": paragraphs["id"].astype(int), "title": paragraphs["title"], "text": paragraphs["text"],
                       "embedding": vectors})
    table = f"{a.project}.{config.DATASET}.{config.TABLE}"
    schema = [bigquery.SchemaField("id", "INT64"), bigquery.SchemaField("title", "STRING"),
              bigquery.SchemaField("text", "STRING"), bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED")]
    t1 = time.perf_counter()
    bq.load_table_from_dataframe(df, table, job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")).result()
    load_seconds = time.perf_counter() - t1
    words = int(paragraphs["text"].str.split().str.len().sum())
    out = {"table": table, "paragraphs": int(len(df)), "dims": len(vectors[0]), "embed_model": config.EMBED_MODEL,
           "embed_seconds": round(embed_seconds, 1), "paragraphs_per_second": round(len(df) / embed_seconds, 1),
           "words_embedded": words, "bigquery_load_seconds": round(load_seconds, 1)}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "index.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
