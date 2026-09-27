"""Configuration for RAG service."""

import os
from pathlib import Path

PROJECT = os.environ.get("GCP_PROJECT", "project-1adf2361-a5bc-4d4a-a7f")
LOCATION = "us-central1"
DATASET = "rag"
TABLE = "paragraphs"
EMBED_MODEL = "text-embedding-005"
GEN_MODEL = "gemini-2.5-flash"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
