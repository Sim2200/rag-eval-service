"""Data loading utilities."""

import pandas as pd
from ragsvc.config import DATA_DIR


def load_paragraphs() -> pd.DataFrame:
    """Load paragraphs from parquet file."""
    return pd.read_parquet(DATA_DIR / "paragraphs.parquet")


def load_questions(eval_only: bool = True) -> pd.DataFrame:
    """Load questions from parquet file.

    Args:
        eval_only: If True, load eval_questions.parquet; else load questions.parquet
    """
    if eval_only:
        return pd.read_parquet(DATA_DIR / "eval_questions.parquet")
    else:
        return pd.read_parquet(DATA_DIR / "questions.parquet")
