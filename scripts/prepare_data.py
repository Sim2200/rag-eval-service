"""Download and prepare SQuAD dataset for RAG service."""

import pandas as pd
from pathlib import Path
import datasets
from ragsvc.config import DATA_DIR


def prepare_data():
    """Download SQuAD v1.1 validation split and create parquet files."""
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)

    # Download SQuAD v1.1 validation split
    dataset = datasets.load_dataset(
        "rajpurkar/squad",
        split="validation",
        cache_dir=DATA_DIR / "hf_cache"
    )

    # Build paragraphs.parquet: one row per distinct (title, context) pair
    paragraphs_data = []
    paragraph_seen = {}
    para_counter = 0

    for example in dataset:
        title = example["title"]
        context = example["context"]
        key = (title, context)

        if key not in paragraph_seen:
            paragraph_seen[key] = para_counter
            paragraphs_data.append({
                "id": para_counter,
                "title": title,
                "text": context
            })
            para_counter += 1

    paragraphs_df = pd.DataFrame(paragraphs_data)
    paragraphs_df.to_parquet(DATA_DIR / "paragraphs.parquet", index=False)
    print(f"paragraphs.parquet: {len(paragraphs_df)} rows")

    # Build questions.parquet
    questions_data = []
    for example in dataset:
        title = example["title"]
        context = example["context"]
        paragraph_id = paragraph_seen[(title, context)]
        question_id = example["id"]
        question = example["question"]

        # Get distinct answer texts
        answers = example["answers"]
        answer_texts = list(dict.fromkeys(answers["text"]))  # Remove duplicates, preserve order

        questions_data.append({
            "id": question_id,
            "paragraph_id": paragraph_id,
            "question": question,
            "answers": answer_texts
        })

    questions_df = pd.DataFrame(questions_data)
    questions_df.to_parquet(DATA_DIR / "questions.parquet", index=False)
    print(f"questions.parquet: {len(questions_df)} rows")

    # Build eval_questions.parquet: seeded random sample of 500 rows
    eval_questions_df = questions_df.sample(n=500, random_state=0)
    eval_questions_df.to_parquet(DATA_DIR / "eval_questions.parquet", index=False)
    print(f"eval_questions.parquet: {len(eval_questions_df)} rows")


if __name__ == "__main__":
    prepare_data()
