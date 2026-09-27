FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ragsvc/ ragsvc/
COPY service/ service/
COPY data/paragraphs.parquet data/
COPY data/eval_questions.parquet data/

ENV PYTHONPATH=/app

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
