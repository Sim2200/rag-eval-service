.PHONY: setup data test index eval deploy loadtest down

PY := .venv/bin/python
GCP_PROJECT ?= your-gcp-project
REGION := us-central1
SERVICE := rag-eval-service

## setup: Initialize virtual environment and install dependencies
setup:
	uv venv -q -p 3.11 .venv && uv pip install -q -p .venv/bin/python -r requirements-dev.txt

## data: Download and prepare SQuAD dataset
data:
	$(PY) scripts/prepare_data.py

## test: Run pytest tests
test:
	$(PY) -m pytest -q

## index: Index paragraphs to BigQuery and create embeddings
index:
	$(PY) -m ragsvc.index --project $(GCP_PROJECT)

## eval: Evaluate retriever and generator on eval_questions
eval:
	$(PY) -m ragsvc.evaluate --project $(GCP_PROJECT)

## deploy: Deploy service to Cloud Run
deploy:
	gcloud run deploy $(SERVICE) --source . --region $(REGION) --project $(GCP_PROJECT) --allow-unauthenticated --memory 1Gi --cpu 1 --min-instances 0 --max-instances 4 --set-env-vars GCP_PROJECT=$(GCP_PROJECT)

## loadtest: Run load test against deployed service
loadtest:
	$(PY) scripts/loadtest.py --url $$(gcloud run services describe $(SERVICE) --region $(REGION) --project $(GCP_PROJECT) --format='value(status.url)')

## down: Delete service from Cloud Run
down:
	gcloud run services delete $(SERVICE) --region $(REGION) --project $(GCP_PROJECT) --quiet
