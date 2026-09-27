# RAG question answering with an evaluation harness

![Vertex AI](https://img.shields.io/badge/Vertex_AI-Gemini_%C2%B7_text--embedding--005-4285F4?logo=googlecloud&logoColor=white)
![BigQuery](https://img.shields.io/badge/BigQuery-VECTOR__SEARCH-669DF6?logo=googlebigquery&logoColor=white)
![Cloud Run](https://img.shields.io/badge/Cloud_Run-FastAPI-4285F4?logo=googlecloud&logoColor=white)
![Tests](https://img.shields.io/badge/pytest-20_passing-brightgreen)

A retrieval-augmented question-answering service over Wikipedia paragraphs, built to answer one
question with numbers rather than opinions: **which retriever, and does the retrieved context help?**
SQuAD's development set is the corpus because every question has gold answers, so answers are
scored by exact match and F1, not only by another model's opinion. An LLM judge is used for what
gold answers cannot measure (is the answer supported by the retrieved text?), never instead of them.

| | |
|---|---|
| **Corpus** | SQuAD v1.1 dev: 2,067 paragraphs from 48 Wikipedia articles (254k words), 10,570 questions with gold answer spans; 500 questions sampled (seed 0) for retrieval, the first 200 of them for generation. |
| **Retrievers** | `bm25` (Okapi BM25 in memory) · `dense` (text-embedding-005 vectors, BigQuery `VECTOR_SEARCH`) · `dense_local` (the same vectors loaded once and searched in memory) · `hybrid` (reciprocal rank fusion of dense_local and BM25). |
| **Generator** | Gemini 2.5 Flash on Vertex AI, temperature 0, thinking off, asked for the shortest exact phrase or `unknown`. Arms: the three retrievers with k=5, `oracle` (the gold paragraph only, an upper bound) and `closed` (no paragraphs). |
| **Headline** | Hybrid retrieval finds the gold paragraph in the top 5 for **94.6%** of questions (BM25 91.0%, dense 88.6%). With it, Gemini reaches **EM 0.78 / F1 0.86**, against 0.81 / 0.90 with the gold paragraph handed to it and **0.09 / 0.14 with no retrieval** (it answers `unknown` 72% of the time). The judge finds 98.9% of answers supported. **$0.29 per 1,000 questions**, 370 ms p50 per answer. |
| **Service** | FastAPI on Cloud Run: `/ask` returns the answer, the contexts and per-stage timings. Measured under load from a laptop: making the route synchronous took throughput at concurrency 8 from **2.4 to 12.0 requests/s** on one 1-vCPU instance, p50 0.6 s. |
| **Stack** | Python 3.11 · google-genai · BigQuery · rank-bm25 · FastAPI · Cloud Run (`gcloud run deploy --source`) · pytest · GitHub Actions |

## How it works

```
questions ──► retriever ──► k paragraphs ──► Gemini (answer prompt) ──► answer
                 │                              │
   bm25 · dense (BigQuery) · dense_local · hybrid (RRF)   judge prompt: "supported?"
                                                          exact match / F1 vs gold
```

- **Index** (`ragsvc/index.py`): the 2,067 paragraphs are embedded with `text-embedding-005`
  (task type `RETRIEVAL_DOCUMENT`, batches of 40 because the endpoint caps a request at 20k tokens)
  in 40 s and loaded into `rag.paragraphs` in BigQuery with an `ARRAY<FLOAT64>` column.
- **Retrievers** (`ragsvc/retrieve.py`): one interface, `search(question, k)`. `dense` runs
  BigQuery `VECTOR_SEARCH` (cosine, brute force; no vector index at this size). `dense_local` pulls
  the same vectors once at startup (2,067 × 768 floats, 6 MB) and does a normalised dot product in
  numpy. `hybrid` takes the top 2k from `dense_local` and `bm25` and fuses them with reciprocal rank
  fusion (`ragsvc/text.py`).
- **Generation and judging** (`ragsvc/generate.py`): the answer prompt lists the paragraphs and asks
  for the shortest exact phrase, or `unknown` when the paragraphs do not contain it. The judge prompt
  asks whether the proposed answer is fully supported by the paragraphs and returns JSON. One client
  per thread; retries with exponential backoff on the per-minute token quota (429).
- **Harness** (`ragsvc/evaluate.py`): recall@1, recall@5 and MRR per retriever on 500 questions,
  then per arm on 200 questions: exact match and F1 with SQuAD's official normalisation
  (`ragsvc/text.py`, unit-tested against hand-computed cases), `unknown` rate, whether the gold
  paragraph was in the context, F1 split by that, judge support rate, tokens and list-price cost.
  Retrieval results are saved before generation starts, so a generation failure never loses them.

## Results

Measured 2026-09-27, `results/eval.json`, `results/index.json`, `results/loadtest_*.json`.

### Retrieval (500 questions, k = 5)

| Retriever | Recall@1 | Recall@5 | MRR | p50 latency | Where the time goes |
|---|---|---|---|---|---|
| bm25 | 0.750 | 0.910 | 0.811 | **4.8 ms** | tokenise + score 2,067 paragraphs in memory |
| dense (BigQuery VECTOR_SEARCH) | 0.624 | 0.886 | 0.728 | 877 ms | ~120 ms embedding call + a BigQuery query job per question |
| dense_local (same vectors, numpy) | 0.624 | 0.886 | 0.728 | 121 ms | 120 ms embedding call + **0.4 ms** search |
| **hybrid (RRF of dense_local + bm25)** | **0.758** | **0.946** | **0.836** | 121 ms | the embedding call; fusion is free |

Three things worth knowing. BM25 beats dense embeddings on this corpus at every cut-off: SQuAD
questions were written by people looking at the paragraph, so they reuse its exact words, and
lexical matching is at home. Dense retrieval still adds something BM25 lacks (paraphrase), and
fusing the two gives the best of both, +3.6 points of recall@5 over BM25 and +6 over dense, at no
extra cost because both rankings are already computed. And BigQuery `VECTOR_SEARCH` returns the
identical ranking to the in-memory search (same vectors, same cosine) but pays ~750 ms of query-job
overhead per call; it is the right tool for batch or analytical joins over embeddings, not for a
request path, which is why the service uses `dense_local`.

### Generation (200 questions, Gemini 2.5 Flash, k = 5)

| Arm | Exact match | F1 | `unknown` | Gold paragraph in context | F1 when gold in context / not | Judge: supported | Prompt tokens | p50 | Cost / 1k questions |
|---|---|---|---|---|---|---|---|---|---|
| **hybrid** | **0.780** | **0.858** | 5.5% | 95.0% | 0.897 / 0.125 | 98.9% (189 judged) | 941 | 370 ms | $0.29 |
| bm25 | 0.775 | 0.850 | 7.5% | 93.0% | 0.902 / 0.161 | 99.5% | 942 | 373 ms | $0.29 |
| dense | 0.740 | 0.816 | 7.5% | 90.5% | 0.884 / 0.171 | 99.5% | 935 | 383 ms | $0.29 |
| oracle (gold paragraph only) | 0.810 | 0.902 | 4.0% | 100% | 0.902 / – | 97.9% | 232 | 362 ms | $0.08 |
| closed (no paragraphs) | 0.090 | 0.140 | 72.0% | – | – | – | 66 | 380 ms | $0.02 |

Reading the numbers. The generator is not the bottleneck: given the right paragraph it scores F1
0.90, and with hybrid retrieval it scores 0.90 on the 95% of questions where the gold paragraph was
retrieved and 0.13 on the 5% where it was not, so essentially all of the gap to the oracle is
retrieval misses. The closed-book arm is the control that makes the case for retrieval: the same
model with no context answers `unknown` 72% of the time and gets F1 0.14, so the paragraphs are
doing the work, not the model's memory. The judge agrees with the gold answers where it should: it
marks 98.9% of the hybrid arm's answers as supported, and the few it rejects are mostly the model
answering from a distractor paragraph on the same topic. Cost is dominated by the ~940 prompt tokens
of five paragraphs; the oracle arm with one paragraph is 3.6× cheaper for +4 F1, which is the
argument for a reranker or k=3 as the next experiment.

Honest notes. 200 questions means ±3 points on EM at 95% confidence; the ordering hybrid > bm25 >
dense is consistent with the retrieval recall but the hybrid–bm25 gap is within noise. The
`dense` (BigQuery) latency row was measured in the first full run, whose generation stage then
failed on a client shared across threads; retrieval numbers were identical on the rerun, and the
harness now saves them before generation. The `closed` arm's judge is not applicable (no contexts).

### The service under load (Cloud Run, 1 vCPU, 1 GiB, 60 requests per level, hybrid retriever)

| Concurrency | `async` route (blocking calls on the event loop) | plain `def` route (FastAPI thread pool) |
|---|---|---|
| 1 | p50 549 ms · 1.7 req/s | p50 550 ms · 1.8 req/s |
| 4 | p50 1,664 ms · 2.2 req/s | p50 552 ms · 4.5 req/s |
| 8 | p50 3,293 ms · **2.4 req/s** | p50 597 ms · **12.0 req/s** |

Server-side time per request is ~110 ms retrieval (the embedding call) + ~350 ms generation in both
cases. With the route declared `async`, every blocking client call held the event loop, so the
instance served requests one at a time and throughput froze at 1 / 0.46 s; declaring it `def` lets
FastAPI run it in its thread pool and the same instance serves five times as many requests with
the same p50. Zero errors at all levels. Cold start is 16 s (BM25 index and vectors are built on the
first request); `min-instances 1` removes it for about $8 a month. Note for Cloud Run: the front end
answers `/healthz` itself with a 404, so the health route is `/health`.

### Cost of the whole experiment

Embedding 254k words once: under a cent. The evaluation's ~1,700 Gemini calls (5 arms × 200
answers + 750 judgements): about $0.40. Cloud Run and BigQuery: cents. The service was deleted
after the load test; the BigQuery table (2,067 rows) was kept.

## Running it

```bash
make setup                      # uv venv + requirements-dev.txt
make data                       # SQuAD dev -> data/{paragraphs,questions,eval_questions}.parquet
make test                       # 20 tests, no network
make index GCP_PROJECT=<id>     # embed + load rag.paragraphs into BigQuery (needs the Vertex AI and BigQuery APIs)
make eval  GCP_PROJECT=<id>     # -> results/eval.json (about 30 minutes; ~$0.40 of Gemini)
make deploy GCP_PROJECT=<id>    # Cloud Run from source; prints the URL
make loadtest GCP_PROJECT=<id>  # -> results/loadtest.json
make down  GCP_PROJECT=<id>     # delete the service
```

Local credentials come from `gcloud auth application-default login`; the service uses the
project's default service account (Vertex AI user + BigQuery job user/data viewer).

## Layout

```
ragsvc/       config.py · data.py · text.py (normalise, EM, F1, RRF) · index.py · retrieve.py · generate.py · evaluate.py
service/      app.py (FastAPI: /health, /ask)        Dockerfile at the repo root (Cloud Run source deploy)
scripts/      prepare_data.py · loadtest.py
tests/        test_text.py · test_service.py (mocked retriever and generator)
results/      eval.json · eval_retrieval.json · index.json · loadtest_serial.json · loadtest_threaded.json
```

## Author

**Simran Kharbanda** · [github.com/Sim2200](https://github.com/Sim2200)
