"""The evaluation harness. Writes results/eval.json.

    python -m ragsvc.evaluate --project <id> [--n-retrieval 500] [--n-generation 200]

Retrieval (all sampled questions, each retriever: bm25, dense = BigQuery VECTOR_SEARCH,
dense_local = the same vectors in memory, hybrid): recall@1, recall@5, MRR, latency per query.
Generation (a subset, several arms): exact match and F1 against SQuAD's gold answers,
"unknown" rate, LLM-judged faithfulness, tokens, latency and list-price cost per query.
Arms: hybrid / dense / bm25 retrieval with k=5, `oracle` (the gold paragraph only, an upper
bound for the generator) and `closed` (no paragraphs, what the model knows on its own).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config
from .data import load_paragraphs, load_questions
from .generate import answer, judge
from .retrieve import get_retriever
from .text import exact_match, f1

# Vertex AI list prices (USD), used only for the cost-per-query estimate.
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.30, 2.50        # gemini-2.5-flash text
PRICE_EMBED_PER_M_CHARS = 0.025                       # text-embedding-005


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 1) if len(xs) else None


def eval_retrieval(kind: str, qs, k: int, project: str) -> dict:
    r = get_retriever(kind, project)
    hits1, hits5, rr, lat, embed_ms, vs_ms = [], [], [], [], [], []
    for n, q in enumerate(qs.itertuples(), 1):
        if n % 100 == 0:
            print(f"    {kind}: {n}/{len(qs)}", flush=True)
        t0 = time.perf_counter()
        res = r.search(q.question, k)
        lat.append((time.perf_counter() - t0) * 1000)
        ids = [x["id"] for x in res]
        rank = ids.index(q.paragraph_id) + 1 if q.paragraph_id in ids else None
        hits1.append(rank == 1)
        hits5.append(rank is not None)
        rr.append(1.0 / rank if rank else 0.0)
        t = getattr(getattr(r, "dense", r), "last_timings", {})
        if t:
            embed_ms.append(t["embed_ms"])
            vs_ms.append(t["vector_search_ms"])
    out = {"retriever": kind, "questions": len(qs), "k": k, "recall_at_1": round(float(np.mean(hits1)), 4),
           "recall_at_5": round(float(np.mean(hits5)), 4), "mrr": round(float(np.mean(rr)), 4),
           "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95)}}
    if embed_ms:
        out["latency_ms"]["embed_p50"] = pct(embed_ms, 50)
        out["latency_ms"]["vector_search_p50"] = pct(vs_ms, 50)
    print(f"  {kind:7s} R@1 {out['recall_at_1']:.3f}  R@5 {out['recall_at_5']:.3f}  MRR {out['mrr']:.3f}  "
          f"p50 {out['latency_ms']['p50']} ms", flush=True)
    return out


def eval_generation(arm: str, qs, paragraphs, k: int, project: str, with_judge: bool, workers: int) -> dict:
    r = get_retriever(arm, project) if arm in ("hybrid", "dense", "bm25") else None
    by_id = paragraphs.set_index("id")

    def one(q):
        if arm == "closed":
            ctx = []
        elif arm == "oracle":
            p = by_id.loc[q.paragraph_id]
            ctx = [{"id": int(q.paragraph_id), "title": str(p["title"]), "text": str(p["text"]), "score": 1.0}]
        else:
            ctx = r.search(q.question, k)
        a = answer(q.question, ctx)
        row = {"id": q.id, "pred": a["answer"], "golds": list(q.answers), "em": exact_match(a["answer"], q.answers),
               "f1": f1(a["answer"], q.answers), "unknown": a["answer"].strip().lower() == "unknown",
               "retrieved_gold": any(c["id"] == q.paragraph_id for c in ctx),
               "prompt_tokens": a["prompt_tokens"], "output_tokens": a["output_tokens"], "gen_ms": a["latency_ms"]}
        if with_judge and ctx and not row["unknown"]:
            j = judge(q.question, ctx, a["answer"])
            row["supported"] = j["supported"]
            row["judge_tokens"] = j["prompt_tokens"] + j["output_tokens"]
        return row

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, list(qs.itertuples())))
    wall = time.perf_counter() - t0
    n = len(rows)
    in_tok = sum(x["prompt_tokens"] for x in rows)
    out_tok = sum(x["output_tokens"] for x in rows)
    cost_gen = in_tok / 1e6 * PRICE_IN_PER_M + out_tok / 1e6 * PRICE_OUT_PER_M
    judged = [x for x in rows if "supported" in x]
    out = {"arm": arm, "questions": n, "k": k if r else 0,
           "exact_match": round(float(np.mean([x["em"] for x in rows])), 4),
           "f1": round(float(np.mean([x["f1"] for x in rows])), 4),
           "unknown_rate": round(float(np.mean([x["unknown"] for x in rows])), 4),
           "gold_paragraph_retrieved": round(float(np.mean([x["retrieved_gold"] for x in rows])), 4),
           "f1_when_gold_retrieved": round(float(np.mean([x["f1"] for x in rows if x["retrieved_gold"]] or [0])), 4),
           "f1_when_gold_missed": round(float(np.mean([x["f1"] for x in rows if not x["retrieved_gold"]] or [0])), 4),
           "judged_supported_rate": round(float(np.mean([x["supported"] for x in judged])), 4) if judged else None,
           "judged": len(judged),
           "tokens_per_query": {"prompt": round(in_tok / n, 1), "output": round(out_tok / n, 1)},
           "gen_latency_ms": {"p50": pct([x["gen_ms"] for x in rows], 50), "p95": pct([x["gen_ms"] for x in rows], 95)},
           "cost_usd_per_1k_queries": round(cost_gen / n * 1000, 4),
           "wall_seconds": round(wall, 1), "examples": rows[:8]}
    print(f"  {arm:7s} EM {out['exact_match']:.3f}  F1 {out['f1']:.3f}  unknown {out['unknown_rate']:.3f}  "
          f"gold-in-ctx {out['gold_paragraph_retrieved']:.3f}  supported {out['judged_supported_rate']}  "
          f"${out['cost_usd_per_1k_queries']}/1k", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=config.PROJECT)
    ap.add_argument("--n-retrieval", type=int, default=500)
    ap.add_argument("--n-generation", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--arms", default="hybrid,dense,bm25,oracle,closed")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()
    qs = load_questions(eval_only=True)
    paragraphs = load_paragraphs()
    q_ret, q_gen = qs.head(a.n_retrieval), qs.head(a.n_generation)
    print(f"retrieval on {len(q_ret)} questions, generation on {len(q_gen)}")
    out = {"config": {"embed_model": config.EMBED_MODEL, "gen_model": config.GEN_MODEL, "k": a.k,
                      "paragraphs": int(len(paragraphs)), "n_retrieval": len(q_ret), "n_generation": len(q_gen)},
           "retrieval": [eval_retrieval(kind, q_ret, a.k, a.project) for kind in ("bm25", "dense", "dense_local", "hybrid")],
           "generation": [eval_generation(arm, q_gen, paragraphs, a.k, a.project, not a.no_judge, a.workers)
                          for arm in a.arms.split(",")]}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "eval.json").write_text(json.dumps(out, indent=2))
    print("wrote results/eval.json")


if __name__ == "__main__":
    main()
