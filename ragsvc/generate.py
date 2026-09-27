"""Answer generation and LLM judging with Gemini on Vertex AI.

`answer(question, contexts)` asks for a short extractive answer grounded in the given
paragraphs (or "unknown" when they do not contain it) and returns the text with token
counts and latency. `judge(question, contexts, answer_text)` asks the model whether the
answer is supported by the contexts (faithfulness, 0/1) with a one-line reason; it is
used by the evaluation harness alongside the exact-match/F1 scores against gold answers,
never instead of them.
"""

from __future__ import annotations

import json
import re
import threading
import time

from google import genai
from google.genai import types

from . import config

_LOCAL = threading.local()   # one client per thread: the client is not safe to share across threads

ANSWER_PROMPT = """You answer questions using only the paragraphs below.
Reply with the shortest exact phrase from the paragraphs that answers the question (a name, number, date or short span).
If the paragraphs do not contain the answer, reply exactly: unknown

{contexts}

Question: {question}
Answer:"""

JUDGE_PROMPT = """Paragraphs:
{contexts}

Question: {question}
Proposed answer: {answer}

Is the proposed answer fully supported by the paragraphs above? Reply with JSON only: {{"supported": true or false, "reason": "<one short sentence>"}}"""


def client(project: str = config.PROJECT) -> genai.Client:
    if getattr(_LOCAL, "client", None) is None:
        _LOCAL.client = genai.Client(vertexai=True, project=project, location=config.LOCATION)
    return _LOCAL.client


def format_contexts(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] {c['title']}: {c['text']}" for i, c in enumerate(contexts))


def _generate(prompt: str, max_tokens: int) -> tuple[str, int, int, float]:
    t0 = time.perf_counter()
    for attempt in range(7):  # 429 (per-minute token quota) and 5xx: back off up to about a minute
        try:
            r = client().models.generate_content(
                model=config.GEN_MODEL, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_tokens,
                                                   thinking_config=types.ThinkingConfig(thinking_budget=0)))
            break
        except Exception:  # noqa: BLE001
            if attempt == 6:
                raise
            time.sleep(min(60, 2 ** attempt))
    ms = (time.perf_counter() - t0) * 1000
    u = r.usage_metadata
    return (r.text or "").strip(), int(u.prompt_token_count or 0), int(u.candidates_token_count or 0), ms


def answer(question: str, contexts: list[dict]) -> dict:
    text, p, o, ms = _generate(ANSWER_PROMPT.format(contexts=format_contexts(contexts), question=question), 64)
    return {"answer": text.split("\n")[0].strip().strip('"'), "prompt_tokens": p, "output_tokens": o,
            "latency_ms": round(ms, 1)}


def judge(question: str, contexts: list[dict], answer_text: str) -> dict:
    text, p, o, ms = _generate(JUDGE_PROMPT.format(contexts=format_contexts(contexts), question=question,
                                                   answer=answer_text), 96)
    m = re.search(r"\{.*\}", text, re.S)
    try:
        parsed = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        parsed = {}
    return {"supported": bool(parsed.get("supported", False)), "reason": str(parsed.get("reason", text))[:200],
            "prompt_tokens": p, "output_tokens": o, "latency_ms": round(ms, 1)}
