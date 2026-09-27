"""Load test script for RAG service."""

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
import pandas as pd
from ragsvc.config import RESULTS_DIR


def load_test_questions(url: str, num_requests: int = 60):
    """Run load test against the service.

    Args:
        url: Base URL of the service (e.g., https://service.run.app)
        num_requests: Number of requests to send in total
    """
    # Load evaluation questions
    questions_df = pd.read_parquet(Path("data") / "eval_questions.parquet")

    results = {
        "1": {"requests": num_requests, "p50_ms": 0, "p95_ms": 0, "max_ms": 0, "throughput_rps": 0, "errors": 0, "mean_retrieve_ms": 0, "mean_generate_ms": 0},
        "4": {"requests": num_requests, "p50_ms": 0, "p95_ms": 0, "max_ms": 0, "throughput_rps": 0, "errors": 0, "mean_retrieve_ms": 0, "mean_generate_ms": 0},
        "8": {"requests": num_requests, "p50_ms": 0, "p95_ms": 0, "max_ms": 0, "throughput_rps": 0, "errors": 0, "mean_retrieve_ms": 0, "mean_generate_ms": 0},
    }

    for concurrency in [1, 4, 8]:
        latencies = []
        retrieve_times = []
        generate_times = []
        errors = 0

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            def make_request():
                try:
                    # Select a random question
                    question_row = questions_df.iloc[random.randint(0, len(questions_df) - 1)]
                    question = question_row["question"]

                    # Send request to service
                    start_time = time.perf_counter()
                    with httpx.Client(timeout=60) as client:
                        response = client.post(
                            f"{url}/ask",
                            json={"question": question, "retriever": "hybrid", "k": 5}
                        )
                    end_time = time.perf_counter()

                    if response.status_code == 200:
                        data = response.json()
                        latency_ms = (end_time - start_time) * 1000
                        latencies.append(latency_ms)
                        retrieve_times.append(data["timings_ms"]["retrieve"])
                        generate_times.append(data["timings_ms"]["generate"])
                    else:
                        errors += 1
                except Exception as e:
                    errors += 1

            # Submit all requests
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            for future in futures:
                future.result()

        if latencies:
            latencies.sort()
            results[str(concurrency)] = {
                "requests": num_requests,
                "p50_ms": statistics.median(latencies),
                "p95_ms": latencies[int(len(latencies) * 0.95)],
                "max_ms": max(latencies),
                "throughput_rps": num_requests / (latencies[-1] / 1000),
                "errors": errors,
                "mean_retrieve_ms": statistics.mean(retrieve_times) if retrieve_times else 0,
                "mean_generate_ms": statistics.mean(generate_times) if generate_times else 0,
            }
        else:
            results[str(concurrency)]["errors"] = num_requests

    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)

    # Write results to file
    with open(RESULTS_DIR / "loadtest.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print("Load test results:")
    for concurrency in [1, 4, 8]:
        result = results[str(concurrency)]
        print(f"\nConcurrency {concurrency}:")
        print(f"  Requests: {result['requests']}")
        print(f"  P50 latency: {result['p50_ms']:.2f}ms")
        print(f"  P95 latency: {result['p95_ms']:.2f}ms")
        print(f"  Max latency: {result['max_ms']:.2f}ms")
        print(f"  Throughput: {result['throughput_rps']:.2f} rps")
        print(f"  Errors: {result['errors']}")
        print(f"  Mean retrieve time: {result['mean_retrieve_ms']:.2f}ms")
        print(f"  Mean generate time: {result['mean_generate_ms']:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test RAG service")
    parser.add_argument("--url", required=True, help="Base URL of the service")
    parser.add_argument("--requests", type=int, default=60, help="Number of requests to send")
    args = parser.parse_args()

    load_test_questions(args.url, args.requests)
