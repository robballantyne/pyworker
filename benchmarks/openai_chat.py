"""
Benchmark for OpenAI-compatible chat completion APIs (vLLM, Ollama, TGI, llama.cpp).

This benchmark measures throughput in tokens per second by sending concurrent
chat completion requests to the /v1/chat/completions endpoint.

Usage:
    PYWORKER_BENCHMARK=benchmarks.openai_chat:benchmark

Environment variables:
    MODEL_NAME: Model name to use in requests (default: "model")
"""
import os
import time
import random
import logging
import asyncio
import traceback
from aiohttp import ClientSession

try:
    import nltk
    nltk.download("words", quiet=True)
    WORD_LIST = nltk.corpus.words.words()
except Exception:
    # Fallback word list if nltk not available
    WORD_LIST = ["test", "benchmark", "performance", "throughput", "workload"] * 50

log = logging.getLogger(__name__)

# System prompt used for benchmark and load test requests
SYSTEM_PROMPT = """You are a helpful AI assistant. You have access to the following knowledge base:

Zebras (US: /ˈziːbrəz/, UK: /ˈzɛbrəz, ˈziː-/)[2] (subgenus Hippotigris) are African equines
with distinctive black-and-white striped coats. There are three living species: Grévy's zebra
(Equus grevyi), the plains zebra (E. quagga), and the mountain zebra (E. zebra).

Please answer the following question based on the above context."""


def get_test_request() -> tuple[str, dict, float]:
    """
    Get a single test request for load testing.

    Returns:
        tuple: (endpoint_path, payload, workload)
            - endpoint_path: API endpoint (e.g., "/v1/chat/completions")
            - payload: Request payload dict
            - workload: Workload cost (tokens)
    """
    model_name = os.environ.get("MODEL_NAME", "model")

    # Generate test prompt
    user_prompt = " ".join(random.choices(WORD_LIST, k=250))
    max_tokens = 500

    endpoint = "/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    workload = max_tokens

    return endpoint, payload, workload


async def benchmark(backend_url: str, session: ClientSession, runs: int = 8) -> float:
    """
    Benchmark an OpenAI-compatible API.

    Args:
        backend_url: Base URL of the backend server (used for logging only)
        session: aiohttp ClientSession for making requests (already configured with base URL)
        runs: Number of benchmark runs (default: 8)

    Returns:
        max_throughput: Maximum workload processed per second
    """
    model_name = os.environ.get("MODEL_NAME", "model")
    endpoint = "/v1/chat/completions"

    log.info(f"Benchmarking OpenAI API at {backend_url}{endpoint}")

    # Initial warmup request
    log.info("Warming up...")
    warmup_prompt = " ".join(random.choices(WORD_LIST, k=50))
    warmup_payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": warmup_prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }

    try:
        async with session.post(endpoint, json=warmup_payload) as response:
            if response.status != 200:
                error_body = await response.text()
                log.error(
                    f"Warmup failed with status {response.status}\n"
                    f"Response: {error_body[:500]}"
                )
                return 1.0
            await response.read()  # Ensure response is fully consumed
    except Exception as e:
        log.error(
            f"Warmup failed with exception: {type(e).__name__}: {str(e)}\n"
            f"Exception details: {repr(e)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        return 1.0

    # Run benchmark
    max_throughput = 0
    sum_throughput = 0
    concurrent_requests = 10  # OpenAI APIs usually support parallel

    for run in range(1, runs + 1):
        start = time.time()
        workloads = []

        # Create benchmark payloads
        async def run_single_request():
            user_prompt = " ".join(random.choices(WORD_LIST, k=250))
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7,
            }
            workload = payload["max_tokens"]  # Workload is max_tokens

            try:
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Use actual completion tokens from response, fallback to max_tokens
                        actual_tokens = data.get("usage", {}).get("completion_tokens", workload)
                        return actual_tokens
                    else:
                        error_body = await response.text()
                        log.warning(
                            f"Request failed with status {response.status}\n"
                            f"Response: {error_body[:200]}"
                        )
                        return 0
            except Exception as e:
                log.warning(f"Request failed: {type(e).__name__}: {str(e)}")
                return 0

        # Run concurrent requests
        results = await asyncio.gather(*[run_single_request() for _ in range(concurrent_requests)])

        total_workload = sum(results)
        time_elapsed = time.time() - start
        successful = sum(1 for w in results if w > 0)

        if successful == 0:
            log.error(f"Benchmark run {run} failed: no successful responses")
            continue

        throughput = total_workload / time_elapsed
        sum_throughput += throughput
        max_throughput = max(max_throughput, throughput)

        log.info(
            f"Run {run}/{runs}: {successful}/{concurrent_requests} successful, "
            f"{total_workload} tokens in {time_elapsed:.2f}s = {throughput:.2f} tokens/s"
        )

    average_throughput = sum_throughput / runs if runs > 0 else 1.0
    log.info(
        f"Benchmark complete: avg={average_throughput:.2f} tokens/s, max={max_throughput:.2f} tokens/s"
    )

    return max_throughput if max_throughput > 0 else 1.0
