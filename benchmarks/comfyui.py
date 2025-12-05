"""
Benchmark for ComfyUI image generation API (comfyui-json worker).

This benchmark measures throughput by sending image generation requests
to the /generate/sync endpoint. Workload is fixed at 100.0 units per request.

Usage:
    PYWORKER_BENCHMARK=benchmarks.comfyui:benchmark

Environment variables:
    PYWORKER_COMFYUI_BENCHMARK_FILE: Path to benchmark.json workflow file (optional)
    BENCHMARK_TEST_WIDTH: Fallback image width (default: 512)
    BENCHMARK_TEST_HEIGHT: Fallback image height (default: 512)
    BENCHMARK_TEST_STEPS: Fallback generation steps (default: 20)
"""
import os
import sys
import json
import time
import random
import logging
import asyncio
import traceback
from pathlib import Path
from aiohttp import ClientSession, ClientTimeout

log = logging.getLogger(__name__)

# Fixed workload for ComfyUI - represents % of a job completed
WORKLOAD = 100.0

# Test prompts for fallback image generation
TEST_PROMPTS = [
    "a beautiful landscape with mountains and lakes",
    "a futuristic city at sunset",
    "a portrait of a wise old wizard",
    "a serene beach with palm trees",
    "a majestic castle on a hilltop",
    "a vibrant flower garden in spring",
    "a cozy cabin in the snowy woods",
    "an astronaut riding a horse on mars",
    "a steampunk airship in the clouds",
    "a magical forest with glowing mushrooms",
]


def load_benchmark_workflow() -> dict | None:
    """
    Try to load a benchmark workflow from file.

    Returns:
        Workflow dict if found and valid, None otherwise.
        Must be placed by docker image/provisioning script
    """
    benchmark_file = os.environ.get("PYWORKER_COMFYUI_BENCHMARK_FILE")

    if benchmark_file:
        path = Path(benchmark_file)
        if path.exists():
            try:
                with open(path, "r") as f:
                    workflow = json.load(f)
                log.info(f"Loaded benchmark workflow from {benchmark_file}")
                return workflow
            except (json.JSONDecodeError, IOError) as e:
                log.warning(f"Failed to load benchmark workflow from {benchmark_file}: {e}")

    return None


def get_test_request() -> tuple[str, dict, float]:
    """
    Get a single test request for load testing.

    Returns:
        tuple: (endpoint_path, payload, workload)
            - endpoint_path: API endpoint ("/generate/sync")
            - payload: Request payload dict
            - workload: Fixed workload cost (100.0)
    """
    endpoint = "/generate/sync"

    # Try to load benchmark workflow
    workflow = load_benchmark_workflow()

    if workflow:
        payload = {
            "input": {
                "request_id": f"test-{random.randint(1000, 99999)}",
                "workflow_json": workflow
            }
        }
    else:
        # Fallback to modifier-based request
        test_prompt = random.choice(TEST_PROMPTS)
        payload = {
            "input": {
                "request_id": f"test-{random.randint(1000, 99999)}",
                "modifier": "Text2Image",
                "modifications": {
                    "prompt": test_prompt,
                    "width": int(os.environ.get("BENCHMARK_TEST_WIDTH", 512)),
                    "height": int(os.environ.get("BENCHMARK_TEST_HEIGHT", 512)),
                    "steps": int(os.environ.get("BENCHMARK_TEST_STEPS", 20)),
                    "seed": random.randint(0, sys.maxsize),
                }
            }
        }

    return endpoint, payload, WORKLOAD


async def benchmark(backend_url: str, session: ClientSession, runs: int = 3) -> float:
    """
    Benchmark ComfyUI API.

    Args:
        backend_url: Base URL of the backend server (used for logging only)
        session: aiohttp ClientSession for making requests (already configured with base URL)
        runs: Number of benchmark runs (default: 3, fewer because image gen is slow)

    Returns:
        max_throughput: Maximum workload units processed per second
    """
    endpoint = "/generate/sync"

    log.info(f"Benchmarking ComfyUI API at {backend_url}{endpoint}")

    # Try to load benchmark workflow
    workflow = load_benchmark_workflow()

    if workflow:
        log.info("Using custom benchmark workflow from file")
    else:
        log.info("Using fallback Text2Image modifier for benchmarking")

    def create_payload() -> dict:
        """Create a benchmark request payload."""
        if workflow:
            return {
                "input": {
                    "request_id": f"benchmark-{random.randint(1000, 99999)}",
                    "workflow_json": workflow
                }
            }
        else:
            test_prompt = random.choice(TEST_PROMPTS)
            return {
                "input": {
                    "request_id": f"benchmark-{random.randint(1000, 99999)}",
                    "modifier": "Text2Image",
                    "modifications": {
                        "prompt": test_prompt,
                        "width": int(os.environ.get("BENCHMARK_TEST_WIDTH", 512)),
                        "height": int(os.environ.get("BENCHMARK_TEST_HEIGHT", 512)),
                        "steps": int(os.environ.get("BENCHMARK_TEST_STEPS", 20)),
                        "seed": random.randint(0, sys.maxsize),
                    }
                }
            }

    # Initial warmup request
    log.info("Warming up...")
    warmup_payload = create_payload()

    try:
        async with session.post(endpoint, json=warmup_payload, timeout=ClientTimeout(total=600)) as response:
            if response.status != 200:
                error_body = await response.text()
                log.error(
                    f"Warmup failed with status {response.status}\n"
                    f"Response: {error_body[:500]}"
                )
                return 1.0
            await response.read()
            log.info("Warmup successful")
    except Exception as e:
        log.error(
            f"Warmup failed with exception: {type(e).__name__}: {str(e)}\n"
            f"Exception details: {repr(e)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        return 1.0

    # Run benchmark
    # ComfyUI typically handles one request at a time
    max_throughput = 0.0
    sum_throughput = 0.0

    for run in range(1, runs + 1):
        start = time.time()

        payload = create_payload()

        try:
            async with session.post(endpoint, json=payload, timeout=ClientTimeout(total=600)) as response:
                if response.status == 200:
                    await response.read()
                    time_elapsed = time.time() - start
                    throughput = WORKLOAD / time_elapsed
                    sum_throughput += throughput
                    max_throughput = max(max_throughput, throughput)

                    requests_per_sec = 1.0 / time_elapsed
                    log.info(
                        f"Run {run}/{runs}: {WORKLOAD:.0f} workload in {time_elapsed:.2f}s = {throughput:.2f} workload/s ({requests_per_sec:.3f} req/s)"
                    )
                else:
                    error_body = await response.text()
                    log.warning(
                        f"Run {run}/{runs} failed with status {response.status}\n"
                        f"Response: {error_body[:200]}"
                    )
        except Exception as e:
            log.warning(f"Run {run}/{runs} failed: {type(e).__name__}: {str(e)}")

    average_throughput = sum_throughput / runs if runs > 0 else 1.0
    avg_req_per_sec = average_throughput / WORKLOAD
    max_req_per_sec = max_throughput / WORKLOAD
    log.info(
        f"Benchmark complete: avg={average_throughput:.2f} workload/s ({avg_req_per_sec:.3f} req/s), "
        f"max={max_throughput:.2f} workload/s ({max_req_per_sec:.3f} req/s)"
    )

    return max_throughput if max_throughput > 0 else 1.0
