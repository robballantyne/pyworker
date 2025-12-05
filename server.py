"""
PyWorker - Universal serverless proxy for Vast.ai

This server proxies requests to any backend API without custom transformation.
It handles authentication, metrics tracking, and benchmarking.

Environment variables:
- PYWORKER_BACKEND_URL: URL of the backend server (e.g., http://localhost:8000)
- PYWORKER_BENCHMARK: Python module path with benchmark function (e.g., benchmarks.openai_chat:benchmark)
- PYWORKER_HEALTHCHECK_ENDPOINT: Optional healthcheck endpoint (e.g., /health)
- PYWORKER_ALLOW_PARALLEL: Whether to allow parallel requests (default: true)
- PYWORKER_MAX_WAIT_TIME: Maximum queue wait time before rejecting (default: 10.0)

Usage:
    python server.py
"""
import os
import logging
import importlib
from typing import Optional, Callable, Awaitable, cast
from aiohttp import web, ClientSession
from lib.backend import Backend
from lib.server import start_server

# Configure logging from environment variable
LOG_LEVEL = os.environ.get("PYWORKER_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


def load_benchmark_function() -> Optional[Callable[[str, ClientSession], Awaitable[float]]]:
    """
    Load benchmark function from BENCHMARK env var.

    Format: module.path:function_name
    Example: benchmarks.openai_chat:benchmark

    The function should have signature:
        async def benchmark(backend_url: str, session: ClientSession) -> float:
            # Run benchmark and return max_throughput in workload units per second
            return max_throughput
    """
    benchmark_spec = os.environ.get("PYWORKER_BENCHMARK")

    if not benchmark_spec:
        log.warning("No PYWORKER_BENCHMARK env var set, worker will error on startup")
        log.warning("Set PYWORKER_BENCHMARK to specify benchmark function (e.g., benchmarks.openai_chat:benchmark)")
        return None

    try:
        # Parse module:function
        if ":" not in benchmark_spec:
            raise ValueError(f"PYWORKER_BENCHMARK must be in format 'module:function', got: {benchmark_spec}")

        module_path, function_name = benchmark_spec.rsplit(":", 1)

        log.debug(f"Loading benchmark function: {module_path}:{function_name}")
        module = importlib.import_module(module_path)
        benchmark_func = getattr(module, function_name)

        if not callable(benchmark_func):
            raise ValueError(f"Benchmark {benchmark_spec} is not callable")

        log.debug(f"Successfully loaded benchmark function: {benchmark_spec}")
        # Cast to expected type since we can't verify signature at import time
        return cast(Callable[[str, ClientSession], Awaitable[float]], benchmark_func)

    except Exception as e:
        log.error(f"Failed to load benchmark function '{benchmark_spec}': {e}")
        log.error("Worker will error on startup without valid benchmark function")
        return None


# Load configuration from environment
backend_url = os.environ.get("PYWORKER_BACKEND_URL", "http://localhost:8000")
healthcheck_endpoint = os.environ.get("PYWORKER_HEALTHCHECK_ENDPOINT")
allow_parallel = os.environ.get("PYWORKER_ALLOW_PARALLEL", "true").lower() == "true"
max_wait_time = float(os.environ.get("PYWORKER_MAX_WAIT_TIME", "10.0"))

# Load benchmark function
benchmark_func = load_benchmark_function()

# Create backend
backend = Backend(
    backend_url=backend_url,
    benchmark_func=benchmark_func,
    healthcheck_endpoint=healthcheck_endpoint,
    allow_parallel_requests=allow_parallel,
    max_wait_time=max_wait_time,
)


async def handle_ping(_):
    """Simple ping endpoint for testing"""
    return web.Response(body="pong")


# Create catch-all route that forwards any path to the backend
# The actual backend endpoint path comes from the HTTP request path
# auth_data.endpoint contains the Vast endpoint name (for signature verification only)
routes = [
    # Catch-all route for all HTTP methods
    # Forwards request.path to the backend API
    web.post("/{path:.*}", backend.create_handler()),
    web.get("/{path:.*}", backend.create_handler()),
    web.put("/{path:.*}", backend.create_handler()),
    web.patch("/{path:.*}", backend.create_handler()),
    web.delete("/{path:.*}", backend.create_handler()),
]

# Add ping endpoint for testing (doesn't require auth)
routes.append(web.get("/ping", handle_ping))

if __name__ == "__main__":
    log.info(f"Starting PyWorker for backend: {backend_url}")
    log.info(f"Healthcheck endpoint: {healthcheck_endpoint or 'None'}")
    log.info(f"Allow parallel requests: {allow_parallel}")
    log.info(f"Max wait time: {max_wait_time}s")
    log.info(f"Benchmark: {os.environ.get('PYWORKER_BENCHMARK', 'None (worker will error on startup)')}")

    start_server(backend, routes)
