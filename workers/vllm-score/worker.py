import json
import os
from itertools import cycle
from pathlib import Path

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# vLLM model configuration
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18000
MODEL_LOG_FILE             = '/var/log/portal/vllm.log'
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# vLLM-specific log messages
MODEL_LOAD_LOG_MSG = [
    "Application startup complete.",
]

MODEL_ERROR_LOG_MSGS = [
    "INFO exited: vllm",
    "RuntimeError: Engine",
    "Traceback (most recent call last):"
]

MODEL_INFO_LOG_MSGS = [
    '"message":"Download'
]

# Load benchmark pairs from JSON file
_benchmark_file = Path(__file__).parent / "benchmark_pairs.json"
with open(_benchmark_file) as f:
    _benchmark_data = json.load(f)

# Convert to list of tuples and create cycling iterator
BENCHMARK_TEXT_PAIRS = [(pair["query"], pair["document"]) for pair in _benchmark_data]
_benchmark_iterator = cycle(BENCHMARK_TEXT_PAIRS)


def request_parser(request):
    """Parse incoming request to extract data."""
    data = request
    if request.get("input") is not None:
        data = request.get("input")
    return data


def estimate_token_count(data: dict) -> int:
    """
    Estimate token count from text_1 and text_2 arrays using word count.
    Word count is a reasonable approximation for English text.
    Used for both benchmark and per-request workload calculation.
    """
    total_words = 0
    for text_list in [data.get("text_1", []), data.get("text_2", [])]:
        for text in text_list:
            if isinstance(text, str):
                total_words += len(text.split())
    return max(total_words, 1)  # Ensure at least 1 to avoid zero workload


def score_benchmark_generator() -> dict:
    """Generate benchmark payload for /score endpoint with sequential text pairs."""
    model = os.environ.get("MODEL_NAME")
    if not model:
        raise ValueError("MODEL_NAME environment variable not set")

    # Get the next text pair from the cycling iterator
    query, document = next(_benchmark_iterator)

    benchmark_data = {
        "model": model,
        "text_1": [query],
        "text_2": [document],
    }

    return benchmark_data


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/score",
            workload_calculator=estimate_token_count,
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
            benchmark_config=BenchmarkConfig(
                generator=score_benchmark_generator,
                concurrency=10,
                runs=3
            )
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

Worker(worker_config).run()
