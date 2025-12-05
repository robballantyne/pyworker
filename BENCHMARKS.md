# PyWorker Benchmarks

Benchmark functions measure backend throughput for autoscaling.

## Built-in Benchmarks

### OpenAI Chat Completions (`benchmarks.openai_chat`)
```bash
export PYWORKER_BENCHMARK="benchmarks.openai_chat:benchmark"
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
```
Works with vLLM, Ollama, TGI (OpenAI mode), llama.cpp.

### Text Generation Inference (`benchmarks.tgi`)
```bash
export PYWORKER_BENCHMARK="benchmarks.tgi:benchmark"
```

### ComfyUI (`benchmarks.comfyui`)
```bash
export PYWORKER_BENCHMARK="benchmarks.comfyui:benchmark"

# Optional: custom workflow file
export PYWORKER_COMFYUI_BENCHMARK_FILE="/path/to/benchmark.json"

# Fallback settings (used when no benchmark.json provided)
export BENCHMARK_TEST_WIDTH=512
export BENCHMARK_TEST_HEIGHT=512
export BENCHMARK_TEST_STEPS=20
```
Works with comfyui-json worker. Uses `/generate/sync` endpoint with fixed 100.0 workload.
If no benchmark file is set, falls back to `Text2Image` modifier.

## Writing Custom Benchmarks

```python
"""benchmarks/myapi.py"""
import time
import logging
import asyncio
from aiohttp import ClientSession

log = logging.getLogger(__name__)


def get_test_request() -> tuple[str, dict, float]:
    """Return (endpoint, payload, workload) for load testing."""
    return "/my-endpoint", {"data": "test"}, 100.0


async def benchmark(backend_url: str, session: ClientSession) -> float:
    """
    Args:
        backend_url: For logging only
        session: ClientSession with base URL configured
    Returns:
        Maximum workload/second
    """
    # IMPORTANT: Use relative paths, NOT absolute URLs
    endpoint = "/my-endpoint"

    # Warmup
    async with session.post(endpoint, json={"test": True}) as r:
        if r.status != 200:
            return 1.0
        await r.read()

    # Benchmark runs
    max_throughput = 0
    for run in range(8):
        start = time.time()

        async def request():
            try:
                async with session.post(endpoint, json={"data": "..."}) as r:
                    if r.status == 200:
                        await r.read()
                        return 100  # workload units
                    return 0
            except:
                return 0

        results = await asyncio.gather(*[request() for _ in range(10)])
        throughput = sum(results) / (time.time() - start)
        max_throughput = max(max_throughput, throughput)
        log.info(f"Run {run+1}: {throughput:.2f}/s")

    return max_throughput or 1.0
```

Use with:
```bash
export PYWORKER_BENCHMARK="benchmarks.myapi:benchmark"
```

## Key Points

1. **Relative paths only** - Session already has base URL
2. **Return max throughput** - Not average
3. **Warmup first** - Trigger model loading
4. **Handle errors** - Return 1.0 as fallback
5. **Workload units** - Must match autoscaler's `auth_data.cost`

## Testing

```python
import asyncio
from aiohttp import ClientSession
from benchmarks.openai_chat import benchmark

async def test():
    async with ClientSession("http://localhost:8000") as s:
        result = await benchmark("http://localhost:8000", s)
        print(f"Throughput: {result}")

asyncio.run(test())
```

## Cached Results

Results cached in `.has_benchmark`. Delete to re-benchmark:
```bash
rm .has_benchmark
```
