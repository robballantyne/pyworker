# PyWorker - Universal Serverless Proxy for Vast.ai

Lightweight HTTP proxy enabling serverless compute on Vast.ai for any backend API.

## Features

- **Universal** - Works with any HTTP API (OpenAI, vLLM, TGI, Ollama, ComfyUI)
- **Zero boilerplate** - No custom handlers or transformations
- **Streaming** - Automatic detection and pass-through
- **All HTTP methods** - GET, POST, PUT, PATCH, DELETE

## Quick Start

### On Vast.ai

```bash
export PYWORKER_BACKEND_URL="http://localhost:8000"
export PYWORKER_BENCHMARK="benchmarks.openai_chat:benchmark"
./start_server.sh
```

### Local Development

```bash
export PYWORKER_BACKEND_URL="http://localhost:8000"
export PYWORKER_BENCHMARK="benchmarks.openai_chat:benchmark"
export PYWORKER_WORKER_PORT="3000"
export PYWORKER_UNSECURED="true"
python server.py
```

```bash
curl -X POST http://localhost:3000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "prompt": "Hello", "max_tokens": 100}'
```

## Configuration

### Required

| Variable | Description |
|----------|-------------|
| `PYWORKER_BACKEND_URL` | Backend API URL |
| `PYWORKER_WORKER_PORT` | Proxy listen port (default: 3000 on Vast.ai) |

### Core Options

| Variable | Default | Description |
|----------|---------|-------------|
| `PYWORKER_BENCHMARK` | None | Benchmark module path (e.g., `benchmarks.openai_chat:benchmark`) |
| `PYWORKER_HEALTHCHECK_ENDPOINT` | `/health` | Health check path |
| `PYWORKER_ALLOW_PARALLEL` | `true` | Allow concurrent requests |
| `PYWORKER_MAX_WAIT_TIME` | `10.0` | Max queue wait (seconds) |
| `PYWORKER_READY_TIMEOUT_INITIAL` | `1200` | Startup timeout for model downloads |
| `PYWORKER_READY_TIMEOUT_RESUME` | `300` | Resume timeout (models on disk) |
| `PYWORKER_UNSECURED` | `false` | Skip signature verification (dev only) |
| `PYWORKER_USE_SSL` | varies | SSL enabled (true on Vast.ai, false locally) |
| `PYWORKER_LOG_LEVEL` | `INFO` | Logging level |
| `PYWORKER_BLOCKED_PATHS` | None | Comma-separated paths to block (supports `*` and `?` wildcards) |

### Advanced Options

See all tunables in `lib/backend.py` and `lib/metrics.py`:
- Connection pooling: `PYWORKER_CONNECTION_LIMIT*`
- Healthcheck tuning: `PYWORKER_HEALTHCHECK_*`
- Metrics reporting: `PYWORKER_METRICS_*`

## Benchmarks

PyWorker requires a benchmark to measure throughput:

```python
async def benchmark(backend_url: str, session: ClientSession) -> float:
    # Use relative paths - session has base URL configured
    endpoint = "/v1/completions"
    async with session.post(endpoint, json=payload) as response:
        ...
    return max_throughput  # workload units/second
```

Built-in benchmarks:
- `benchmarks.openai_chat:benchmark` - OpenAI-compatible APIs
- `benchmarks.tgi:benchmark` - Text Generation Inference
- `benchmarks.comfyui:benchmark` - ComfyUI

See [BENCHMARKS.md](BENCHMARKS.md) for writing custom benchmarks.

## Client Proxy

Call Vast.ai endpoints from your code:

```bash
# Interactive mode
python client.py

# Or specify endpoint
python client.py --endpoint my-endpoint --account-key YOUR_KEY
```

Then use `http://localhost:8010` as your API base URL.

See [CLIENT.md](CLIENT.md) for details.

## Serverless Compatibility

Your backend **must hold the HTTP connection open until processing is complete**. PyWorker marks a request as "done" when the backend returns a response - if your backend returns early with a job ID, the worker may scale down before the job finishes.

**Compatible backends** (work out of the box):
- vLLM, TGI, Ollama - hold connection until generation complete
- Any synchronous API

**Async backends** require a sync wrapper:
- ComfyUI - use a wrapper that submits workflow and polls until complete
- Queue-based systems - add a synchronous endpoint that blocks until done

### Blocking Async Endpoints

If your backend exposes both sync and async endpoints, block the async ones:

```bash
# Block specific paths (comma-separated, supports wildcards)
PYWORKER_BLOCKED_PATHS="/generate,/queue/submit"

# Wildcard examples
PYWORKER_BLOCKED_PATHS="/api/*/async"     # Matches /api/foo/async, /api/bar/async
PYWORKER_BLOCKED_PATHS="/jobs/*"          # Matches /jobs/123, /jobs/status
PYWORKER_BLOCKED_PATHS="/v?/queue"        # Matches /v1/queue, /v2/queue
```

Requests to blocked paths return `403 Forbidden` with an error message.

## How It Works

```
Client → Autoscaler → PyWorker → Backend API
            ↓            ↓
     (routes/signs)  (validates/streams)
```

1. Client sends request to Vast.ai
2. Autoscaler signs and routes to worker
3. PyWorker validates, forwards to backend
4. Response streams back to client

## File Structure

```
vespa/
├── server.py           # Entry point
├── client.py           # Client proxy
├── lib/
│   ├── backend.py      # Core proxy logic
│   ├── metrics.py      # Metrics tracking
│   ├── data_types.py   # Data structures
│   └── server.py       # Server setup
├── benchmarks/         # Benchmark functions
└── start_server.sh     # Production startup
```

## Troubleshooting

**Backend Connection Error**
- Verify `PYWORKER_BACKEND_URL` is correct
- Ensure backend is running

**Benchmark Fails**
- Check backend health endpoint
- Test benchmark function directly

**Worker Not Ready**
- Increase `PYWORKER_READY_TIMEOUT_INITIAL` for slow-loading models
- Verify healthcheck returns HTTP 200

## Resources

- [Benchmark Guide](BENCHMARKS.md)
- [Client Guide](CLIENT.md)
- [Vast.ai Discord](https://discord.gg/Pa9M29FFye)

## License

MIT License
