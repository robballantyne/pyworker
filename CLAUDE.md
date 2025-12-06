# PyWorker - Universal Serverless Proxy for Vast.ai

HTTP proxy enabling serverless compute on Vast.ai for any backend API.

## Architecture

```
Client → Autoscaler → PyWorker → Backend API
            ↓            ↓
     (routes/signs)  (validates/streams)
```

## File Structure

```
pyworker/
├── server.py           # Entry point, loads config and benchmark
├── client.py           # Client proxy for calling Vast.ai endpoints
├── lib/
│   ├── backend.py      # Core proxy logic, request handling
│   ├── metrics.py      # Metrics tracking, autoscaler reporting
│   ├── data_types.py   # AuthData, RequestMetrics, ModelMetrics
│   └── server.py       # aiohttp server setup
├── benchmarks/         # Benchmark functions
│   ├── openai_chat.py  # OpenAI chat completions (works for all LLMs)
│   └── comfyui.py      # ComfyUI (100% standard job convention)
└── start_server.sh     # Production startup script
```

## Key Components

### Backend (`lib/backend.py`)
- **Single shared session**: Safe for concurrent use (aiohttp guarantees)
- **Universal handler**: Forwards any HTTP method without transformation
- **Streaming**: Auto-detects via Content-Type or Transfer-Encoding
- **Startup**: Wait for backend → Run benchmark → Start healthchecks

### Metrics (`lib/metrics.py`)
- Reports to autoscaler every 1 second
- Tracks: workload served/received/cancelled/errored/rejected

### Data Types (`lib/data_types.py`)
- `AuthData`: cost, endpoint, reqnum, request_idx, signature, url
- `RequestMetrics`: Per-request tracking
- `ModelMetrics`: Aggregated workload metrics

## Environment Variables

### Required
| Variable | Description |
|----------|-------------|
| `PYWORKER_BACKEND_URL` | Backend API URL (e.g., `http://localhost:8000`) |
| `PYWORKER_WORKER_PORT` | Proxy listen port |

### Core Options
| Variable | Default | Description |
|----------|---------|-------------|
| `PYWORKER_BENCHMARK` | None | Benchmark module (e.g., `benchmarks.openai_chat:benchmark`) |
| `PYWORKER_HEALTHCHECK_ENDPOINT` | `/health` | Health check path |
| `PYWORKER_ALLOW_PARALLEL` | `true` | Allow concurrent requests |
| `PYWORKER_MAX_WAIT_TIME` | `10.0` | Max queue wait (seconds) |
| `PYWORKER_READY_TIMEOUT_INITIAL` | `1200` | Startup timeout (models downloading) |
| `PYWORKER_READY_TIMEOUT_RESUME` | `300` | Resume timeout (models on disk) |
| `PYWORKER_UNSECURED` | `false` | Skip signature verification (dev only) |
| `PYWORKER_USE_SSL` | `false`/`true` | SSL (false direct, true via start_server.sh) |
| `PYWORKER_BLOCKED_PATHS` | None | Comma-separated paths to block (supports `*` and `?` wildcards) |
| `PYWORKER_DEFAULT_COST` | None | Default workload when user cost <= 1 (e.g., `100` for job backends) |

### Client Options
| Variable | Default | Description |
|----------|---------|-------------|
| `PYWORKER_TIMEOUT` | None | Request timeout in seconds (no limit by default) |
| `PYWORKER_MAX_CONNECTIONS` | `100` | Max total HTTP connections |
| `PYWORKER_MAX_CONNECTIONS_PER_HOST` | `20` | Max connections per host |

### Set by Vast.ai
`MASTER_TOKEN`, `REPORT_ADDR`, `CONTAINER_ID`, `PUBLIC_IPADDR`, `VAST_TCP_PORT_*`

## Request Flow

### POST/PUT/PATCH (with body)
```json
{
  "auth_data": {"cost": 1.0, "endpoint": "/v1/completions", "reqnum": 1, ...},
  "payload": {"model": "...", "prompt": "..."}
}
```

### GET/DELETE/HEAD (no body)
Auth via query params with `serverless_` prefix:
```
?serverless_cost=1.0&serverless_endpoint=/v1/models&serverless_signature=...
```

### Unsecured Mode (local dev)
Send payload directly without auth_data wrapper.

## Workload and Cost

PyWorker uses `auth_data.cost` (user-provided) for all workload calculations:

```
wait_time = current_workload / max_throughput
if wait_time > max_wait_time: reject 429
```

**Units must match between benchmark and `auth_data.cost`:**

| Backend Type | Benchmark Returns | User Sends `cost` | Example |
|--------------|-------------------|-------------------|---------|
| LLMs | tokens/sec | expected tokens | 500 |
| Jobs (ComfyUI, Blender) | %/sec of standard job | % of standard job | 100 |

For jobs: `cost=100` = 100% of standard job, `cost=50` = lighter job, `cost=200` = heavier job.

## Benchmarks

```python
async def benchmark(backend_url: str, session: ClientSession) -> float:
    """
    Args:
        backend_url: For logging only
        session: ClientSession with base URL configured
    Returns:
        max_throughput in workload units/second (must match cost units)
    """
    # IMPORTANT: Use relative paths, NOT absolute URLs
    endpoint = "/v1/completions"
    async with session.post(endpoint, json=payload) as response:
        ...
    return throughput

# LLM benchmark: return tokens/sec (from response usage stats)
# Job benchmark: return (N * 100) / elapsed_seconds
```

## Important Patterns

1. **Shared session is safe** - aiohttp handles concurrent requests correctly
2. **Healthcheck before benchmark** - Always wait for backend ready first
3. **Consecutive failure tracking** - 3 failures before marking errored
4. **Relative paths in benchmarks** - Session already has base URL
5. **Signature over sorted JSON** - Must use `indent=4, sort_keys=True`

## Local Development

### Server (PyWorker proxy)
```bash
export PYWORKER_BACKEND_URL="http://localhost:8000"
export PYWORKER_BENCHMARK="benchmarks.openai_chat:benchmark"
export PYWORKER_WORKER_PORT="3000"
export PYWORKER_UNSECURED="true"
python server.py

# Test
curl -X POST http://localhost:3000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "prompt": "Hello", "max_tokens": 100}'
```

### Client Setup

The client requires the Vast.ai root certificate for SSL connections:

```bash
# Download and install Vast.ai certificate
curl -o /tmp/jvastai_root.cer https://console.vast.ai/static/jvastai_root.cer
cat /tmp/jvastai_root.cer >> $(python -c "import certifi; print(certifi.where())")
```

Run the client:
```bash
# Interactive mode (easiest)
python client.py

# With endpoint API key
python client.py --endpoint my-endpoint --api-key ENDPOINT_KEY

# With account API key (auto-fetches endpoint key)
python client.py --endpoint my-endpoint --account-key ACCOUNT_KEY

# With custom settings
python client.py --endpoint my-endpoint --timeout 600 --max-connections 200
```

Client features:
- `/health` endpoint for load balancer health checks
- Automatic retry on transient failures (5xx, network errors)
- Configurable connection pooling and timeouts
