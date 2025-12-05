# PyWorker Client

Local proxy for Vast.ai serverless endpoints.

## Quick Start

### Interactive Mode (Easiest)
```bash
python client.py
```
Prompts for API key, shows endpoints, auto-fetches keys.

### Command Line
```bash
# With account key (auto-fetches endpoint key)
python client.py --endpoint my-endpoint --account-key YOUR_KEY

# With endpoint key directly
python client.py --endpoint my-endpoint --api-key ENDPOINT_KEY

# List endpoints
python client.py --list --account-key YOUR_KEY
```

### Environment Variables
```bash
export VAST_ACCOUNT_KEY="your-key"
export VAST_ENDPOINT="my-endpoint"
python client.py
```

## Using the Proxy

Point any SDK at `http://localhost:8010`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8010/v1", api_key="unused")
response = client.chat.completions.create(
    model="llama-2-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True  # Streaming works
)
```

```python
import requests

response = requests.post(
    "http://localhost:8010/v1/completions",
    json={"prompt": "Hello", "max_tokens": 100}
)
```

### Workload Cost

Specify via header (defaults to 1.0):
```python
requests.post(url, headers={"X-Serverless-Cost": "500"}, json=...)
```

## CLI Options

```
--endpoint NAME       Endpoint name (or VAST_ENDPOINT env)
--api-key KEY         Endpoint API key (or VAST_API_KEY env)
--account-key KEY     Account key for auto-fetch (or VAST_ACCOUNT_KEY env)
--list                List available endpoints
--port PORT           Local port (default: 8010)
--debug               Enable debug logging
```

## API Key Priority

1. CLI flags
2. Environment variables
3. `~/.vast_api_key` file (vastai CLI compatible)

## Python Module

```python
from client import VastClient
import asyncio

async def main():
    client = VastClient(endpoint_name="my-endpoint", api_key="KEY")
    await client.start()
    print(f"Proxy at: {client.url}")
    # Use client.url with any SDK
    await client.run_forever()

asyncio.run(main())
```

## Troubleshooting

**401 Unauthorized** - Use `--account-key`, not endpoint key directly

**No endpoints found** - Check console.vast.ai for endpoints

**Connection refused** - Ensure proxy is running
