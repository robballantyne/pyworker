# vLLM Score PyWorker

This PyWorker provides support for vLLM's `/score` endpoint, typically used with reranker models like `Qwen/Qwen3-Reranker-0.6B`.

## Instance Setup

1. Pick a template

This worker is compatible with vLLM backends that support the `/score` endpoint for reranking/scoring tasks.

- [vLLM + Qwen3-Reranker-0.6B](https://cloud.vast.ai?ref_id=145102&template_id=89aab9c71caf4a133ae46c79523b81ba)
- [vLLM + Qwen3-Reranker-4B](https://cloud.vast.ai?ref_id=145102&template_id=1b0bae2b77e110dffeda458f1ec8368f)

2. Follow the [getting started guide](https://docs.vast.ai/documentation/serverless/quickstart) for help with configuring your serverless setup.

## Score Endpoint

The `/score` endpoint accepts text pairs and returns similarity/relevance scores.

### Request Format

```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "text_1": ["Query or instruction text"],
  "text_2": ["Document text to score"]
}
```

### Response Format

```json
{
  "id": "score-...",
  "object": "list",
  "created": 1234567890,
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "data": [
    {
      "index": 0,
      "object": "score",
      "score": 0.85
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "total_tokens": 100,
    "completion_tokens": 0
  }
}
```

## Client Setup

1. Clone the PyWorker repository and install dependencies:

```bash
git clone https://github.com/vast-ai/pyworker
cd pyworker
pip install uv
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

2. Set your API key:

```bash
export VAST_API_KEY=<your_api_key>
```

## Using the Test Client

### Simple Score

Basic query-document scoring:

```bash
python -m workers.vllm-score.client --simple --endpoint <ENDPOINT_NAME>
```

### Instruction Format

Instruction-formatted scoring (typical for reranker models):

```bash
python -m workers.vllm-score.client --instruct --endpoint <ENDPOINT_NAME>
```

### Batch Mode

Send multiple pairs (first is real, rest are garbage for load testing):

```bash
python -m workers.vllm-score.client --simple --batch 10 --endpoint <ENDPOINT_NAME>
python -m workers.vllm-score.client --instruct --batch 50 --endpoint <ENDPOINT_NAME>
```

### Options

| Flag | Description |
|------|-------------|
| `--simple` | Simple query-document scoring |
| `--instruct` | Instruction-formatted scoring |
| `--batch N` | Send N pairs (first real, rest garbage) |
| `--model` | Model name (default: `Qwen/Qwen3-Reranker-0.6B`) |
| `--endpoint` | Endpoint name (default: `my-vllm-score-endpoint`) |
