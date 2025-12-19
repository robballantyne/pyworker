# vLLM Score PyWorker

This PyWorker provides support for vLLM's `/score` endpoint, typically used with reranker models like `Qwen/Qwen3-Reranker-0.6B`.

## Instance Setup

1. Pick a template

This worker is compatible with vLLM backends that support the `/score` endpoint for reranking/scoring tasks.

- [vLLM](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=vLLM%20(Serverless)) with a reranker model

2. Follow the [getting started guide](https://docs.vast.ai/documentation/serverless/quickstart) for help with configuring your serverless setup.

## Score Endpoint

The `/score` endpoint accepts text pairs and returns similarity/relevance scores.

### Request Format

```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "text_1": [
    "Query or instruction text"
  ],
  "text_2": [
    "Document text to score"
  ]
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

## Client Setup (Demo)

1. Clone the PyWorker repository to your local machine and install the necessary requirements.

```bash
git clone https://github.com/vast-ai/pyworker
cd pyworker
pip install uv
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Using the Test Client

First, set your API key as an environment variable:

```bash
export VAST_API_KEY=<your_api_key>
```

### Simple Score Test

Test scoring with a single query-document pair:

```bash
python -m workers.vllm_score.client --simple --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Batch Score Test

Test batch scoring with multiple pairs:

```bash
python -m workers.vllm_score.client --batch --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Reranker Format Test

Test with instruction-formatted prompts typical for reranker models:

```bash
python -m workers.vllm_score.client --reranker --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```
