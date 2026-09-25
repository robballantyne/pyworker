# OpenAI Compatible PyWorker

This is the base PyWorker for OpenAI compatible inference servers.  See the [Serverless documentation](https://docs.vast.ai/serverless) for guides and how-to's.

All worker logic lives in `core.py`. The per-engine backends `vllm`, `sglang` and `llama` are thin adapters over this core, differing only in their baked default log grammar (every value is env-overridable by the image). `BACKEND=openai` is a backwards-compatible **alias for `vllm`** — `openai/worker.py` runs the vLLM worker directly, so there is one definition of the vLLM defaults and no second copy to drift; templates declaring `openai` must run a pyworker new enough to contain this split. The demo test client in `client.py` is shared — run it as `python -m workers.openai.client` regardless of which engine backend you deployed.

## Routes

| Route | Request | Response |
|---|---|---|
| `/v1/completions`, `/v1/chat/completions` | JSON | JSON or SSE |
| `/v1/embeddings` | JSON | JSON |
| `/v1/audio/speech` | JSON (voice-clone `ref_audio`: http(s) URL or base64) | audio bytes |
| `/v1/audio/transcriptions`, `/v1/audio/translations` | JSON with the file base64'd in `file` | JSON or text |
| `/v1/images/generations` | JSON | JSON |
| `/v1/images/edits` | JSON with `image` (base64, or a list), or `url` | JSON |

The worker envelope is JSON, so uploads arrive base64-encoded (`file`, `image`, `mask`) and
are sent to the engine as multipart form data. `filename` / `mask_filename` set the file
type. Requires a `vastai` SDK with multipart support; on an older SDK the three upload
routes are not served.

**All routes are served by default.** The worker does not know what the loaded model
supports, so it registers every route and lets the engine answer. A route the model does
not serve returns the engine's 404 from behind the worker — with one caveat worth knowing:
the request is still counted as work against this worker's queue estimate before it is
forwarded, so a client hammering a route the model does not serve can make the worker shed
load it could otherwise accept. `OPENAI_ROUTES` narrows the set if that matters for a
deployment.

What each engine actually serves depends on the engine and the loaded model:

| Engine | Typically serves |
|---|---|
| vLLM, SGLang, llama.cpp | completions, chat; embeddings with an embedding model |
| vLLM with Whisper or Voxtral | transcriptions, translations (and no completions at all) |
| vLLM-Omni (`--omni`) | image generations, speech, alongside text |

## Benchmarking

`BENCHMARK_ROUTE` names the route to benchmark, and defaults to `/v1/completions`, so an
LLM deployment benchmarks exactly what it did before. A deployment serving something else
sets it: `/v1/audio/transcriptions` for Whisper, `/v1/embeddings` for an embedding model.

The route is a property of the deployment, not of the engine. An omni model may serve chat
and speech, and only the template knows which the endpoint exists for -- benchmarking the
wrong one prices every request against the other one's throughput.

Every route's benchmark request weighs the same, one reference-sized request, so the score
means the same thing whichever route is benchmarked. Exactly one route carries a
benchmark, which is what the SDK expects, so this needs no SDK support.

If the model does not serve the benchmarked route, the benchmark fails and the worker does
not become ready. The startup log names the route and the variable to change:

```
benchmarking: /v1/completions. If this model does not serve it, set BENCHMARK_ROUTE to
the route it does.
```

So a Whisper deployment that forgets `BENCHMARK_ROUTE=/v1/audio/transcriptions` shows up
as an endpoint that never becomes ready, not as one that serves wrong answers.

## Settings

| Variable | Default | Effect |
|---|---|---|
| `BENCHMARK_ROUTE` | `/v1/completions` | The route to benchmark. It must be served, and one of: completions, chat, embeddings, speech, image generations, image edits, transcriptions. An edit-only model must use `/v1/images/edits`; the edit benchmark sends a synthetic 1024x1024 input. |
| `OPENAI_ROUTES` | all | Comma-separated routes to serve. Must include `BENCHMARK_ROUTE`. |
| `BENCHMARK_SPEECH_VOICE` | none | `voice` to send when benchmarking speech. |
| `BENCHMARK_AUDIO_URL` | a public speech sample | Clip the transcription benchmark transcribes — point it at audio like your own traffic, since language, noise and speech density drive the decode work the score depends on. Any format this worker accepts. WAV is resized to the 30s reference and varied per request; other containers are sent as supplied and priced by their real duration. Fetched once at first use; on failure it falls back to synthetic noise, which leaves the decoder idle and overstates real-speech throughput by roughly 2x. |
| `BENCHMARK_AUDIO_TIMEOUT` | 20 | Seconds allowed for that fetch. |
| `BENCHMARK_EMBED_CHARS` | 600 | Characters the embeddings benchmark sends, and what one reference embedding request means. Sized for a 256-token encoder; raise it for a long-context model, lower it if an unusually small one refuses the payload. |
| `WORKER_MAX_UPLOAD_BYTES` | 26214400 | Largest single uploaded file (decoded). |
| `WORKER_MAX_REQUEST_UPLOAD_BYTES` | 67108864 | Largest total of all uploads and inline references in one request. |

## Instance Setup

1. Pick a template

This worker is compatible with any backend API that properly implements the `/v1/completions` and `/v1/chat/completions` endpoints.  We currently have three templates you can choose from but you can also create your own without having to modify the PyWorker.

- [vLLM](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=vLLM%20(Serverless)) (recommended)
- [Ollama](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=Ollama%20%2B%20Qwen3%3A32b%20(Serverless))


All of these templates can be configured via the template interface.  You may want to change the model or startup arguments, depending on the template you selected.

2. Follow the [getting started guide](https://docs.vast.ai/documentation/serverless/quickstart) for help with configuring your serverless setup.  For testing, we recommend that you use the default options presented by the web interface.

## Client Setup (Demo)

1. Clone the PyWorker repository to your local machine and install the necessary requirements for running the test client.

```bash
git clone https://github.com/vast-ai/pyworker
cd pyworker
pip install uv
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Using the Test Client

Several examples have been provided in the client to help you get started with your own implementation.

First, set your API key as an environment variable:

```bash
export VAST_API_KEY=<your_api_key>
```

The `--model` and `--endpoint` flags are optional. If not provided, they default to `Qwen/Qwen3-8B` and `my-vllm-endpoint` respectively.

### Chat Completion (streaming)

Call to `/v1/chat/completions` with streaming response

```bash
python -m workers.openai.client --chat-stream --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Interactive Chat (streaming)

Interactive session with calls to `/v1/chat/completions`.

Type `clear` to clear the chat history or `quit` to exit.

```bash
python -m workers.openai.client --interactive --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Chat Completion (json)

Call to `/v1/chat/completions` with json response

```bash
python -m workers.openai.client --chat --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Tool Use (json)

Call to `/v1/chat/completions` with tool and json response.

This test defines a simple tool which will list the contents of the local pyworker directory.  The output is then analysed by the model.

```bash
python -m workers.openai.client --tools --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

### Completions

Call to `/v1/completions` with json response

```bash
python -m workers.openai.client --completion --endpoint <ENDPOINT_NAME> --model <MODEL_NAME>
```

