# <INFERENCE_SERVER> + <MODEL_NAME> (serverless)

Run <INFERENCE_SERVER> with our serverless autoscaling infrastructure.

See the [serverless documentation](https://docs.vast.ai/serverless) and the [Getting Started](https://docs.vast.ai/serverless/getting-started) guide for in-depth details about how to use these templates.

## Configuration

Two environment variables are provided to help you configure the <INFERENCE_SERVER> server:

| Variable | Default Value | Used For |
| --- | --- | --- |
| `MODEL_NAME` | `<MODEL_NAME>` | The model to load.  Also accepts [hf.co/repo/model](#) links |
| `<ARGS_VAR>` | `<ARGS_VAL>` | Arguments to pass to the `<ARGS_RECEIVER>` command |

This template has been configured to work with <MIN_VRAM> VRAM. Setting alternative models and server arguments will change the VRAM requirements. Check model cards and <INFERENCE_SERVER_DOCS> for guidance.

## Usage

We have provided a demonstration client to help you implement this template into your own infrastructure

### Client Setup

Clone the PyWorker repository to your local machine and install the necessary requirements for running the test client.

```bash
git clone https://github.com/vast-ai/pyworker
cd pyworker
pip install uv
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Completions

Call to `/v1/completions` with json response

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --completion --model <MODEL_NAME>
```

### Chat Completion (json)

Call to `/v1/chat/completions` with json response

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --chat --model <MODEL_NAME>
```

### Chat Completion (streaming)

Call to `/v1/chat/completions` with streaming response

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --chat-stream --model <MODEL_NAME>
```

### Tool Use (json)

Call to `/v1/chat/completions` with tool and json response.

This test defines a simple tool which will list the contents of the local pyworker directory.  The output is then analysed by the model.

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --tools --model <MODEL_NAME>
```

### Interactive Chat (streaming)

Interactive session with calls to `/v1/chat/completions`.

Type `clear` to clear the chat history or `quit` to exit.

```bash
python -m workers.openai.client -k <API_KEY> -e <ENDPOINT_NAME> --interactive --model <MODEL_NAME>
```
