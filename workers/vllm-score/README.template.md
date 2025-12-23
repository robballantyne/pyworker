# vLLM Score / Reranker Inference Engine
> **[Create an Instance](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=vLLM-Score)**

## What is this template?

This template gives you a **hosted vLLM Score API server** running in a Docker container. It's optimized for reranker models that score the relevance between text pairs - perfect for search, RAG pipelines, and document retrieval systems.

**Think:** *"Your own private, high-performance reranking API for scoring document relevance."*

---

## What can I do with this?

- **Serve reranker models** like Qwen3-Reranker for scoring text pairs
- **Score document relevance** against queries for search and RAG applications
- **Load any compatible reranker model** from HuggingFace
- **Send API requests** to score text pairs via the `/score` endpoint
- **Integrate with your search pipeline** via REST API
- **Scale with multiple GPUs** automatically using tensor parallelism
- **Access your API** from anywhere or via SSH tunnel
- **Terminal access** with some root privileges (unprivileged Docker container)

---

## Who is this for?

This is **perfect** if you:
- Are building search or retrieval systems that need document reranking
- Want to improve RAG pipeline accuracy with relevance scoring
- Need a high-throughput scoring API for production applications
- Are building semantic search applications
- Want to evaluate document relevance at scale

---

## Quick Start Guide

### **Step 1: Configure Your Model**
Set the `MODEL_NAME` environment variable with your desired reranker model:
- **Recommended:** `Qwen/Qwen3-Reranker-0.6B` (lightweight, fast)
- **Higher accuracy:** `Qwen/Qwen3-Reranker-4B` (more compute required)
- **Any compatible reranker:** Use the full HuggingFace model path

**Optional configuration:**
- `VLLM_ARGS`: Additional arguments to pass to the vLLM serve command
- `USE_ALL_GPUS`: Set to `true` to automatically use all available GPUs with tensor parallelism

> **Template Customization:** Templates can't be changed directly, but you can easily make your own version! Just click **edit**, make your changes, and save it as your own template. You'll find it in your **"My Templates"** section later. [Full guide here](https://docs.vast.ai/templates)

### **Step 2: Launch Instance**
Click **"[Rent](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=vLLM-Score)"** when you've found an instance that works for you

### **Step 3: Wait for Setup**
vLLM and your chosen reranker model will install and start automatically *(this might take a few minutes depending on model size)*

### **Step 4: Access Your Instance**
**Easy access:** Just click the **"Open"** button - authentication is handled automatically!

**For external API calls:** If you want to make requests from outside (like curl commands), you'll need:
- **API Endpoint:** Your instance IP with the mapped external port
- **Auth Token:** Auto-generated when your instance starts - see the **"Finding Your Token"** section below

> **HTTPS Option:** Want secure connections? Set `ENABLE_HTTPS=true` in the **Environment Variables section** of your Vast.ai account settings page. You'll need to [install the Vast.ai certificate](https://docs.vast.ai/instances/jupyter) to avoid browser warnings.

### **Step 5: Make Your First Request**
```bash
curl -X POST -H 'Authorization: Bearer <YOUR_TOKEN>' \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "Qwen/Qwen3-Reranker-0.6B",
       "text_1": ["What is the capital of France?"],
       "text_2": ["Paris is the capital and largest city of France."]
     }' \
     http://<INSTANCE_IP>:<MAPPED_PORT>/score
```

---

## Key Features

### **Authentication & Access**
| Method | Use Case | Setup Required |
|--------|----------|----------------|
| **Web Interface** | Quick testing | Click "Open" button |
| **API Calls** | Development | Use Bearer token |
| **SSH Tunnel** | Local development | [SSH port forwarding](https://docs.vast.ai/instances/sshscp) to port 18000 |

### **Finding Your Token**
Your authentication token is available as `OPEN_BUTTON_TOKEN` in your instance environment. You can find it by:
- SSH: `echo $OPEN_BUTTON_TOKEN`
- Jupyter terminal: `echo $OPEN_BUTTON_TOKEN`

### **Score API Endpoint**

The `/score` endpoint accepts text pairs and returns relevance scores:

**Request Format:**
```json
{
  "model": "Qwen/Qwen3-Reranker-0.6B",
  "text_1": ["Query or instruction text"],
  "text_2": ["Document text to score"]
}
```

**Response Format:**
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

**Batch Scoring:**
```bash
curl -X POST -H 'Authorization: Bearer <TOKEN>' \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "Qwen/Qwen3-Reranker-0.6B",
       "text_1": [
         "What is machine learning?",
         "How does photosynthesis work?"
       ],
       "text_2": [
         "Machine learning is a subset of AI that enables systems to learn from data.",
         "Photosynthesis converts sunlight into chemical energy in plants."
       ]
     }' \
     http://<INSTANCE_IP>:<MAPPED_PORT>/score
```

### **Instruction Format (Advanced)**
For reranker models that support instruction formatting:
```bash
curl -X POST -H 'Authorization: Bearer <TOKEN>' \
     -H 'Content-Type: application/json' \
     -d '{
       "model": "Qwen/Qwen3-Reranker-0.6B",
       "text_1": ["<|im_start|>system\nDetermine if the document answers the query.<|im_end|>\n<|im_start|>user\n<Query>: What is Python?<|im_end|>"],
       "text_2": ["<Document>: Python is a programming language.<|im_end|>\n<|im_start|>assistant\n"]
     }' \
     http://<INSTANCE_IP>:<MAPPED_PORT>/score
```

### **Python Environment**
- Clean virtual environment (`/venv/main/`) ready for your custom setup
- Environment activates automatically when you connect
- Install any packages you need with `pip install` or `uv pip install`
- Perfect for building search applications that use your reranker API

### **Node, npm, nvm**
- **Node Version Manager (NVM)** manages Node.js environments
- Pre-installed with latest LTS Node.js version
- Great for building web applications or APIs that use your reranker
- Essential for modern JavaScript/TypeScript development

### **Instance Portal (Application Manager)**
- Web-based dashboard for managing your applications
- Cloudflare tunnels for easy sharing (no port forwarding needed!)
- Log monitoring for running services
- Start and stop services with a few clicks

### **Dynamic Provisioning**
Need specific software installed automatically? Set the `PROVISIONING_SCRIPT` environment variable to a plain-text script URL (GitHub, Gist, etc.), and we'll run your setup script on first boot!

### **Multiple Access Methods**
| Method | Best For | What You Get |
|--------|----------|--------------|
| **Jupyter** | Interactive development | Browser-based coding environment |
| **SSH** | Terminal work | Full command-line access with tmux |
| **Instance Portal** | Managing services | Application manager dashboard |

### **Service Management**
- **Supervisor** manages all background services
- Easy commands: `supervisorctl status`, `supervisorctl restart vllm`
- Add your own services with simple configuration files

### **Task Scheduling**
- **Cron** is enabled for automating routine tasks
- Schedule model downloads, API health checks, or maintenance tasks
- Just add entries to your crontab to get started

### **Instance Control**
- **Vast.ai CLI** comes pre-installed with instance-specific API key
- Stop your instance from within itself: `vastai stop instance $CONTAINER_ID`
- Perfect for automated shutdown based on API usage or conditions

---

## Use Cases

### **RAG Pipeline Reranking**
Improve retrieval accuracy by reranking candidate documents:
```python
import requests

def rerank_documents(query, documents, api_url, token):
    response = requests.post(
        f"{api_url}/score",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": "Qwen/Qwen3-Reranker-0.6B",
            "text_1": [query] * len(documents),
            "text_2": documents
        }
    )
    scores = [d["score"] for d in response.json()["data"]]
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return ranked
```

### **Search Result Optimization**
Score and filter search results by relevance threshold:
```python
results = rerank_documents(user_query, search_results, api_url, token)
relevant_docs = [(doc, score) for doc, score in results if score > 0.5]
```

---

## Customization Tips

### **Installing Software**
```bash
# You have root access - install anything!
apt update && apt install -y your-favorite-package

# Install Python packages for API integrations
uv pip install requests openai langchain

# Add system services
echo "your-service-config" > /etc/supervisor/conf.d/my-app.conf
supervisorctl reread
```

### **Environment Variables**
Customize your experience with these handy variables:
- `MODEL_NAME`: Set your reranker model (e.g., `Qwen/Qwen3-Reranker-0.6B`)
- `VLLM_ARGS`: Additional arguments for vLLM serve command (e.g., `--max-model-len 4096 --gpu-memory-utilization 0.9`)
- `USE_ALL_GPUS`: Set to `true` to automatically use all available GPUs with tensor parallelism
- `WORKSPACE`: Change your default working directory
- `PROVISIONING_SCRIPT`: Auto-run setup scripts from GitHub, Gist, or any plain-text URL
- `ENABLE_HTTPS`: Force HTTPS connections - set in your Vast.ai account settings

### **Template Customization**
Want to save your perfect setup? Templates can't be changed directly, but you can easily make your own version! Just click **edit**, make your changes, and save it as your own template. You'll find it in your **"My Templates"** section later.

---

## Need More Help?

- **Base Image Features:** [GitHub Repository](https://github.com/vast-ai/base-image/)
- **Instance Portal Guide:** [Vast.ai Instance Portal Documentation](https://docs.vast.ai/instance-portal)
- **SSH Setup Guide:** [Vast.ai SSH Documentation](https://docs.vast.ai/instances/sshscp)
- **Template Configuration:** [Vast.ai Template Guide](https://docs.vast.ai/templates)
- **vLLM Documentation:** [Official vLLM Documentation](https://docs.vllm.ai/)
- **Support:** Use the messaging icon in the Vast.ai console

updated 20251222
