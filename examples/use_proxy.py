"""
Example: Using the Vast.ai proxy server

The proxy server runs locally and handles all Vast.ai routing automatically.
Just point your existing code at localhost:8010!
"""
import requests

# Start the proxy first:
#   python client.py --endpoint my-endpoint --api-key YOUR_KEY
#
# Then run this script

# Now your code just talks to localhost - no Vast.ai complexity!
response = requests.post(
    "http://localhost:8010/v1/chat/completions",
    json={
        "model": "llama-2-7b",
        "messages": [
            {"role": "user", "content": "Tell me a joke about programming"}
        ],
        "max_tokens": 100,
    }
)

print(response.json())

# That's it! The proxy handles:
# - Calling /route/ to get worker assignment
# - Wrapping in auth_data + payload format
# - Forwarding to worker
# - Streaming back the response

# Works with ANY API:
# - OpenAI: http://localhost:8010/v1/completions
# - TGI: http://localhost:8010/generate
# - ComfyUI: http://localhost:8010/generate/sync
# - Your API: http://localhost:8010/your/endpoint
