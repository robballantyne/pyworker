"""
Example: Using VastClient as a Python module

The client starts a local proxy that handles all Vast.ai routing.
Use client.url with any SDK (OpenAI, Anthropic, requests, etc.)
"""
import sys
sys.path.insert(0, "..")  # Add parent directory to path

from client import VastClient
import asyncio


async def main():
    # Initialize and start the proxy
    client = VastClient(
        endpoint_name="my-endpoint",
        api_key="YOUR_ENDPOINT_API_KEY",  # Or set VAST_API_KEY env var
        port=8010,  # Optional, defaults to 8010
    )

    await client.start()
    print(f"Proxy started at: {client.url}")

    # Example 1: Use with OpenAI SDK
    print("\n=== Example 1: OpenAI SDK ===")
    try:
        from openai import OpenAI

        openai_client = OpenAI(
            base_url=f"{client.url}/v1",
            api_key="not-used"  # Proxy handles auth
        )

        response = openai_client.chat.completions.create(
            model="llama-2-7b",
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            max_tokens=50,
        )

        print("Response:", response.choices[0].message.content)
    except ImportError:
        print("OpenAI SDK not installed, skipping example")

    # Example 2: Use with requests library
    print("\n=== Example 2: Requests Library ===")
    import requests

    response = requests.post(
        f"{client.url}/v1/completions",
        json={
            "model": "llama-2-7b",
            "prompt": "Once upon a time",
            "max_tokens": 50,
        }
    )

    print("Completion:", response.json())

    # Example 3: Streaming with OpenAI SDK
    print("\n=== Example 3: Streaming with OpenAI SDK ===")
    try:
        from openai import OpenAI

        openai_client = OpenAI(
            base_url=f"{client.url}/v1",
            api_key="not-used"
        )

        print("Streaming response: ", end="", flush=True)
        response = openai_client.chat.completions.create(
            model="llama-2-7b",
            messages=[{"role": "user", "content": "Write a haiku about programming"}],
            max_tokens=50,
            stream=True,  # Native streaming support!
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    except ImportError:
        print("OpenAI SDK not installed, skipping example")

    # Example 4: Streaming with requests
    print("\n=== Example 4: Streaming with requests ===")
    response = requests.post(
        f"{client.url}/v1/completions",
        json={
            "model": "llama-2-7b",
            "prompt": "Count to 10:",
            "max_tokens": 50,
            "stream": True,
        },
        stream=True,
    )

    print("Streaming: ", end="", flush=True)
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            print(chunk.decode(), end="", flush=True)
    print()

    # Example 5: Specifying workload cost via header
    print("\n=== Example 5: Workload Cost ===")
    response = requests.post(
        f"{client.url}/v1/completions",
        headers={"X-Serverless-Cost": "500"},  # Indicate 500 workload units
        json={
            "model": "llama-2-7b",
            "prompt": "Write a long essay about AI",
            "max_tokens": 2000,
        }
    )
    print("High-cost completion status:", response.status_code)

    # Example 6: GET request
    print("\n=== Example 6: GET Request ===")
    response = requests.get(f"{client.url}/v1/models")
    print("Models:", response.json())

    # Example 7: Health check
    print("\n=== Example 7: Health Check ===")
    response = requests.get(f"{client.url}/health")
    print("Health:", response.text)

    print("\n" + "="*60)
    print("All examples completed!")
    print("The proxy handles all Vast.ai routing automatically:")
    print("  - Calls /route/ to get worker assignment")
    print("  - Wraps requests in auth_data + payload")
    print("  - Streams responses transparently")
    print("  - Works with any HTTP client or SDK")
    print("="*60)

    # Stop the proxy
    await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
