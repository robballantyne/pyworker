import logging
import json
import os
import sys
import argparse
from typing import Any, Dict, List

from vastai import Serverless
import asyncio

# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

# ---------------------- Defaults ----------------------
ENDPOINT_NAME = "my-vllm-score-endpoint"
DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"


# ---------------------- Score API Call ----------------------
async def call_score(
    client: Serverless,
    *,
    model: str,
    text_1: List[str],
    text_2: List[str],
    endpoint_name: str,
) -> Dict[str, Any]:
    """
    Call the /score endpoint for reranking/scoring text pairs.

    Args:
        client: Serverless client instance
        model: Model name to use
        text_1: List of query/instruction texts
        text_2: List of document texts to score against text_1
        endpoint_name: Name of the endpoint to call

    Returns:
        Score response containing similarity scores
    """
    endpoint = await client.get_endpoint(name=endpoint_name)

    payload = {
        "model": model,
        "text_1": text_1,
        "text_2": text_2,
    }
    log.debug("POST /score %s", json.dumps(payload)[:500])
    resp = await endpoint.request("/score", payload, cost=len(text_1))
    return resp["response"]


# ---------------------- Demo Runner ----------------------
class ScoreDemo:
    """Demo and testing functionality for the score API"""

    def __init__(self, client: Serverless, model: str, endpoint_name: str):
        self.client = client
        self.model = model
        self.endpoint_name = endpoint_name

    async def demo_simple_score(self) -> None:
        """Simple scoring example with a query and document pair."""
        print("=" * 60)
        print("SIMPLE SCORE DEMO")
        print("=" * 60)

        text_1 = ["What is the capital of France?"]
        text_2 = ["Paris is the capital and largest city of France."]

        print(f"\nQuery: {text_1[0]}")
        print(f"Document: {text_2[0]}")

        response = await call_score(
            client=self.client,
            model=self.model,
            text_1=text_1,
            text_2=text_2,
            endpoint_name=self.endpoint_name,
        )

        print("\nResponse:")
        print(json.dumps(response, indent=2))

        if "data" in response and response["data"]:
            score = response["data"][0].get("score", "N/A")
            print(f"\nScore: {score}")

    async def demo_batch_score(self) -> None:
        """Batch scoring example with multiple query-document pairs."""
        print("=" * 60)
        print("BATCH SCORE DEMO")
        print("=" * 60)

        # Multiple query-document pairs for batch scoring
        text_1 = [
            "What is machine learning?",
            "How does photosynthesis work?",
            "What is the speed of light?",
        ]
        text_2 = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
            "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        ]

        print("\nQuery-Document Pairs:")
        for i, (q, d) in enumerate(zip(text_1, text_2)):
            print(f"  [{i}] Query: {q[:50]}...")
            print(f"      Document: {d[:50]}...")

        response = await call_score(
            client=self.client,
            model=self.model,
            text_1=text_1,
            text_2=text_2,
            endpoint_name=self.endpoint_name,
        )

        print("\nResponse:")
        print(json.dumps(response, indent=2))

        if "data" in response:
            print("\nScores:")
            for item in response["data"]:
                idx = item.get("index", "?")
                score = item.get("score", "N/A")
                print(f"  [{idx}] Score: {score}")

    async def demo_reranker_format(self) -> None:
        """Demo using the reranker instruction format (similar to provided example)."""
        print("=" * 60)
        print("RERANKER FORMAT DEMO")
        print("=" * 60)

        # Using the instruction format typical for reranker models
        instruct = (
            "Given a query and a document, determine if the document is relevant "
            "to answering the query. Reply 'yes' if relevant, 'no' if not."
        )
        query = "What programming languages are good for data science?"
        document = "Python is widely used in data science due to its rich ecosystem of libraries like pandas, numpy, and scikit-learn."

        # Format as instruction prompt
        text_1 = [
            f"<|im_start|>system\n{instruct}<|im_end|>\n"
            f"<|im_start|>user\n<Query>: {query}<|im_end|>"
        ]
        text_2 = [
            f"<Document>: {document}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ]

        print(f"\nInstruction: {instruct[:80]}...")
        print(f"Query: {query}")
        print(f"Document: {document[:80]}...")

        response = await call_score(
            client=self.client,
            model=self.model,
            text_1=text_1,
            text_2=text_2,
            endpoint_name=self.endpoint_name,
        )

        print("\nResponse:")
        print(json.dumps(response, indent=2))

        if "data" in response and response["data"]:
            score = response["data"][0].get("score", "N/A")
            print(f"\nRelevance Score: {score}")


# ---------------------- CLI ----------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vast vLLM Score Demo (Serverless SDK)")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use for requests (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--endpoint",
        default=ENDPOINT_NAME,
        help=f"Vast endpoint name (default: {ENDPOINT_NAME})",
    )

    modes = p.add_mutually_exclusive_group(required=False)
    modes.add_argument(
        "--simple", action="store_true", help="Test simple scoring with one pair"
    )
    modes.add_argument(
        "--batch", action="store_true", help="Test batch scoring with multiple pairs"
    )
    modes.add_argument(
        "--reranker",
        action="store_true",
        help="Test reranker format with instruction prompt",
    )
    return p


async def main_async():
    args = build_arg_parser().parse_args()

    selected = sum([args.simple, args.batch, args.reranker])
    if selected == 0:
        print("Please specify exactly one test mode:")
        print("  --simple    : Test simple scoring with one pair")
        print("  --batch     : Test batch scoring with multiple pairs")
        print("  --reranker  : Test reranker format with instruction prompt")
        print(
            f"\nExample: python -m workers.vllm_score.client --model {DEFAULT_MODEL} --simple --endpoint my-endpoint"
        )
        sys.exit(1)
    elif selected > 1:
        print("Please specify exactly one test mode")
        sys.exit(1)

    print("=" * 60)
    print(f"Using model: {args.model}")
    print(f"Using endpoint: {args.endpoint}")

    try:
        async with Serverless() as client:
            demo = ScoreDemo(client, args.model, args.endpoint)

            if args.simple:
                await demo.demo_simple_score()
            elif args.batch:
                await demo.demo_batch_score()
            elif args.reranker:
                await demo.demo_reranker_format()

    except Exception as e:
        log.error("Error during test: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
