import logging
import json
import random
import sys
import argparse
from typing import Any, Dict, List

from vastai import Serverless
import asyncio

from .utils import estimate_token_count

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

# Words for generating garbage pairs
GARBAGE_WORDS = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "lambda",
    "sigma", "omega", "quick", "brown", "fox", "jumps", "lazy", "dog", "hello",
    "world", "python", "code", "test", "data", "model", "score", "query",
    "document", "system", "process", "result", "value", "index", "array",
]


def generate_garbage_text(min_words: int = 5, max_words: int = 50) -> str:
    """Generate random garbage text of varying length."""
    num_words = random.randint(min_words, max_words)
    return " ".join(random.choices(GARBAGE_WORDS, k=num_words))


def generate_garbage_pairs(count: int) -> tuple[List[str], List[str]]:
    """Generate garbage query-document pairs of varying lengths."""
    queries = []
    documents = []
    for _ in range(count):
        queries.append(generate_garbage_text(3, 15))
        documents.append(generate_garbage_text(10, 100))
    return queries, documents


# ---------------------- Score API Call ----------------------
async def call_score(
    client: Serverless,
    *,
    model: str,
    text_1: List[str],
    text_2: List[str],
    endpoint_name: str,
) -> Dict[str, Any]:
    """Call the /score endpoint for reranking/scoring text pairs."""
    endpoint = await client.get_endpoint(name=endpoint_name)

    payload = {
        "model": model,
        "text_1": text_1,
        "text_2": text_2,
    }
    cost = estimate_token_count(payload)
    log.debug("POST /score (cost=%d, pairs=%d)", cost, len(text_1))
    resp = await endpoint.request("/score", payload, cost=cost)
    return resp["response"]


# ---------------------- Demo Functions ----------------------
async def demo_simple(
    client: Serverless, model: str, endpoint_name: str, batch: int
) -> None:
    """Simple scoring example with a query and document pair."""
    print("=" * 60)
    print("SIMPLE SCORE DEMO")
    print("=" * 60)

    text_1 = ["What is the capital of France?"]
    text_2 = ["Paris is the capital and largest city of France."]

    # Add garbage pairs if batch > 1
    if batch > 1:
        garbage_q, garbage_d = generate_garbage_pairs(batch - 1)
        text_1.extend(garbage_q)
        text_2.extend(garbage_d)

    print(f"\nQuery: {text_1[0]}")
    print(f"Document: {text_2[0]}")
    if batch > 1:
        print(f"(+ {batch - 1} garbage pairs)")

    response = await call_score(
        client=client,
        model=model,
        text_1=text_1,
        text_2=text_2,
        endpoint_name=endpoint_name,
    )

    print("\nResponse:")
    print(json.dumps(response, indent=2))

    if "data" in response and response["data"]:
        print(f"\nFirst pair score: {response['data'][0].get('score', 'N/A')}")


async def demo_instruct(
    client: Serverless, model: str, endpoint_name: str, batch: int
) -> None:
    """Demo using the instruction format for reranker models."""
    print("=" * 60)
    print("INSTRUCT FORMAT DEMO")
    print("=" * 60)

    instruct = (
        "Given a query and a document, determine if the document is relevant "
        "to answering the query. Reply 'yes' if relevant, 'no' if not."
    )
    query = "What programming languages are good for data science?"
    document = (
        "Python is widely used in data science due to its rich ecosystem "
        "of libraries like pandas, numpy, and scikit-learn."
    )

    text_1 = [
        f"<|im_start|>system\n{instruct}<|im_end|>\n"
        f"<|im_start|>user\n<Query>: {query}<|im_end|>"
    ]
    text_2 = [
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ]

    # Add garbage pairs if batch > 1
    if batch > 1:
        garbage_q, garbage_d = generate_garbage_pairs(batch - 1)
        text_1.extend(garbage_q)
        text_2.extend(garbage_d)

    print(f"\nInstruction: {instruct[:60]}...")
    print(f"Query: {query}")
    print(f"Document: {document[:60]}...")
    if batch > 1:
        print(f"(+ {batch - 1} garbage pairs)")

    response = await call_score(
        client=client,
        model=model,
        text_1=text_1,
        text_2=text_2,
        endpoint_name=endpoint_name,
    )

    print("\nResponse:")
    print(json.dumps(response, indent=2))

    if "data" in response and response["data"]:
        print(f"\nFirst pair score: {response['data'][0].get('score', 'N/A')}")


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
    p.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Number of pairs to send (first is real, rest are garbage)",
    )

    modes = p.add_mutually_exclusive_group(required=False)
    modes.add_argument(
        "--simple", action="store_true", help="Simple query-document scoring"
    )
    modes.add_argument(
        "--instruct", action="store_true", help="Instruction-formatted scoring"
    )
    return p


async def main_async():
    args = build_arg_parser().parse_args()

    selected = sum([args.simple, args.instruct])
    if selected == 0:
        print("Please specify a test mode:")
        print("  --simple   : Simple query-document scoring")
        print("  --instruct : Instruction-formatted scoring")
        print("\nOptional:")
        print("  --batch N  : Send N pairs (first real, rest garbage)")
        print(
            f"\nExample: python -m workers.vllm_score.client --simple --batch 5 --endpoint my-endpoint"
        )
        sys.exit(1)

    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Batch size: {args.batch}")

    try:
        async with Serverless() as client:
            if args.simple:
                await demo_simple(client, args.model, args.endpoint, args.batch)
            elif args.instruct:
                await demo_instruct(client, args.model, args.endpoint, args.batch)

    except Exception as e:
        log.error("Error during test: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
