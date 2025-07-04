import logging
import sys
import json
from urllib.parse import urljoin
import requests
from utils.endpoint_util import Endpoint

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


def call_generic_completions(endpoint_group_name: str, api_key: str, server_url: str) -> None:
    WORKER_ENDPOINT = "/generic"
    COST = 100
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    response = requests.post(
        urljoin(server_url, "/route/"),
        json=route_payload,
        timeout=4,
    )
    response.raise_for_status()  # Raise an exception for bad status codes
    message = response.json()
    url = message["url"]

    auth_data = dict(
        signature=message["signature"],
        cost=message["cost"],
        endpoint=message["endpoint"],
        reqnum=message["reqnum"],
        url=url,
    )

    payload = dict(
        input=dict(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            prompt="The capital of France is",
            max_tokens=10,
            temperature=0
        ),
        endpoint="/v1/completions",
        method="POST"
    )

    req_data = dict(payload=payload, auth_data=auth_data)
    url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {url}")
    response = requests.post(url, json=req_data)
    response.raise_for_status()
    res = response.json()
    print(res)

def call_generic_chat_completions(endpoint_group_name: str, api_key: str, server_url: str) -> None:
    WORKER_ENDPOINT = "/generic"
    COST = 100
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    response = requests.post(
        urljoin(server_url, "/route/"),
        json=route_payload,
        timeout=4,
    )
    response.raise_for_status()  # Raise an exception for bad status codes
    message = response.json()
    url = message["url"]

    auth_data = dict(
        signature=message["signature"],
        cost=message["cost"],
        endpoint=message["endpoint"],
        reqnum=message["reqnum"],
        url=url,
    )

    payload = dict(
        input=dict(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            messages=[
                {"role": "user", "content": "The capital of France is"}
            ],
            max_tokens=250,
            temperature=0
        ),
        endpoint="/v1/chat/completions",
        method="POST"
    )

    req_data = dict(payload=payload, auth_data=auth_data)
    url = urljoin(url, WORKER_ENDPOINT)
    print(f"url: {url}")
    response = requests.post(url, json=req_data)
    response.raise_for_status()
    res = response.json()
    print(res)


def call_generate_stream(
    endpoint_group_name: str, api_key: str, server_url: str
) -> None:
    WORKER_ENDPOINT = "/generate_stream"
    COST = 100
    route_payload = {
        "endpoint": endpoint_group_name,
        "api_key": api_key,
        "cost": COST,
    }
    response = requests.post(
        urljoin(server_url, "/route/"),
        json=route_payload,
        timeout=4,
    )
    response.raise_for_status()  # Raise an exception for bad status codes
    message = response.json()
    url = message["url"]
    print(f"url: {url}")
    auth_data = dict(
        signature=message["signature"],
        cost=message["cost"],
        endpoint=message["endpoint"],
        reqnum=message["reqnum"],
        url=message["url"],
    )
    payload = dict(inputs="tell me about dogs", parameters=dict(max_new_tokens=500))
    req_data = dict(payload=payload, auth_data=auth_data)
    url = urljoin(url, WORKER_ENDPOINT)
    response = requests.post(url, json=req_data, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    for line in response.iter_lines():
        payload = line.decode().lstrip("data:").rstrip()
        if payload:
            try:
                data = json.loads(payload)
                print(data["token"]["text"], end="")
                sys.stdout.flush()
            except (json.JSONDecodeError, KeyError) as e:
                log.warning(f"Failed to parse streaming response: {e}")
                continue
    print()


if __name__ == "__main__":
    from lib.test_utils import test_args

    args = test_args.parse_args()

    endpoint_api_key = Endpoint.get_endpoint_api_key(
        endpoint_name=args.endpoint_group_name,
        account_api_key=args.api_key,
    )
    if endpoint_api_key:
        try:
            print("Calling /generic with target /v1/completions")
            call_generic_completions(
                api_key=endpoint_api_key,
                endpoint_group_name=args.endpoint_group_name,
                server_url=args.server_url,
            )
            print("Calling /generic with target /v1/chat/completions")
            call_generic_chat_completions(
                api_key=endpoint_api_key,
                endpoint_group_name=args.endpoint_group_name,
                server_url=args.server_url,
            )
        except Exception as e:
            log.error(f"Error during API call: {e}")
    else:
        log.error(f"Failed to get API key for endpoint {args.endpoint_group_name} ")
