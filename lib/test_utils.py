"""
Load testing utilities for PyWorker workers.

This script simulates multiple concurrent clients making requests to a Vast.ai endpoint,
using the same test payloads as the benchmark functions.

Usage:
    [MODEL_NAME="Qwen/Qwen3-8B"] python -m lib.test_utils -k YOUR_API_KEY -e endpoint-name -b benchmarks.openai_chat -n 100 -rps 10
"""
import logging
import os
import time
import argparse
import importlib
from typing import Callable, List, Dict, Tuple, Any
from time import sleep
import threading
from enum import Enum
from collections import Counter
from dataclasses import dataclass, field, asdict
from urllib.parse import urljoin
from utils.endpoint_util import Endpoint
from utils.ssl import get_cert_file_path
import requests

from lib.data_types import AuthData

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class ClientStatus(Enum):
    FetchEndpoint = 1
    Generating = 2
    Done = 3
    Error = 4


total_success = 0
last_res = []
stop_event = threading.Event()
start_time = time.time()


def print_truncate_res(res: str):
    if len(res) > 150:
        print(f"{res[:50]}....{res[-100:]}")
    else:
        print(res)


@dataclass
class ClientState:
    endpoint_group_name: str
    api_key: str
    server_url: str
    instance: str
    get_test_request: Callable[[], Tuple[str, Dict[str, Any], float]]
    url: str = ""
    status: ClientStatus = ClientStatus.FetchEndpoint
    as_error: List[str] = field(default_factory=list)
    infer_error: List[str] = field(default_factory=list)
    conn_errors: Counter = field(default_factory=Counter)

    def make_call(self):
        self.status = ClientStatus.FetchEndpoint
        if not self.api_key:
            self.as_error.append(
                f"Endpoint {self.endpoint_group_name} not found for API key",
            )
            self.status = ClientStatus.Error
            return

        # Get test request from benchmark module
        worker_endpoint, payload, workload = self.get_test_request()

        route_payload = {
            "endpoint": self.endpoint_group_name,
            "api_key": self.api_key,
            "cost": workload,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            urljoin(self.server_url, "/route/"),
            json=route_payload,
            headers=headers,
            timeout=4,
        )
        if response.status_code != 200:
            self.as_error.append(
                f"code: {response.status_code}, body: {response.text}",
            )
            self.status = ClientStatus.Error
            return
        message = response.json()
        worker_address = message["url"]
        req_data = dict(
            payload=payload,
            auth_data=asdict(AuthData.from_json_msg(message)),
        )
        self.url = worker_address
        url = urljoin(worker_address, worker_endpoint)
        self.status = ClientStatus.Generating

        response = requests.post(
            url,
            json=req_data,
            verify=get_cert_file_path(),
        )
        if response.status_code != 200:
            self.infer_error.append(
                f"code: {response.status_code}, body: {response.text}, url: {url}",
            )
            self.status = ClientStatus.Error
            return
        res = str(response.json())
        global total_success
        global last_res
        total_success += 1
        last_res.append(res)
        self.status = ClientStatus.Done

    def simulate_user(self) -> None:
        try:
            self.make_call()
        except Exception as e:
            print(e)
            self.status = ClientStatus.Error
            _ = e
            self.conn_errors[self.url] += 1


def print_state(clients: List[ClientState], num_clients: int) -> None:
    print("starting up...")
    sleep(2)
    center_size = 14
    global start_time
    while len(clients) < num_clients or (
        any(
            map(
                lambda client: client.status
                in [ClientStatus.FetchEndpoint, ClientStatus.Generating],
                clients,
            )
        )
    ):
        sleep(0.5)
        os.system("clear")
        print(
            " | ".join(
                [member.name.center(center_size) for member in ClientStatus]
                + [
                    item.center(center_size)
                    for item in [
                        "urls",
                        "as_error",
                        "infer_error",
                        "conn_error",
                        "total_success",
                    ]
                ]
            )
        )
        unique_urls = len(set([c.url for c in clients if c.url != ""]))
        as_errors = sum(
            map(
                lambda client: len(client.as_error),
                [client for client in clients],
            )
        )
        infer_errors = sum(
            map(
                lambda client: len(client.infer_error),
                [client for client in clients],
            )
        )
        conn_errors = sum([client.conn_errors for client in clients], start=Counter())
        conn_errors_str = ",".join(map(str, conn_errors.values())) or "0"
        elapsed = time.time() - start_time
        print(
            " | ".join(
                map(
                    lambda item: str(item).center(center_size),
                    [
                        len(list(filter(lambda x: x.status == member, clients)))
                        for member in ClientStatus
                    ]
                    + [
                        unique_urls,
                        as_errors,
                        infer_errors,
                        conn_errors_str,
                        f"{total_success}({((total_success/elapsed) * 60):.2f}/minute)",
                    ],
                )
            )
        )
        if conn_errors:
            print("conn_errors:")
            for url, count in conn_errors.items():
                print(url.ljust(28), ": ", str(count))
        elapsed = time.time() - start_time
        print(f"\n elapsed: {int(elapsed // 60)}:{int(elapsed % 60)}")
        if last_res:
            for i, res in enumerate(last_res[-10:]):
                print_truncate_res(f"res #{1+i+max(len(last_res )-10,0)}: {res}")
        if stop_event.is_set():
            print("\n### waiting for existing connections to close ###")


def run_test(
    num_requests: int,
    requests_per_second: int,
    endpoint_group_name: str,
    api_key: str,
    server_url: str,
    get_test_request: Callable[[], Tuple[str, Dict[str, Any], float]],
    instance: str,
):
    threads = []

    clients = []
    print_thread = threading.Thread(target=print_state, args=(clients, num_requests))
    print_thread.daemon = True  # makes threads get killed on program exit
    print_thread.start()
    endpoint_api_key = Endpoint.get_endpoint_api_key(
        endpoint_name=endpoint_group_name, account_api_key=api_key, instance=instance
    )
    if not endpoint_api_key:
        log.debug(f"Endpoint {endpoint_group_name} not found for API key")
        return
    try:
        for _ in range(num_requests):
            client = ClientState(
                endpoint_group_name=endpoint_group_name,
                api_key=endpoint_api_key,
                server_url=server_url,
                get_test_request=get_test_request,
                instance=instance,
            )
            clients.append(client)
            thread = threading.Thread(target=client.simulate_user, args=())
            threads.append(thread)
            thread.start()
            sleep(1 / requests_per_second)
        for thread in threads:
            thread.join()
        print("done spawning workers")
    except KeyboardInterrupt:
        stop_event.set()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Test inference endpoint with load")
    arg_parser.add_argument(
        "-k", dest="api_key", type=str, required=True, help="Your vast account API key"
    )
    arg_parser.add_argument(
        "-e",
        dest="endpoint_group_name",
        type=str,
        required=True,
        help="Endpoint group name",
    )
    arg_parser.add_argument(
        "-b",
        dest="benchmark_module",
        type=str,
        required=True,
        help="Benchmark module (e.g., benchmarks.openai_chat, benchmarks.tgi, benchmarks.comfyui)",
    )
    arg_parser.add_argument(
        "-n",
        dest="num_requests",
        type=int,
        required=True,
        help="total number of requests",
    )
    arg_parser.add_argument(
        "-rps",
        dest="requests_per_second",
        type=float,
        required=True,
        help="requests per second",
    )
    arg_parser.add_argument(
        "-i",
        dest="instance",
        type=str,
        default="prod",
        help="Autoscaler shard to run the command against, default: prod",
    )

    args = arg_parser.parse_args()

    # Load benchmark module
    try:
        benchmark_module = importlib.import_module(args.benchmark_module)
        if not hasattr(benchmark_module, 'get_test_request'):
            raise ValueError(
                f"Benchmark module {args.benchmark_module} does not have get_test_request() function"
            )
        get_test_request = benchmark_module.get_test_request
    except Exception as e:
        log.error(f"Failed to load benchmark module '{args.benchmark_module}': {e}")
        exit(1)

    # Determine server URL
    server_url = {
        "prod": "https://run.vast.ai",
        "alpha": "https://run-alpha.vast.ai",
        "candidate": "https://run-candidate.vast.ai",
        "local": "http://localhost:8080",
    }.get(args.instance, "http://localhost:8080")

    run_test(
        num_requests=args.num_requests,
        requests_per_second=args.requests_per_second,
        api_key=args.api_key,
        server_url=server_url,
        endpoint_group_name=args.endpoint_group_name,
        get_test_request=get_test_request,
        instance=args.instance,
    )
