"""
PyWorker Client - Local Proxy for Vast.ai Serverless Endpoints

This client starts a local HTTP proxy that handles all Vast.ai routing complexity:
1. Automatically calls /route/ to get worker assignment
2. Wraps requests in auth_data + payload format
3. Streams responses transparently

Usage as a module:
    from client import VastClient
    import asyncio

    async def main():
        client = VastClient(endpoint_name="my-endpoint", api_key="YOUR_KEY")
        await client.start()

        # Use client.url with any SDK
        from openai import OpenAI
        openai_client = OpenAI(base_url=f"{client.url}/v1", api_key="not-used")

        response = openai_client.chat.completions.create(
            model="llama-2-7b",
            messages=[{"role": "user", "content": "Hello!"}]
        )

    asyncio.run(main())

Usage as CLI:
    python client.py --endpoint my-endpoint --api-key YOUR_KEY
    # Or with account key (auto-fetches endpoint key):
    python client.py --endpoint my-endpoint --account-key YOUR_ACCOUNT_KEY
    # Or interactive mode:
    python client.py

Then point your app at localhost:8010!

Workload/Cost:
- Specify via X-Serverless-Cost header
- Used for routing and queue estimation
- Defaults to 1.0 if not specified
"""
import argparse
import json
import logging
import os
import sys
import ssl
from pathlib import Path
from typing import Optional, Dict, Any, List
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
import asyncio
from urllib.parse import urlencode, urlparse, parse_qs
import certifi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


class VastClient:
    """
    Local proxy server for Vast.ai serverless endpoints.

    Start the proxy and use client.url with any HTTP client or SDK.
    """

    def __init__(
        self,
        endpoint_name: str,
        api_key: str,
        port: int = 8010,
        host: str = "127.0.0.1",
        autoscaler_url: str = "https://run.vast.ai",
        instance: str = "prod",
        timeout: Optional[int] = None,
        max_connections: int = 100,
        max_connections_per_host: int = 20,
    ):
        """
        Initialize Vast.ai client proxy.

        Args:
            endpoint_name: Name of your Vast.ai endpoint
            api_key: Endpoint API key (not your account API key!)
            port: Local proxy port (default: 8010)
            host: Local proxy host (default: 127.0.0.1)
            autoscaler_url: Autoscaler URL (default: https://run.vast.ai)
            instance: Instance name (prod, alpha, candidate)
            timeout: Request timeout in seconds (default: None, no limit)
            max_connections: Max total connections (default: 100)
            max_connections_per_host: Max connections per host (default: 20)
        """
        self.endpoint_name = endpoint_name
        self.api_key = api_key
        self.port = port
        self.host = host
        self.autoscaler_url = autoscaler_url.rstrip("/")
        self.instance = instance
        self.timeout = timeout
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host

        # HTTP client for routing calls
        self._session: Optional[ClientSession] = None

        # Server components
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

        log.debug(f"Initialized VastClient for endpoint: {endpoint_name}")

    @property
    def url(self) -> str:
        """Get the local proxy URL to use with SDKs."""
        return f"http://{self.host}:{self.port}"

    async def _ensure_session(self):
        """Ensure HTTP session is created."""
        if self._session is None:
            timeout = ClientTimeout(total=self.timeout) if self.timeout else ClientTimeout()

            # Create SSL context that uses certifi's certificate bundle
            ssl_context = ssl.create_default_context(cafile=certifi.where())

            connector = TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections_per_host,
                ssl=ssl_context
            )
            self._session = ClientSession(timeout=timeout, connector=connector)

    async def route(self, workload: float = 1.0, max_retries: int = 2) -> Optional[Dict[str, Any]]:
        """
        Call /route/ to get worker assignment.

        Args:
            workload: Estimated workload units (default: 1.0)
            max_retries: Max retries for transient failures (default: 2)

        Returns:
            Dict with worker URL, signature, and routing info
        """
        await self._ensure_session()
        assert self._session is not None, "Session should be initialized"

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                async with self._session.post(
                    f"{self.autoscaler_url}/route/",
                    json={
                        "endpoint": self.endpoint_name,
                        "cost": workload,
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    timeout=ClientTimeout(total=10),
                ) as response:
                    # Read response body first
                    text = await response.text()

                    # Don't retry client errors (4xx) - these are intentional signals
                    if 400 <= response.status < 500:
                        log.error(f"Route failed: {response.status} - {text}")
                        if response.status == 401:
                            log.error("")
                            log.error("HINT: 401 Unauthorized usually means:")
                            log.error("  - You're using the wrong API key type")
                            log.error("  - Endpoint API key is required, not account API key")
                            log.error("  - Use --account-key to auto-fetch the correct key")
                            log.error("")
                        return None

                    # Retry on server errors (5xx)
                    if response.status >= 500:
                        last_error = f"Server error: {response.status} - {text}"
                        if attempt < max_retries:
                            wait_time = 0.5 * (attempt + 1)
                            log.warning(f"Route returned {response.status}, retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        log.error(f"Route failed after {max_retries + 1} attempts: {last_error}")
                        return None

                    # Try to parse as JSON (be lenient with Content-Type)
                    try:
                        data = json.loads(text)
                        log.debug(f"Got worker assignment: {data.get('url', 'unknown')}")
                        log.debug(f"Routing response: {data}")
                        return data
                    except json.JSONDecodeError as e:
                        # Only log Content-Type if JSON parsing fails
                        content_type = response.content_type or ""
                        log.error(f"Failed to decode JSON response (Content-Type: {content_type}): {e}")
                        log.error(f"Response body: {text[:500]}")
                        return None

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    wait_time = 0.5 * (attempt + 1)
                    log.warning(f"Route error: {e}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                log.error(f"Route failed after {max_retries + 1} attempts: {e}", exc_info=True)
                return None

        return None

    async def _handle_request(self, request: web.Request) -> web.StreamResponse:
        """Forward incoming requests to Vast.ai workers with streaming support."""
        path = request.path_qs  # Include query parameters
        method = request.method

        log.info(f"{method} {path}")

        try:
            # Extract workload from X-Serverless-Cost header if present
            workload = 1.0
            if "X-Serverless-Cost" in request.headers:
                try:
                    workload = float(request.headers["X-Serverless-Cost"])
                    log.debug(f"Using workload from X-Serverless-Cost header: {workload}")
                except (ValueError, TypeError):
                    log.warning(f"Invalid X-Serverless-Cost header: {request.headers['X-Serverless-Cost']}")

            # Get worker assignment
            routing_info = await self.route(workload)
            if not routing_info:
                return web.Response(
                    status=503,
                    text="Failed to get worker assignment from autoscaler\n"
                         "Possible causes:\n"
                         "  - Wrong API key (are you using endpoint key, not account key?)\n"
                         "  - Endpoint has no healthy workers\n"
                         "  - Endpoint name is incorrect\n"
                         "Check: https://console.vast.ai/endpoints"
                )

            # Construct auth_data from routing response
            worker_url = routing_info["url"]
            auth_data = {
                "cost": routing_info.get("cost", workload),
                "endpoint": routing_info.get("endpoint", self.endpoint_name),
                "reqnum": routing_info.get("reqnum", 0),
                "request_idx": routing_info.get("request_idx", 0),
                "signature": routing_info.get("signature", ""),
                "url": worker_url,
            }

            log.debug(f"Auth data for signature verification: {auth_data}")
            log.debug(f"Forwarding to worker: {worker_url}{path}")

            # Read request body if present
            json_data = None
            body_bytes = None

            if request.can_read_body:
                body_bytes = await request.read()
                if body_bytes:
                    try:
                        json_data = json.loads(body_bytes)
                    except:
                        pass

            await self._ensure_session()
            assert self._session is not None, "Session should be initialized"

            # Handle GET/DELETE/HEAD differently (no body, use query params)
            if method in ["GET", "DELETE", "HEAD"]:
                # Parse path to separate base path from existing query parameters
                parsed = urlparse(path)
                base_path = parsed.path
                existing_params = parse_qs(parsed.query, keep_blank_values=True)

                # Flatten existing params (parse_qs returns lists)
                flat_existing = {k: v[0] if len(v) == 1 else v for k, v in existing_params.items()}

                # Serverless auth params (prefixed to avoid conflicts)
                query_params = {
                    "serverless_cost": auth_data["cost"],
                    "serverless_endpoint": auth_data["endpoint"],
                    "serverless_reqnum": str(auth_data["reqnum"]),
                    "serverless_request_idx": str(auth_data["request_idx"]),
                    "serverless_signature": auth_data["signature"],
                    "serverless_url": auth_data["url"],
                }

                # Add existing query params from path (unprefixed - these go to backend)
                query_params.update(flat_existing)

                # Add payload fields as additional query params (unprefixed - these go to backend)
                if json_data:
                    query_params.update(json_data)

                # Build full URL with query params
                full_url = f"{worker_url}{base_path}?{urlencode(query_params)}"

                async with self._session.request(
                    method,
                    full_url,
                    headers=dict(request.headers),
                ) as worker_response:
                    return await self._stream_response(worker_response, request)

            else:
                # POST/PUT/PATCH: use JSON body
                payload_data = {
                    "auth_data": auth_data,
                    "payload": json_data or {},
                }

                full_url = f"{worker_url}{path}"

                # Filter headers - only forward safe headers
                forward_headers = {}
                safe_headers = {
                    "user-agent", "accept", "accept-encoding", "accept-language",
                    "x-serverless-cost"  # Custom workload header
                }
                for key, value in request.headers.items():
                    if key.lower() in safe_headers:
                        forward_headers[key] = value

                # Ensure Content-Type is set to application/json
                forward_headers["Content-Type"] = "application/json"

                log.debug(f"Sending to worker: {full_url}")
                log.debug(f"Payload: {payload_data}")

                async with self._session.request(
                    method,
                    full_url,
                    json=payload_data,
                    headers=forward_headers,
                ) as worker_response:
                    return await self._stream_response(worker_response, request)

        except Exception as e:
            log.error(f"Request failed: {e}", exc_info=True)
            return web.Response(
                status=500,
                text=f"Proxy error: {str(e)}",
            )

    async def _stream_response(
        self,
        worker_response: Any,
        client_request: web.Request
    ) -> web.StreamResponse:
        """
        Stream response from worker to client.

        Detects streaming responses and handles them appropriately.
        """
        # Check if response is streaming
        content_type = worker_response.content_type or ""
        transfer_encoding = worker_response.headers.get("Transfer-Encoding", "")

        is_streaming = (
            content_type == "text/event-stream"
            or content_type == "application/x-ndjson"
            or transfer_encoding == "chunked"
            or "stream" in content_type.lower()
        )

        # Prepare response headers (exclude hop-by-hop headers)
        response_headers = {}
        hop_by_hop = {
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"
        }
        for key, value in worker_response.headers.items():
            if key.lower() not in hop_by_hop:
                response_headers[key] = value

        if is_streaming:
            log.debug("Streaming response detected")

            # Create streaming response
            response = web.StreamResponse(
                status=worker_response.status,
                headers=response_headers,
            )

            await response.prepare(client_request)

            # Stream chunks
            try:
                async for chunk in worker_response.content.iter_any():
                    if chunk:
                        await response.write(chunk)
            except Exception as e:
                log.error(f"Streaming error: {e}")
            finally:
                await response.write_eof()

            return response
        else:
            # Non-streaming: read full response
            body = await worker_response.read()
            return web.Response(
                body=body,
                status=worker_response.status,
                headers=response_headers,
            )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint for load balancers and monitoring."""
        return web.json_response({
            "status": "healthy",
            "endpoint": self.endpoint_name,
        })

    async def start(self):
        """Start the proxy server."""
        # Create app
        self._app = web.Application()
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_route("*", "/{path:.*}", self._handle_request)

        # Setup runner
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        # Start site
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        log.info(f"Vast.ai proxy started on {self.url}")
        log.info(f"Forwarding to endpoint: {self.endpoint_name}")
        log.info(f"Use {self.url} as your base URL for API calls")

    async def stop(self):
        """Stop the proxy server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        if self._session:
            await self._session.close()
            self._session = None

        log.info("Proxy stopped")

    async def run_forever(self):
        """Start the proxy and keep it running."""
        await self.start()

        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            log.info("Shutting down proxy...")
        finally:
            await self.stop()


def get_api_key_from_file() -> Optional[str]:
    """Read API key from ~/.vast_api_key file (vastai CLI compatible)"""
    try:
        key_file = Path.home() / ".vast_api_key"
        if key_file.exists():
            key = key_file.read_text().strip()
            if key:
                log.debug(f"Loaded API key from {key_file}")
                return key
    except Exception as e:
        log.debug(f"Could not read ~/.vast_api_key: {e}")
    return None


def fetch_endpoint_key(account_key: str, endpoint_name: str, instance: str = "prod") -> Optional[str]:
    """Fetch endpoint API key using account API key"""
    try:
        from utils.endpoint_util import Endpoint
        log.info(f"Fetching endpoint API key for '{endpoint_name}'...")
        endpoint_key = Endpoint.get_endpoint_api_key(endpoint_name, account_key, instance)
        if endpoint_key:
            log.info("Successfully retrieved endpoint API key")
            return endpoint_key
        else:
            log.error(f"Failed to fetch endpoint key for '{endpoint_name}'")
            log.error("Make sure the endpoint exists and you have access to it")
            return None
    except ImportError:
        log.error("ERROR: utils.endpoint_util not found. Cannot auto-fetch endpoint key.")
        log.error("Install with: pip install requests")
        return None
    except Exception as e:
        log.error(f"Error fetching endpoint key: {e}")
        return None


def list_endpoints(account_key: str, instance: str = "prod") -> List[str]:
    """List all available endpoints"""
    try:
        import requests
        from utils.endpoint_util import Endpoint
        headers = {"Authorization": f"Bearer {account_key}"}
        url = f"{Endpoint.get_server_url(instance)}?autoscaler_instance={instance}"

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            log.error(f"Failed to fetch endpoints: {response.status_code}")
            return []

        data = response.json()
        results = data.get("results", [])
        endpoints = [item.get("endpoint_name") for item in results if item.get("endpoint_name")]
        return sorted(endpoints)
    except Exception as e:
        log.error(f"Error listing endpoints: {e}")
        return []


def interactive_mode() -> tuple[str, str, int, str]:
    """Interactive mode - prompts for all required info"""
    print("\n" + "="*60)
    print("  PyWorker Client - Interactive Setup")
    print("="*60 + "\n")

    # Get account key
    print("First, we need your Vast.ai account API key.")
    print("(Get it from: https://console.vast.ai/account)")
    print()
    account_key = input("Enter account API key: ").strip()

    if not account_key:
        print("\nERROR: Account API key is required")
        sys.exit(1)

    # List endpoints
    print("\nFetching your endpoints...")
    endpoints = list_endpoints(account_key)

    if not endpoints:
        print("\nNo endpoints found or failed to fetch.")
        print("Make sure you have created endpoints in console.vast.ai")
        sys.exit(1)

    # Show endpoints
    print(f"\nAvailable endpoints ({len(endpoints)}):")
    for i, ep in enumerate(endpoints, 1):
        print(f"  {i}. {ep}")
    print()

    # Select endpoint
    while True:
        try:
            choice = input(f"Select endpoint [1-{len(endpoints)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(endpoints):
                endpoint_name = endpoints[idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(endpoints)}")
        except (ValueError, EOFError, KeyboardInterrupt):
            print("\nCancelled")
            sys.exit(0)

    # Get port
    print()
    port_input = input("Local proxy port [8010]: ").strip()
    port = int(port_input) if port_input else 8010

    # Fetch endpoint key
    print()
    endpoint_key = fetch_endpoint_key(account_key, endpoint_name)
    if not endpoint_key:
        sys.exit(1)

    print("\n" + "="*60)
    print(f"  Starting proxy for: {endpoint_name}")
    print(f"  Listening on: http://127.0.0.1:{port}")
    print("="*60 + "\n")

    return endpoint_name, endpoint_key, port, "https://run.vast.ai"


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Vast.ai Client Proxy - Forward requests to serverless endpoints",
        epilog="""
Examples:
  # Interactive mode (easiest)
  python client.py

  # With endpoint API key
  python client.py --endpoint my-endpoint --api-key ENDPOINT_KEY

  # With account API key (auto-fetches endpoint key)
  python client.py --endpoint my-endpoint --account-key ACCOUNT_KEY

  # List available endpoints
  python client.py --list --account-key ACCOUNT_KEY

  # Use environment variables
  export VAST_ACCOUNT_KEY="your-account-key"
  export VAST_ENDPOINT="my-endpoint"
  python client.py

  # Or use ~/.vast_api_key file (vastai CLI compatible)
  echo "your-account-key" > ~/.vast_api_key
  python client.py --endpoint my-endpoint
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--endpoint",
        help="Vast.ai endpoint name (or set VAST_ENDPOINT env var)"
    )
    parser.add_argument(
        "--api-key",
        help="Endpoint API key (or set VAST_API_KEY env var)"
    )
    parser.add_argument(
        "--account-key",
        help="Account API key - will auto-fetch endpoint key (or set VAST_ACCOUNT_KEY env var, or use ~/.vast_api_key)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available endpoints (requires account key)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Local proxy port (default: 8010)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Local proxy host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--autoscaler-url",
        default="https://run.vast.ai",
        help="Autoscaler URL (default: https://run.vast.ai)"
    )
    parser.add_argument(
        "--instance",
        default="prod",
        choices=["prod", "alpha", "candidate"],
        help="Vast.ai instance (default: prod)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ["PYWORKER_TIMEOUT"]) if os.environ.get("PYWORKER_TIMEOUT") else None,
        help="Request timeout in seconds (default: no limit, or PYWORKER_TIMEOUT env var)"
    )
    parser.add_argument(
        "--max-connections",
        type=int,
        default=int(os.environ.get("PYWORKER_MAX_CONNECTIONS", 100)),
        help="Max total connections (default: 100, or PYWORKER_MAX_CONNECTIONS env var)"
    )
    parser.add_argument(
        "--max-connections-per-host",
        type=int,
        default=int(os.environ.get("PYWORKER_MAX_CONNECTIONS_PER_HOST", 20)),
        help="Max connections per host (default: 20, or PYWORKER_MAX_CONNECTIONS_PER_HOST env var)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Priority: CLI flags > env vars > ~/.vast_api_key file
    account_key = (
        args.account_key
        or os.environ.get("VAST_ACCOUNT_KEY")
        or get_api_key_from_file()
    )

    endpoint_key = args.api_key or os.environ.get("VAST_API_KEY")
    endpoint_name = args.endpoint or os.environ.get("VAST_ENDPOINT")

    # Handle --list
    if args.list:
        if not account_key:
            log.error("ERROR: --list requires account key")
            log.error("Use: --account-key KEY or set VAST_ACCOUNT_KEY or create ~/.vast_api_key")
            sys.exit(1)

        log.info("Fetching endpoints...")
        endpoints = list_endpoints(account_key, args.instance)

        if not endpoints:
            log.error("No endpoints found")
            sys.exit(1)

        print(f"\nAvailable endpoints ({len(endpoints)}):")
        for ep in endpoints:
            print(f"  • {ep}")
        print()
        sys.exit(0)

    # Interactive mode if no endpoint specified
    if not endpoint_name:
        try:
            endpoint_name, endpoint_key, port, autoscaler_url = interactive_mode()
            args.port = port
            args.autoscaler_url = autoscaler_url
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled")
            sys.exit(0)

    # Auto-fetch endpoint key if we have account key but no endpoint key
    if not endpoint_key and account_key:
        log.info("No endpoint API key provided, fetching using account key...")
        endpoint_key = fetch_endpoint_key(account_key, endpoint_name, args.instance)
        if not endpoint_key:
            sys.exit(1)

    # Validate we have everything needed
    if not endpoint_key:
        log.error("ERROR: Endpoint API key required!")
        log.error("")
        log.error("Options:")
        log.error("  1. Use --api-key with endpoint API key")
        log.error("  2. Use --account-key to auto-fetch endpoint key")
        log.error("  3. Set VAST_API_KEY or VAST_ACCOUNT_KEY env var")
        log.error("  4. Create ~/.vast_api_key with account key")
        log.error("  5. Run without arguments for interactive mode")
        log.error("")
        log.error("Get your account key from: https://console.vast.ai/account")
        sys.exit(1)

    # Start proxy
    log.info(f"Starting proxy for endpoint: {endpoint_name}")
    client = VastClient(
        endpoint_name=endpoint_name,
        api_key=endpoint_key,
        port=args.port,
        host=args.host,
        autoscaler_url=args.autoscaler_url,
        timeout=args.timeout,
        max_connections=args.max_connections,
        max_connections_per_host=args.max_connections_per_host,
    )

    asyncio.run(client.run_forever())


if __name__ == "__main__":
    main()
