import os
import json
import time
import base64
import importlib
import dataclasses
import logging
from asyncio import wait, sleep, gather, Semaphore, FIRST_COMPLETED, create_task
from typing import Tuple, Awaitable, NoReturn, Callable, Optional
from functools import cached_property

from aiohttp import web, ClientResponse, ClientSession, ClientConnectorError, ClientTimeout, TCPConnector
import asyncio

from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA

from lib.metrics import Metrics
from lib.data_types import (
    AuthData,
    JsonDataException,
    RequestMetrics,
)

VERSION = "0.3.0"

log = logging.getLogger(__file__)

# Configuration constants - can be overridden via environment variables
BENCHMARK_INDICATOR_FILE = os.environ.get("PYWORKER_BENCHMARK_CACHE_FILE", ".has_benchmark")
MAX_PUBKEY_FETCH_ATTEMPTS = int(os.environ.get("PYWORKER_PUBKEY_MAX_RETRIES", "3"))
HEALTHCHECK_RETRY_INTERVAL = int(os.environ.get("PYWORKER_HEALTHCHECK_RETRY_INTERVAL", "5"))
HEALTHCHECK_POLL_INTERVAL = int(os.environ.get("PYWORKER_HEALTHCHECK_POLL_INTERVAL", "10"))
HEALTHCHECK_TIMEOUT = int(os.environ.get("PYWORKER_HEALTHCHECK_TIMEOUT", "10"))
HEALTHCHECK_CONSECUTIVE_FAILURES = int(os.environ.get("PYWORKER_HEALTHCHECK_CONSECUTIVE_FAILURES", "3"))
PUBKEY_FETCH_TIMEOUT = int(os.environ.get("PYWORKER_PUBKEY_TIMEOUT", "10"))
METRICS_RETRY_DELAY = int(os.environ.get("PYWORKER_METRICS_RETRY_DELAY", "2"))
CONNECTION_LIMIT = int(os.environ.get("PYWORKER_CONNECTION_LIMIT", "100"))
CONNECTION_LIMIT_PER_HOST = int(os.environ.get("PYWORKER_CONNECTION_LIMIT_PER_HOST", "20"))


def create_tcp_connector(force_close: bool = True) -> TCPConnector:
    """Create a standard TCP connector with consistent settings"""
    return TCPConnector(
        force_close=force_close,
        enable_cleanup_closed=True,
    )


@dataclasses.dataclass
class Backend:
    """
    Simplified backend that acts as a pass-through proxy.

    This class is responsible for:
    1. Forwarding requests to the model server without transformation
    2. Tracking metrics and reporting to autoscaler
    3. Running benchmarks using a custom benchmark function
    """

    backend_url: str
    benchmark_func: Optional[Callable[[str, ClientSession], Awaitable[float]]]
    healthcheck_endpoint: Optional[str] = dataclasses.field(
        default_factory=lambda: os.environ.get("PYWORKER_HEALTHCHECK_ENDPOINT", "/health")
    )
    allow_parallel_requests: bool = dataclasses.field(
        default_factory=lambda: os.environ.get("PYWORKER_ALLOW_PARALLEL", "true").lower() == "true"
    )
    max_wait_time: float = dataclasses.field(
        default_factory=lambda: float(os.environ.get("PYWORKER_MAX_WAIT_TIME", "10.0"))
    )
    ready_timeout_initial: int = dataclasses.field(
        default_factory=lambda: int(os.environ.get("PYWORKER_READY_TIMEOUT_INITIAL", "1200"))
    )
    ready_timeout_resume: int = dataclasses.field(
        default_factory=lambda: int(os.environ.get("PYWORKER_READY_TIMEOUT_RESUME", "300"))
    )
    reqnum = -1
    version = VERSION
    sem: Semaphore = dataclasses.field(default_factory=Semaphore)
    unsecured: bool = dataclasses.field(
        default_factory=lambda: os.environ.get("PYWORKER_UNSECURED", "false").lower() == "true",
    )
    report_addr: str = dataclasses.field(
        default_factory=lambda: os.environ.get("REPORT_ADDR", "https://run.vast.ai")
    )
    mtoken: str = dataclasses.field(
        default_factory=lambda: os.environ.get("MASTER_TOKEN", "")
    )

    def __post_init__(self):
        self.metrics = Metrics()
        self.metrics._set_version(self.version)
        self.metrics._set_mtoken(self.mtoken)
        self._total_pubkey_fetch_errors = 0
        self._pubkey = self._fetch_pubkey()
        self.__start_healthcheck: bool = False
        self.__consecutive_healthcheck_failures: int = 0

    @property
    def pubkey(self) -> Optional[RSA.RsaKey]:
        if self._pubkey is None:
            self._pubkey = self._fetch_pubkey()
        return self._pubkey

    @cached_property
    def session(self):
        """Main session for forwarding requests to backend API"""
        log.debug(f"starting session with {self.backend_url}")
        connector = create_tcp_connector(force_close=True)  # Required for long running jobs
        timeout = ClientTimeout(total=None)
        return ClientSession(self.backend_url, timeout=timeout, connector=connector)

    def create_handler(self, path: Optional[str] = None):
        """
        Create a generic request handler that forwards any request to the backend.

        If path is provided, it will be used as the target endpoint.
        Otherwise, the path from auth_data.endpoint will be used.
        """
        async def handler_fn(request: web.Request) -> web.StreamResponse:
            return await self.__handle_request(request, path)

        return handler_fn

    async def __parse_and_validate_request(
        self, request: web.Request
    ) -> Tuple[Optional[AuthData], Optional[dict], Optional[web.Response]]:
        """Parse and validate incoming request. Returns (auth_data, payload, error_response)"""
        try:
            # GET/DELETE/HEAD requests don't have bodies - use query params for auth_data
            if request.method in ["GET", "DELETE", "HEAD"]:
                # Try to parse auth_data from query parameters (prefixed with serverless_)
                query_params = dict(request.query)

                # Check if auth_data fields are present in query params (with serverless_ prefix)
                auth_param_keys = [
                    "serverless_cost", "serverless_endpoint", "serverless_reqnum",
                    "serverless_request_idx", "serverless_signature", "serverless_url"
                ]
                has_auth_params = any(key in query_params for key in auth_param_keys)

                if has_auth_params:
                    # Parse auth_data from query parameters
                    try:
                        # Query params are strings, but cost must be numeric to match signature
                        cost_str = query_params.get("serverless_cost", "1.0")
                        cost = float(cost_str) if '.' in cost_str else int(cost_str)

                        auth_data = AuthData(
                            cost=cost,  # Convert to number to match autoscaler's signature
                            endpoint=query_params.get("serverless_endpoint", request.path),
                            reqnum=int(query_params.get("serverless_reqnum", 0)),
                            request_idx=int(query_params.get("serverless_request_idx", 0)),
                            signature=query_params.get("serverless_signature", ""),
                            url=query_params.get("serverless_url", "")
                        )

                        # Validate signature if not unsecured
                        if not self.unsecured and not self.__check_signature(auth_data):
                            return None, None, web.json_response(
                                dict(error="invalid signature"),
                                status=401
                            )

                        # Remaining query params (excluding serverless_ prefixed fields) become payload
                        payload = {
                            k: v for k, v in query_params.items()
                            if k not in auth_param_keys
                        }

                        return auth_data, payload, None
                    except (ValueError, TypeError) as e:
                        return None, None, web.json_response(
                            dict(error=f"Invalid auth_data in query params: {str(e)}"),
                            status=400
                        )

                # No auth_data in query params
                if self.unsecured:
                    # In unsecured mode, create minimal auth_data
                    auth_data = AuthData(
                        cost=1.0,  # Use number for consistency
                        endpoint=request.path,
                        reqnum=0,
                        request_idx=0,
                        signature="",
                        url=""
                    )
                    # All query params become payload
                    return auth_data, query_params, None
                else:
                    return None, None, web.json_response(
                        dict(error=f"{request.method} requests require auth_data in query params (serverless_cost, serverless_endpoint, serverless_reqnum, serverless_request_idx, serverless_signature, serverless_url)"),
                        status=400
                    )

            # POST/PUT/PATCH requests should have JSON body
            data = await request.json()
            auth_data, payload = self.__parse_request(data, request.path)
            return auth_data, payload, None
        except JsonDataException as e:
            return None, None, web.json_response(data=e.message, status=422)
        except json.JSONDecodeError:
            return None, None, web.json_response(dict(error="invalid JSON"), status=422)

    async def __wait_for_client_disconnect(self, request: web.Request, request_metrics: RequestMetrics) -> NoReturn:
        """Wait for client disconnect and mark request as canceled"""
        await request.wait_for_disconnection()
        log.debug(f"request with reqnum: {request_metrics.reqnum} was canceled")
        self.metrics._request_canceled(request_metrics)
        raise asyncio.CancelledError

    async def __forward_request_to_backend(
        self,
        request: web.Request,
        auth_data: AuthData,
        payload: dict,
        request_metrics: RequestMetrics,
        target_path: Optional[str] = None,
    ) -> web.StreamResponse:
        """Forward request to backend and return response"""
        try:
            # Determine endpoint to use
            # Priority: explicit target_path > HTTP request path > auth_data.endpoint (for backward compat)
            endpoint = target_path if target_path else (request.path if request.path else auth_data.endpoint)

            # Forward request to backend
            response = await self.__call_api(
                endpoint=endpoint,
                method=request.method,
                payload=payload
            )

            status_code = response.status
            log.debug(
                f"request with reqnum:{request_metrics.reqnum} "
                f"returned status code: {status_code}"
            )

            # Pass through response
            res = await self.__pass_through_response(request, response)
            self.metrics._request_success(request_metrics)
            return res
        except Exception as e:
            log.debug(f"[backend] Request error: {e}")
            self.metrics._request_errored(request_metrics)
            return web.Response(status=500)

    async def __handle_request(
        self,
        request: web.Request,
        target_path: Optional[str] = None,
    ) -> web.StreamResponse:
        """Forward requests to the model endpoint as-is"""
        # Parse and validate request
        auth_data, payload, error_response = await self.__parse_and_validate_request(request)
        if error_response:
            return error_response

        # At this point, auth_data and payload must be non-None (or we would have returned error)
        # Note: payload can be an empty dict {} for GET/DELETE/HEAD requests with no query params
        assert auth_data is not None, "auth_data should not be None after error check"
        assert payload is not None, "payload should not be None after error check"

        # Create request metrics
        workload = float(auth_data.cost)
        request_metrics = RequestMetrics(
            request_idx=auth_data.request_idx,
            reqnum=auth_data.reqnum,
            workload=workload,
            status="Created"
        )

        # Validate signature and queue
        if self.__check_signature(auth_data) is False:
            self.metrics._request_reject(request_metrics)
            return web.Response(status=401)

        if self.metrics.model_metrics.wait_time > self.max_wait_time:
            self.metrics._request_reject(request_metrics)
            return web.Response(status=429)

        # Process request
        acquired = False
        try:
            self.metrics._request_start(request_metrics)

            # Acquire semaphore if parallel requests not allowed
            if self.allow_parallel_requests is False:
                log.debug(f"Waiting to acquire Sem for reqnum:{request_metrics.reqnum}")
                await self.sem.acquire()
                acquired = True
                log.debug(f"Sem acquired for reqnum:{request_metrics.reqnum}, starting request...")
            else:
                log.debug(f"Starting request for reqnum:{request_metrics.reqnum}")

            # Race between making request and client disconnect
            done, pending = await wait(
                [
                    create_task(self.__forward_request_to_backend(
                        request, auth_data, payload, request_metrics, target_path
                    )),
                    create_task(self.__wait_for_client_disconnect(request, request_metrics)),
                ],
                return_when=FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            # Return result from completed task
            done_task = done.pop()
            try:
                return done_task.result()
            except Exception as e:
                log.debug(f"Request task raised exception: {e}")
                return web.Response(status=500)
        except asyncio.CancelledError:
            # Client is gone. Do not write a response; just unwind.
            return web.Response(status=499)
        except Exception as e:
            log.debug(f"Exception in main handler loop {e}")
            return web.Response(status=500)
        finally:
            # Always release the semaphore if it was acquired
            if acquired:
                self.sem.release()
            self.metrics._request_end(request_metrics)

    def __parse_request(self, data: dict, request_path: str = "/") -> Tuple[AuthData, dict]:
        """Parse request JSON into auth_data and payload

        In production mode (unsecured=false):
            Requires both "auth_data" and "payload" fields

        In local dev mode (unsecured=true):
            Supports passthrough - if no "auth_data" field, treats entire request as payload
        """
        # Passthrough mode: if unsecured and no auth_data, treat entire request as payload
        if self.unsecured and "auth_data" not in data:
            log.debug("Passthrough mode: treating entire request as payload")
            # Create minimal auth_data for metrics tracking
            auth_data = AuthData(
                cost=1.0,  # Default workload (numeric for signature consistency)
                endpoint=request_path,
                reqnum=0,
                request_idx=0,
                signature="",
                url=""
            )
            return (auth_data, data)

        # Standard mode: require both auth_data and payload
        errors = {}
        parsed_auth_data: Optional[AuthData] = None
        parsed_payload: Optional[dict] = None

        try:
            if "auth_data" in data:
                parsed_auth_data = AuthData.from_json_msg(data["auth_data"])
            else:
                errors["auth_data"] = "field missing"
        except JsonDataException as e:
            errors["auth_data"] = e.message

        try:
            if "payload" in data:
                parsed_payload = data["payload"]
            else:
                errors["payload"] = "field missing"
        except Exception as e:
            errors["payload"] = str(e)

        if errors:
            raise JsonDataException(errors)

        # At this point, both must be non-None (or we would have raised)
        assert parsed_auth_data is not None, "auth_data should not be None after validation"
        assert parsed_payload is not None, "payload should not be None after validation"

        return (parsed_auth_data, parsed_payload)

    async def __call_api(
        self, endpoint: str, method: str, payload: dict
    ) -> ClientResponse:
        """Call the backend API with the given method and payload"""
        url = endpoint
        log.debug(f"{method} to endpoint: '{url}', payload: {payload}")

        # Map HTTP methods to session methods
        method_handlers = {
            "GET": lambda: self.session.get(url=url, params=payload if payload else None),
            "POST": lambda: self.session.post(url=url, json=payload),
            "PUT": lambda: self.session.put(url=url, json=payload),
            "PATCH": lambda: self.session.patch(url=url, json=payload),
            "DELETE": lambda: self.session.delete(url=url, json=payload),
        }

        # Get handler or default to POST
        handler = method_handlers.get(method, method_handlers["POST"])
        return await handler()

    async def __pass_through_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> web.StreamResponse:
        """Pass through the model response to client without transformation"""

        if model_response.status != 200:
            # Pass through error responses directly
            content = await model_response.read()
            return web.Response(
                body=content,
                status=model_response.status,
                content_type=model_response.content_type
            )

        # Check if response is streaming
        is_streaming = (
            model_response.content_type == "text/event-stream"
            or model_response.content_type == "application/x-ndjson"
            or model_response.headers.get("Transfer-Encoding") == "chunked"
            or "stream" in model_response.content_type.lower()
        )

        if is_streaming:
            log.debug("Streaming response detected, proxying chunks...")
            response = web.StreamResponse()
            response.content_type = model_response.content_type

            # Copy relevant headers
            for header in ["Transfer-Encoding", "Cache-Control"]:
                if header in model_response.headers:
                    response.headers[header] = model_response.headers[header]

            await response.prepare(client_request)

            async for chunk in model_response.content.iter_any():
                await response.write(chunk)

            await response.write_eof()
            log.debug("Streaming complete")
            return response
        else:
            log.debug("Non-streaming response, proxying body...")
            content = await model_response.read()
            return web.Response(
                body=content,
                status=200,
                content_type=model_response.content_type
            )

    @cached_property
    def healthcheck_session(self):
        """Dedicated session for healthchecks to avoid conflicts with API session"""
        log.debug("creating dedicated healthcheck session")
        connector = create_tcp_connector(force_close=True)
        timeout = ClientTimeout(total=HEALTHCHECK_TIMEOUT)
        return ClientSession(timeout=timeout, connector=connector)

    async def __wait_for_backend_ready(self, is_resume: bool = False) -> None:
        """Poll healthcheck endpoint until backend is ready or timeout

        Args:
            is_resume: If True, use shorter resume timeout (models already on disk).
                      If False, use longer initial timeout (models need to download).
        """
        url = f"{self.backend_url}{self.healthcheck_endpoint}"

        # Choose timeout based on whether this is initial boot or resume
        timeout = self.ready_timeout_resume if is_resume else self.ready_timeout_initial
        timeout_type = "resume" if is_resume else "initial"

        log.info(f"Waiting for backend to be ready at {url} ({timeout_type} timeout: {timeout}s)")

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time

            if elapsed >= timeout:
                error_msg = f"Backend failed to become ready after {timeout} seconds ({timeout_type} timeout)"
                log.error(error_msg)
                self.backend_errored(error_msg)
                raise RuntimeError(error_msg)

            try:
                async with self.healthcheck_session.get(url) as response:
                    if response.status == 200:
                        log.info(f"Backend is ready! (took {elapsed:.1f}s)")
                        return
                    else:
                        log.debug(f"Backend not ready yet (status {response.status}), retrying...")
            except Exception as e:
                log.debug(f"Backend not reachable yet: {e}, retrying...")

            await sleep(HEALTHCHECK_RETRY_INTERVAL)

    async def __healthcheck(self):
        """Periodic healthcheck of the backend with consecutive failure tracking"""
        while True:
            await sleep(HEALTHCHECK_POLL_INTERVAL)
            if self.__start_healthcheck is False:
                continue

            healthcheck_failed = False
            failure_reason = ""

            try:
                log.debug(f"Performing healthcheck on {self.healthcheck_endpoint}")
                url = f"{self.backend_url}{self.healthcheck_endpoint}"
                async with self.healthcheck_session.get(url) as response:
                    if response.status == 200:
                        # Success - reset failure counter
                        if self.__consecutive_healthcheck_failures > 0:
                            log.info(f"Healthcheck recovered after {self.__consecutive_healthcheck_failures} failures")
                        self.__consecutive_healthcheck_failures = 0
                        log.debug("Healthcheck successful")
                    elif response.status == 503:
                        healthcheck_failed = True
                        failure_reason = f"Healthcheck returned 503 Service Unavailable"
                    else:
                        healthcheck_failed = True
                        failure_reason = f"Healthcheck returned status {response.status}"
            except Exception as e:
                healthcheck_failed = True
                failure_reason = f"Healthcheck exception: {type(e).__name__}: {str(e)}"

            # Handle failure
            if healthcheck_failed:
                self.__consecutive_healthcheck_failures += 1
                log.warning(
                    f"Healthcheck failed ({self.__consecutive_healthcheck_failures}/{HEALTHCHECK_CONSECUTIVE_FAILURES}): {failure_reason}"
                )

                # Only mark as errored after consecutive failures threshold
                if self.__consecutive_healthcheck_failures >= HEALTHCHECK_CONSECUTIVE_FAILURES:
                    error_msg = f"Backend failed {HEALTHCHECK_CONSECUTIVE_FAILURES} consecutive healthchecks: {failure_reason}"
                    log.error(error_msg)
                    self.backend_errored(error_msg)
                    # Reset counter so we don't spam error messages
                    self.__consecutive_healthcheck_failures = 0

    async def _start_tracking(self) -> None:
        """Run benchmark, then start background tasks for metrics and healthcheck"""
        # Run benchmark first to completion
        await self.__run_benchmark_on_startup()

        # Then start infinite background loops
        await gather(
            self.metrics._send_metrics_loop(),
            self.__healthcheck(),
            self.metrics._send_delete_requests_loop()
        )

    def backend_errored(self, msg: str) -> None:
        """Mark backend as errored"""
        self.metrics._model_errored(msg)

    async def __run_benchmark_on_startup(self) -> None:
        """Run benchmark on startup to determine max throughput"""

        # Check if this is initial boot or resume (based on benchmark cache)
        benchmark_cached = False
        max_throughput = None

        try:
            with open(BENCHMARK_INDICATOR_FILE, "r") as f:
                max_throughput = float(f.readline())
                benchmark_cached = True
                log.info(f"Benchmark cache found - this is a resume from stopped state")
        except FileNotFoundError:
            log.info(f"No benchmark cache - this is initial startup")

        # Always wait for backend to be ready, regardless of benchmark cache
        # Use different timeout depending on whether this is initial boot or resume
        # Initial: Models need to download (20 min default)
        # Resume: Models already on disk (5 min default)
        await self.__wait_for_backend_ready(is_resume=benchmark_cached)

        # Use cached benchmark if available, otherwise run it
        if benchmark_cached:
            log.info(f"Using cached benchmark result: {max_throughput} workload/s")
        else:
            # No cache, need to run benchmark
            if self.benchmark_func is None:
                error_msg = (
                    "No benchmark function provided. "
                    "Set PYWORKER_BENCHMARK environment variable to specify benchmark function. "
                    "Example: PYWORKER_BENCHMARK=benchmarks.openai_chat:benchmark"
                )
                log.error(error_msg)
                self.backend_errored(error_msg)
                raise RuntimeError(error_msg)

            try:
                log.debug("Running benchmark...")
                max_throughput = await self.benchmark_func(
                    self.backend_url,
                    self.session
                )
                log.debug(f"Benchmark completed: {max_throughput} workload/s")
            except Exception as e:
                error_msg = f"Benchmark failed: {e}"
                log.error(error_msg)
                self.backend_errored(error_msg)
                raise RuntimeError(error_msg) from e

            # Save benchmark result to cache
            with open(BENCHMARK_INDICATOR_FILE, "w") as f:
                f.write(str(max_throughput))

        # Ensure max_throughput is set (should never be None at this point)
        assert max_throughput is not None, "max_throughput should not be None after benchmark"

        # Mark as loaded and enable periodic healthchecks
        self.metrics._model_loaded(max_throughput=max_throughput)
        self.__start_healthcheck = True
        log.info(f"Worker ready (benchmark {'cached' if benchmark_cached else 'completed'}), starting periodic healthchecks")

    def __verify_signature(self, message: str, signature: str) -> bool:
        """Verify PKCS#1 signature"""
        if self.pubkey is None:
            log.debug("Signature verification skipped: no public key available")
            return False

        try:
            key = self.pubkey
            h = SHA256.new(message.encode())
            pkcs1_15.new(key).verify(h, base64.b64decode(signature))
            return True
        except Exception as e:
            log.debug(f"Signature verification failed: {e}")
            return False

    def __check_signature(self, auth_data: AuthData) -> bool:
        """Verify request signature from autoscaler"""
        if self.unsecured is True:
            return True

        # Include all auth_data fields except signature (matching autoscaler behavior)
        auth_data_dict = {
            "cost": auth_data.cost,
            "endpoint": auth_data.endpoint,
            "reqnum": auth_data.reqnum,
            "request_idx": auth_data.request_idx,
            "url": auth_data.url,
        }

        # IMPORTANT: Must use indent=4 to match autoscaler's JSON formatting
        message = json.dumps(auth_data_dict, indent=4, sort_keys=True)
        log.debug(f"Verifying signature for message: {message}")
        log.debug(f"Signature: {auth_data.signature[:20]}..." if len(auth_data.signature) > 20 else f"Signature: {auth_data.signature}")
        result = self.__verify_signature(message, auth_data.signature)
        if not result:
            log.debug(f"Signature verification failed for auth_data: {auth_data_dict}")
        return result

    def _fetch_pubkey(self) -> Optional[RSA.RsaKey]:
        """
        Fetch public key from autoscaler synchronously.

        Note: This is called during __post_init__ (sync context) so we can't use async.
        Consider refactoring to fetch async during startup instead.
        """
        if self.unsecured:
            log.debug("Running in unsecured mode, skipping pubkey fetch")
            return None

        try:
            import requests
            response = requests.get(
                f"{self.report_addr}/pubkey/",
                timeout=PUBKEY_FETCH_TIMEOUT
            )
            response.raise_for_status()
            pubkey_str = response.text
            log.debug(f"Fetched pubkey: {pubkey_str[:50]}...")
            return RSA.import_key(pubkey_str)
        except Exception as e:
            self._total_pubkey_fetch_errors += 1
            log.error(f"Failed to fetch pubkey (attempt {self._total_pubkey_fetch_errors}/{MAX_PUBKEY_FETCH_ATTEMPTS}): {e}")

            if self._total_pubkey_fetch_errors >= MAX_PUBKEY_FETCH_ATTEMPTS:
                error_msg = (
                    f"Failed to fetch public key after {MAX_PUBKEY_FETCH_ATTEMPTS} attempts. "
                    f"Cannot verify request signatures. "
                    f"Set PYWORKER_UNSECURED=true for local development only."
                )
                log.error(error_msg)
                raise RuntimeError(error_msg)

            return None
