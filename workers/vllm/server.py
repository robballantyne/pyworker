import os
import logging
from typing import Union, Type, ClassVar, Dict, Optional, Any, Tuple
import dataclasses

from aiohttp import web, ClientResponse

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler, AuthData
from lib.server import start_server
from .data_types import GenericData


MODEL_SERVER_URL = "http://127.0.0.1:18000"

# This line indicates that the vLLM inference server is listening
MODEL_SERVER_START_LOG_MSG = [
    'Application startup complete.'
]

MODEL_SERVER_ERROR_LOG_MSGS = [
    "INFO exited: vllm",
    "RuntimeError: Engine"
]

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


@dataclasses.dataclass
class GenericHandler(EndpointHandler[GenericData]):
    # These fields maintain compatibility with base class and current backend expectations
   
    # Class variable to store current payload for each request.
    _current_payload: ClassVar[Optional[GenericData]] = None
    
    @property
    def endpoint(self) -> str:
        """Dynamic endpoint from current request payload"""
        if GenericHandler._current_payload:
            return GenericHandler._current_payload.endpoint
    
    @property
    def healthcheck_endpoint(self) -> str:
        return f"{MODEL_SERVER_URL}/health"

    @classmethod
    def payload_cls(cls) -> Type[GenericData]:
        return GenericData

    @classmethod  
    def get_data_from_request(cls, req_data: Dict[str, Any]) -> Tuple[AuthData, GenericData]:
        """Override to capture payload for dynamic endpoint"""
        auth_data, payload = super().get_data_from_request(req_data)
        
        # Store payload in class variable for endpoint access
        cls._current_payload = payload
        
        return auth_data, payload

    def make_benchmark_payload(self) -> GenericData:
        payload = GenericData.for_test()
        GenericHandler._current_payload = payload  # Set ClassVar for endpoint property
        return payload

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
            ) -> Union[web.Response, web.StreamResponse]:
        match model_response.status:
            case 200:
                # Check if streaming is expected
                if GenericHandler._current_payload and GenericHandler._current_payload.is_stream:
                    log.debug("Streaming response...")
                    res = web.StreamResponse()
                    res.content_type = "text/event-stream"
                    await res.prepare(client_request)
                    async for chunk in model_response.content:
                        await res.write(chunk)
                    await res.write_eof()
                    log.debug("Done streaming response")
                    return res
                else:
                    log.debug("Non-streaming response...")
                    content = await model_response.read()
                    return web.Response(
                        body=content,
                        status=200,
                        content_type=model_response.content_type
                    )
            case code:
                log.debug("SENDING RESPONSE: ERROR: unknown code")
                return web.Response(status=code)

backend = Backend(
    model_server_url=MODEL_SERVER_URL,
    model_log_file=os.environ["MODEL_LOG"],
    allow_parallel_requests=True,
    benchmark_handler=GenericHandler(benchmark_runs=3, benchmark_words=256),
    log_actions=[
        *[(LogAction.ModelLoaded, info_msg) for info_msg in MODEL_SERVER_START_LOG_MSG],
        (LogAction.Info, '"message":"Download'),
        *[
            (LogAction.ModelError, error_msg)
            for error_msg in MODEL_SERVER_ERROR_LOG_MSGS
        ],
    ],
)

async def handle_ping(_):
    return web.Response(body="pong")


routes = [
    web.post("/proxy", backend.create_handler(GenericHandler())),
    web.get("/ping", handle_ping),
]

if __name__ == "__main__":
    start_server(backend, routes)
