import os, json, random
from dataclasses import dataclass
from lib.data_types import EndpointHandler, AuthData, ApiPayload, JsonDataException
from typing import Union, Type, ClassVar, Dict, Optional, Any, Tuple
from aiohttp import web, ClientResponse
import nltk
import logging

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()
log = logging.getLogger(__name__)
            
"""
Generic dataclass accepts any dictionary in input.
"""
@dataclass
class GenericData(ApiPayload):
    endpoint: str
    method: str
    input: Dict[str, Any]

    @property
    def is_stream(self) -> bool:
        """Check if caller requested streaming response"""
        return str(self.input.get("stream", False)).lower() == "true"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenericData":
        return cls(
            method=data.get("method", "POST"),  # Default to POST
            endpoint=data["endpoint"],          # Dynamic endpoint selection
            input=data["input"]                 # Actual payload
        )
    
    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "GenericData":
        errors = {}
        
        # Validate required parameters (method is optional, defaults to POST)
        required_params = ["endpoint", "input"]
        for param in required_params:
            if param not in json_msg:
                errors[param] = "missing parameter"
        
        if errors:
            raise JsonDataException(errors)
        
        try:
            # Create clean data dict and delegate to from_dict
            clean_data = {
                "endpoint": json_msg["endpoint"],
                "input": json_msg["input"]
            }
            
            # Include method if provided
            if "method" in json_msg:
                clean_data["method"] = json_msg["method"]
            
            return cls.from_dict(clean_data)
            
        except (json.JSONDecodeError, JsonDataException) as e:
            errors["parameters"] = str(e)
            raise JsonDataException(errors)

    @classmethod
    def for_test(cls) -> "GenericData":
        prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
        model = os.getenv("VLLM_MODEL")
        if not model:
            raise ValueError("VLLM_MODEL environment variable not set")
        
        test_endpoint = "/v1/completions"
        test_method = "POST"
        test_input = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7
        }
        return cls(endpoint=test_endpoint, method=test_method, input=test_input)

    def generate_payload_json(self) -> Dict[str, Any]:
        return self.input

    def count_workload(self) -> int:
        return self.input.get("max_tokens", 0)
    
@dataclass
class GenericHandler(EndpointHandler[GenericData]):
    # Class variable to store current payload for each request.
    _current_payload: ClassVar[Optional[GenericData]] = None
    
    @property
    def endpoint(self) -> str:
        """Dynamic endpoint from current request payload"""
        if GenericHandler._current_payload:
            return GenericHandler._current_payload.endpoint
    
    @property
    def healthcheck_endpoint(self) -> str:
        return f"{os.environ.get("MODEL_SERVER", "127.0.0.1:8000")}/health"

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