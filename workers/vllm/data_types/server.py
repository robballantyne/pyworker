import os, json, random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from lib.data_types import EndpointHandler, ApiPayload, JsonDataException
from typing import Union, Type, Dict, Any
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
class GenericData(ApiPayload, ABC):
    input: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenericData":
        return cls(
            input=data["input"]
        )
    
    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "GenericData":
        errors = {}
        
        # Validate required parameters
        required_params = ["input"]
        for param in required_params:
            if param not in json_msg:
                errors[param] = "missing parameter"
        
        if errors:
            raise JsonDataException(errors)
        
        try:
            # Create clean data dict and delegate to from_dict
            clean_data = {
                "input": json_msg["input"]
            }
            
            return cls.from_dict(clean_data)
            
        except (json.JSONDecodeError, JsonDataException) as e:
            errors["parameters"] = str(e)
            raise JsonDataException(errors)

    @classmethod
    @abstractmethod
    def for_test(cls) -> "GenericData":
        pass

    def generate_payload_json(self) -> Dict[str, Any]:
        return self.input

    def count_workload(self) -> int:
        return self.input.get("max_tokens", 0)
    
@dataclass
class GenericHandler(EndpointHandler[GenericData], ABC):

    @property
    @abstractmethod
    def endpoint(self) -> str:
        pass
    
    @property
    def healthcheck_endpoint(self) -> str:
        return f"{os.environ.get('MODEL_SERVER_URL', '127.0.0.1:8000')}/health"

    @classmethod
    def payload_cls(cls) -> Type[GenericData]:
        return GenericData

    @abstractmethod
    def make_benchmark_payload(self) -> GenericData:
        pass

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
            ) -> Union[web.Response, web.StreamResponse]:
        match model_response.status:
            case 200:
                # Check if the response is actually streaming based on response headers/content-type
                is_streaming_response = (
                    model_response.content_type == "text/event-stream" or
                    model_response.content_type == "application/x-ndjson" or
                    model_response.headers.get("Transfer-Encoding") == "chunked" or
                    "stream" in model_response.content_type.lower()
                )
                
                if is_streaming_response:
                    log.debug("Detected streaming response...")
                    res = web.StreamResponse()
                    res.content_type = model_response.content_type
                    await res.prepare(client_request)
                    async for chunk in model_response.content:
                        await res.write(chunk)
                    await res.write_eof()
                    log.debug("Done streaming response")
                    return res
                else:
                    log.debug("Detected non-streaming response...")
                    content = await model_response.read()
                    return web.Response(
                        body=content,
                        status=200,
                        content_type=model_response.content_type
                    )
            case code:
                log.debug("SENDING RESPONSE: ERROR: unknown code")
                return web.Response(status=code)

@dataclass
class CompletionsData(GenericData):
    @classmethod
    def for_test(cls) -> "CompletionsData":
        prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
        model = os.getenv("VLLM_MODEL")
        if not model:
            raise ValueError("VLLM_MODEL environment variable not set")
        
        test_input = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7
        }
        return cls(input=test_input)
    
@dataclass
class CompletionsHandler(GenericHandler):
    @property
    def endpoint(self) -> str:
        return "/v1/completions"
    
    @classmethod
    def payload_cls(cls) -> Type[CompletionsData]:
        return CompletionsData
    
    def make_benchmark_payload(self) -> CompletionsData:
        return CompletionsData.for_test()

@dataclass
class ChatCompletionsData(GenericData):
    """Chat completions-specific data implementation"""
    
    @classmethod
    def for_test(cls) -> "ChatCompletionsData":
        prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
        model = os.getenv("VLLM_MODEL")
        if not model:
            raise ValueError("VLLM_MODEL environment variable not set")
        
        # Chat completions use messages format instead of prompt
        test_input = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7
        }
        return cls(input=test_input)
    
@dataclass  
class ChatCompletionsHandler(GenericHandler): 
    @property
    def endpoint(self) -> str:
        return "/v1/chat/completions"
    
    @classmethod
    def payload_cls(cls) -> Type[ChatCompletionsData]:
        return ChatCompletionsData
        
    def make_benchmark_payload(self) -> ChatCompletionsData:
        return ChatCompletionsData.for_test()
