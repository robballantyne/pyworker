import dataclasses
import random
import inspect
import json
import os
from typing import Dict, Any

from transformers import OpenAIGPTTokenizer
import nltk

from lib.data_types import ApiPayload, JsonDataException

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

@dataclasses.dataclass
class GenericData(ApiPayload):
    endpoint: str
    method: str
    input: Dict[str, Any]

    @property
    def is_stream(self) -> bool:
        return self.input.get("stream", False) is True

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
    
