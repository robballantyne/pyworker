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
class CompletionsData(ApiPayload):
    input: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompletionsData":
        return cls(
            input=data["input"]
        )

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

    def generate_payload_json(self) -> Dict[str, Any]:
        return self.input

    def count_workload(self) -> int:
        return self.input.get("max_tokens", 0)

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]) -> "CompletionsData":
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        try:
            input_data = json_msg["input"]
            if isinstance(input_data, str):
                input_data = json.loads(input_data)
            return cls(input=input_data)
        
        except (json.JSONDecodeError, JsonDataException) as e:
            errors["parameters"] = str(e)
            raise JsonDataException(errors)
