import json
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional, List, Dict, Any


class SerializableDataclass:
    def _serialize_recursive(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return {
                field.name: self._serialize_recursive(getattr(obj, field.name))
                for field in fields(obj)
            }
        elif isinstance(obj, dict):
            return {key: self._serialize_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_recursive(item) for item in obj]
        elif isinstance(obj, set):
            return [self._serialize_recursive(item) for item in obj]
        else:
            return obj

    def to_dict(self) -> Dict[str, Any]:
        return self._serialize_recursive(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class CompletionConfig(SerializableDataclass):
    """Configuration for completion requests"""

    model: str
    prompt: str = "Hello"
    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.4
    stream: bool = False


@dataclass
class ChatCompletionConfig(SerializableDataclass):
    """Configuration for chat completion requests"""

    model: str
    messages: list = field(default_factory=list)
    max_tokens: int = 2096
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.4
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    tool_choice: str = "auto"

    def __post_init__(self):
        if self.messages is None:
            self.messages = [{"role": "user", "content": "Hello"}]
