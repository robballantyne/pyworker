from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class CompletionConfig:
    """Configuration for completion requests"""
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    prompt: str = "Hello"
    max_tokens: int = 10
    temperature: float = 0.7
    top_k: int = 20
    top_p: int = 40
    stream: bool = False


@dataclass
class ChatCompletionConfig:
    """Configuration for chat completion requests"""
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    messages: list = None
    max_tokens: int = 250
    temperature: float = 0.7
    top_k: int = 20
    top_p: int = 40
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: str = "auto"
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = [{"role": "user", "content": "Hello"}]