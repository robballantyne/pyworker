import logging
import sys
import json
import subprocess
from urllib.parse import urljoin
from typing import Dict, Any, Optional, Iterator, Union, List
import requests
from utils.endpoint_util import Endpoint
from .data_types.client import CompletionConfig, ChatCompletionConfig

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)


class APIClient:
    """Lightweight client focused solely on API communication"""
    
    WORKER_ENDPOINT = "/proxy"
    DEFAULT_COST = 100
    DEFAULT_TIMEOUT = 4
    
    def __init__(self, endpoint_group_name: str, api_key: str, server_url: str):
        self.endpoint_group_name = endpoint_group_name
        self.api_key = api_key
        self.server_url = server_url
        self.endpoint_api_key = self._get_endpoint_api_key()
    
    def _get_endpoint_api_key(self) -> Optional[str]:
        """Get the endpoint API key"""
        endpoint_api_key = Endpoint.get_endpoint_api_key(
            endpoint_name=self.endpoint_group_name,
            account_api_key=self.api_key,
        )
        if not endpoint_api_key:
            log.error(f"Failed to get API key for endpoint {self.endpoint_group_name}")
        return endpoint_api_key
    
    def _get_worker_url(self, cost: int = DEFAULT_COST) -> Dict[str, Any]:
        """Get worker URL and auth data from routing service"""
        if not self.endpoint_api_key:
            raise ValueError("No valid endpoint API key available")
            
        route_payload = {
            "endpoint": self.endpoint_group_name,
            "api_key": self.endpoint_api_key,
            "cost": cost,
        }
        
        response = requests.post(
            urljoin(self.server_url, "/route/"),
            json=route_payload,
            timeout=self.DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    
    def _create_auth_data(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Create auth data from routing response"""
        return {
            "signature": message["signature"],
            "cost": message["cost"],
            "endpoint": message["endpoint"],
            "reqnum": message["reqnum"],
            "url": message["url"],
        }
    
    def _make_request(self, payload: Dict[str, Any], endpoint: str, method: str = "POST", 
                     stream: bool = False) -> Union[Dict[str, Any], Iterator[str]]:
        """Make request to worker endpoint"""
        # Get worker URL and auth data
        message = self._get_worker_url()
        worker_url = message["url"]
        auth_data = self._create_auth_data(message)
        
        # Prepare request data
        request_payload = {
            "input": payload,
            "endpoint": endpoint,
            "method": method
        }
        
        req_data = {
            "payload": request_payload,
            "auth_data": auth_data
        }
        
        url = urljoin(worker_url, self.WORKER_ENDPOINT)
        log.debug(f"Making request to: {url}")
        
        # Make the request
        response = requests.post(url, json=req_data, stream=stream)
        response.raise_for_status()
        
        if stream:
            return self._handle_streaming_response(response)
        else:
            return response.json()
    
    def _handle_streaming_response(self, response: requests.Response) -> Iterator[str]:
        """Handle streaming response and yield tokens"""
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    # Handle Server-Sent Events format
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            # Extract content from streaming response
                            if "choices" in data and data["choices"]:
                                choice = data["choices"][0]
                                if "delta" in choice and "content" in choice["delta"]:
                                    yield choice["delta"]["content"]
                                elif "text" in choice:
                                    yield choice["text"]
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            log.error(f"Error handling streaming response: {e}")
            raise
    
    def call_completions(self, config: CompletionConfig) -> Union[Dict[str, Any], Iterator[str]]:
        """Call completions endpoint"""
        payload = {
            "model": config.model,
            "prompt": config.prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": config.stream
        }
        
        return self._make_request(
            payload=payload,
            endpoint="/v1/completions",
            stream=config.stream
        )
    
    def call_chat_completions(self, config: ChatCompletionConfig) -> Union[Dict[str, Any], Iterator[str]]:
        """Call chat completions endpoint"""
        payload = {
            "model": config.model,
            "messages": config.messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": config.stream
        }
        
        # Add tools if provided
        if config.tools:
            payload["tools"] = config.tools
            payload["tool_choice"] = config.tool_choice
        
        return self._make_request(
            payload=payload,
            endpoint="/v1/chat/completions",
            stream=config.stream
        )


class ToolManager:
    """Handles tool definitions and execution"""
    
    @staticmethod
    def list_files() -> str:
        """Execute ls on current directory"""
        try:
            result = subprocess.run(['ls', '-la', '.'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error running ls: {e}"
    
    @staticmethod
    def get_ls_tool_definition() -> List[Dict[str, Any]]:
        """Get the ls tool definition"""
        return [{
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files and directories in the cwd",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }]
    
    def execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call and return the result"""
        function_name = tool_call["function"]["name"]
        
        if function_name == "list_files":
            return self.list_files()
        else:
            raise ValueError(f"Unknown tool function: {function_name}")


class APIDemo:
    """Demo and testing functionality for the API client"""
    
    def __init__(self, client: APIClient, tool_manager: ToolManager = None):
        self.client = client
        self.tool_manager = tool_manager or ToolManager()
    
    def test_tool_support(self) -> bool:
        """Test if the endpoint supports function calling"""
        print("Testing endpoint tool calling support...")
        
        # Try a simple request with minimal tools to test support
        messages = [{"role": "user", "content": "Hello"}]
        minimal_tool = [{
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "Test function"
            }
        }]
        
        config = ChatCompletionConfig(
            messages=messages,
            max_tokens=10,
            tools=minimal_tool,
            tool_choice="none"  # Don't actually call the tool
        )
        
        try:
            response = self.client.call_chat_completions(config)
            return True
        except Exception as e:
            print(f"Error: Endpoint does not support tool calling: {e}")
            return False
    
    def demo_completions(self) -> None:
        """Demo: test basic completions endpoint"""
        print("=" * 60)
        print("COMPLETIONS DEMO")
        print("=" * 60)
        
        config = CompletionConfig(
            prompt="July 4th is",
            max_tokens=100,
            stream=False
        )
        
        print(f"Testing completions with prompt: '{config.prompt}'")
        response = self.client.call_completions(config)
        
        if isinstance(response, dict):
            print("\nResponse:")
            print(json.dumps(response, indent=2))
        else:
            print("Unexpected response format")
    
    def demo_chat(self, use_streaming: bool = True) -> None:
        """Demo: test chat completions endpoint with optional streaming"""
        print("=" * 60)
        print(f"CHAT COMPLETIONS DEMO {'(STREAMING)' if use_streaming else '(NON-STREAMING)'}")
        print("=" * 60)
        
        config = ChatCompletionConfig(
            messages=[{"role": "user", "content": "Tell me about the Python programming language."}],
            max_tokens=500,
            stream=use_streaming,
        )
        
        print("Testing chat completions...")
        response = self.client.call_chat_completions(config)
        
        if use_streaming:
            print("\nAssistant (streaming): ", end="", flush=True)
            full_response = ""
            try:
                for token in response:
                    print(token, end="", flush=True)
                    full_response += token
                print()  # New line after streaming
            except Exception as e:
                print(f"\nError during streaming: {e}")
                return
            
            print(f"\nStreaming completed. Total tokens received: {len(full_response.split())}")
        else:
            if isinstance(response, dict):
                choice = response.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                
                print(f"\nAssistant: {content}")
                print(f"\nFull Response:")
                print(json.dumps(response, indent=2))
            else:
                print("Unexpected response format")
    
    def demo_ls_tool(self) -> None:
        """Demo: ask LLM to list files in the current directory and describe what it sees"""
        print("=" * 60)
        print("TOOL USE DEMO: List Directory Contents")
        print("=" * 60)
        
        # Test if tools are supported first
        if not self.test_tool_support():
            return
        
        # Request with tool available
        messages = [
            {"role": "user", "content": "Can you list the files in the current working directory and tell me what you see? What do you think this directory might be for?"}
        ]
        
        config = ChatCompletionConfig(
            messages=messages,
            max_tokens=300,
            tools=self.tool_manager.get_ls_tool_definition(),
            tool_choice="auto"
        )
        
        print("Making initial request with tool...")
        response = self.client.call_chat_completions(config)
        
        if not isinstance(response, dict):
            raise ValueError("Expected dict response for tool use")
        
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        print(f"Assistant response: {message.get('content', 'No content')}")
        
        # Check for tool calls
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            raise ValueError("No tool calls made - model may not support function calling")
        
        print(f"Tool calls detected: {len(tool_calls)}")
        
        # Execute the tool call
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            print(f"Executing tool: {function_name}")
            
            tool_result = self.tool_manager.execute_tool_call(tool_call)
            print(f"Tool result:\n{tool_result}")
            
            # Add tool result and continue conversation
            messages.append(message)  # Add assistant's message with tool call
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_result
            })
            
            # Get final response
            final_config = ChatCompletionConfig(
                messages=messages,
                max_tokens=400,
                tools=self.tool_manager.get_ls_tool_definition()
            )
            
            print("Getting final response...")
            final_response = self.client.call_chat_completions(final_config)
            
            if isinstance(final_response, dict):
                final_choice = final_response.get("choices", [{}])[0]
                final_message = final_choice.get("message", {})
                final_content = final_message.get("content", "")
                
                print("\n" + "=" * 60)
                print("FINAL LLM ANALYSIS:")
                print("=" * 60)
                print(final_content)
                print("=" * 60)
    
    def interactive_chat(self) -> None:
        """Interactive chat session with streaming"""
        print("=" * 60)
        print("INTERACTIVE STREAMING CHAT")
        print("=" * 60)
        print("Type 'quit' to exit, 'clear' to clear history")
        print()
        
        messages = []
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    messages = []
                    print("Chat history cleared")
                    continue
                elif not user_input:
                    continue
                
                messages.append({"role": "user", "content": user_input})
                
                config = ChatCompletionConfig(
                    messages=messages,
                    max_tokens=500,
                    stream=True,
                    temperature=0.7
                )
                
                print("Assistant: ", end="", flush=True)
                
                response = self.client.call_chat_completions(config)
                assistant_content = ""
                
                for token in response:
                    print(token, end="", flush=True)
                    assistant_content += token
                
                print("\n")
                
                # Add assistant response to conversation history
                messages.append({"role": "assistant", "content": assistant_content})
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue


def main():
    """Main function with CLI switches for different tests"""
    from lib.test_utils import test_args
    
    # Add test mode arguments
    test_args.add_argument(
        "--completion", 
        action="store_true",
        help="Test completions endpoint"
    )
    test_args.add_argument(
        "--chat", 
        action="store_true",
        help="Test chat completions endpoint (non-streaming)"
    )
    test_args.add_argument(
        "--chat-stream", 
        action="store_true",
        help="Test chat completions endpoint with streaming"
    )
    test_args.add_argument(
        "--tools", 
        action="store_true",
        help="Test function calling with ls tool (non-streaming)"
    )
    test_args.add_argument(
        "--interactive", 
        action="store_true",
        help="Start interactive streaming chat session"
    )
    
    args = test_args.parse_args()
    
    # Check that only one test mode is selected
    test_modes = [
        args.completion, args.chat, args.chat_stream, 
        args.tools, args.interactive
    ]
    selected_count = sum(test_modes)
    
    if selected_count == 0:
        print("Please specify exactly one test mode:")
        print("  --completion    : Test completions endpoint")
        print("  --chat          : Test chat completions endpoint (non-streaming)")
        print("  --chat-stream   : Test chat completions endpoint with streaming")
        print("  --tools         : Test function calling with ls tool (non-streaming)")
        print("  --interactive   : Start interactive streaming chat session")
        print(f"\nExample: python {sys.argv[0]} --chat-stream -k YOUR_KEY -e YOUR_ENDPOINT")
        sys.exit(1)
    elif selected_count > 1:
        print("Please specify exactly one test mode")
        sys.exit(1)
    
    try:
        # Create the core API client
        client = APIClient(
            endpoint_group_name=args.endpoint_group_name,
            api_key=args.api_key,
            server_url=args.server_url
        )
        
        # Create tool manager and demo
        tool_manager = ToolManager()
        demo = APIDemo(client, tool_manager)
        
        # Run the selected test
        if args.completion:
            demo.demo_completions()
        elif args.chat:
            demo.demo_chat(use_streaming=False)
        elif args.chat_stream:
            demo.demo_chat(use_streaming=True)
        elif args.tools:
            demo.demo_ls_tool()
        elif args.interactive:
            demo.interactive_chat()
        
    except Exception as e:
        log.error(f"Error during test: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()