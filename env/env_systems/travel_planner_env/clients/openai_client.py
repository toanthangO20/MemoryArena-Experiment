"""
OpenAI Client - Implementation of BaseModelClient for OpenAI API
Supports OpenAI's Tools API (function calling)
"""

import os
import json
import time
from typing import List, Dict, Optional
from openai import OpenAI

from .base_client import BaseModelClient, ModelResponse, ToolCall
from ..cost_tracker import CostTracker


class OpenAIClient(BaseModelClient):
    """
    OpenAI client with Tools API support.
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None, base_url: str = None):
        super().__init__(model_name)
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("OPENAI_API_BASE"),
        )
        self.cost_tracker = CostTracker(model_name)
    
    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 32768
    ) -> ModelResponse:
        """
        Send messages to OpenAI with tools enabled.
        """
        try:
            if self.model_name.startswith('gpt-5'):
                # gpt-5 系列: 用 max_completion_tokens, 不传 temperature
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                    max_completion_tokens=max_tokens
                )
            else:
                # gpt-4 / Claude / 其他: 用 temperature + max_tokens
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            # Track usage
            if response.usage:
                self.total_input_tokens += response.usage.prompt_tokens
                self.total_output_tokens += response.usage.completion_tokens
                self.cost_tracker.add_usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens
                )
            
            # Parse response
            message = response.choices[0].message
            
            # Extract tool calls if any
            tool_calls = None
            if message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    )
                    for tc in message.tool_calls
                ]
            
            return ModelResponse(
                content=message.content,
                tool_calls=tool_calls,
                raw_response=response
            )
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
    
    def format_tool_result(self, tool_call_id: str, result: str, name: str = None) -> Dict:
        """
        Format tool result as a message for OpenAI.
        Note: OpenAI doesn't need the name parameter, but we accept it for API consistency.
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        }
    
    def format_assistant_tool_calls(self, tool_calls: List[ToolCall]) -> Dict:
        """
        Format assistant's tool calls as a message.
        This is needed to maintain proper conversation history.
        """
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in tool_calls
            ]
        }
    
    def get_usage_stats(self) -> Dict:
        """Get usage stats using CostTracker"""
        cost_info = self.cost_tracker.get_cost()
        return {
            'total_input_tokens': cost_info['input_tokens'],
            'total_output_tokens': cost_info['output_tokens'],
            'total_cost': cost_info['total_cost']
        }


# For testing
if __name__ == "__main__":
    client = OpenAIClient(model_name="gpt-4o-mini")
    
    messages = [
        {"role": "system", "content": "You are a travel planning assistant."},
        {"role": "user", "content": "Search for flights from New York to Los Angeles on 2024-03-15"}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "FlightSearch",
                "description": "Search for flights",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "date": {"type": "string"}
                    },
                    "required": ["origin", "destination", "date"]
                }
            }
        }
    ]
    
    response = client.chat_with_tools(messages, tools)
    print(f"Content: {response.content}")
    print(f"Tool calls: {response.tool_calls}")
    print(f"Usage: {client.get_usage_stats()}")
