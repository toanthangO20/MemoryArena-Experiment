import os
from letta_client import Letta
from typing import Optional


class LettaMemorySystem:
    
    def __init__(self, user_id: Optional[str] = None):

        self.client = Letta(api_key=os.getenv("LETTA_API_KEY"))
        self.agent_state = self.client.agents.create(
                model="openai/gpt-4.1-mini",
                embedding="openai/text-embedding-3-small",
                memory_blocks=[
                    {
                        "label": "human",
                        "value": ""
                    },
                    {
                    "label": "persona",
                    "value": "I am a self-improving superintelligence. "
                    }
                ],
                tools=[]
            )

    def add_chunk(self, chunk: str):
        response = self.client.agents.messages.create(
            agent_id=self.agent_state.id,
            input="Remember this:\n" + chunk
        )

        parsed_messages = []
        if hasattr(response, "messages"):
            for message in response.messages:
                if getattr(message, "message_type", None) == "tool_call_message":
                    tool_calls = getattr(message, "tool_calls", None) or []
                    if not tool_calls and hasattr(message, "tool_call"):
                        tool_calls = [message.tool_call]
                    for tool_call in tool_calls:
                        parsed_messages.append({
                            "type": "tool_call",
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        })
                elif getattr(message, "message_type", None) == "assistant_message":
                    parsed_messages.append({
                        "type": "text",
                        "content": message.content,
                    })

        return parsed_messages if parsed_messages else None
    
    def wrap_user_prompt(self, prompt: str):

        response = self.client.agents.messages.create(
            agent_id=self.agent_state.id,
            input="This is the user's prompt: " + prompt + "\n\nRetrieve the most relevant information from your memory and return it in text format."
        )
        
        memory_context_lines = ["<memory_context>"]

        for message in response.messages:
            try:
                memory_context_lines.append(message.content)
            except Exception as e:
                print("Error in message:", message)
                continue
        
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")
        
        return "\n".join(memory_context_lines)
