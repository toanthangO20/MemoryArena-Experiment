import os
import uuid
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from mem0 import MemoryClient

class Mem0MemorySystem:
    
    def __init__(self, user_id: Optional[str] = None, enable_graph: bool = False):
        self.client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY"),
        )
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())
        self.enable_graph = enable_graph

    def add_chunk(self, chunk: str):
        add_kwargs = {"user_id": self.user_id, "version": "v2"}
        if self.enable_graph:
            add_kwargs["enable_graph"] = True
        self.client.add(
            [
                {'role': 'user', 'content': chunk},
                {'role': 'assistant', 'content': "Thanks for the information! I will remember this."}
            ],
            **add_kwargs
        )
    
    def wrap_user_prompt(self, prompt: str):
        filters = {
            "OR": [
                {"user_id": self.user_id}
            ]
        }
        memories = self.client.search(prompt.lower(), version="v2", filters=filters)
        memory_context_lines = ["<memory_context>"]
        for result in memories['results']:
            memory_text = result.get("memory")
            if memory_text:
                categories = result.get("categories") or []
                if categories:
                    categories_text = ", ".join(categories)
                    memory_context_lines.append(f"{memory_text} (categories: {categories_text})")
                else:
                    memory_context_lines.append(memory_text)
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User Prompt: {prompt}")

        return "\n".join(memory_context_lines)
