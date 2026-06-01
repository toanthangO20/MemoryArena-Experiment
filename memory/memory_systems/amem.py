import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    from agentic_memory.memory_system import AgenticMemorySystem
except ImportError:
    print("AgenticMemorySystem not found, please install AMEM")


class AMemMemorySystem:

    def __init__(
        self,
        user_id: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        if api_key is None and llm_backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        self.user_id = user_id
        self.memory_system = AgenticMemorySystem(
            model_name=model_name,
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
        )

    def add_chunk(self, chunk: str):
        if not chunk or not chunk.strip():
            return None
        memory_id = self.memory_system.add_note(chunk)
        memory = self.memory_system.read(memory_id)
        if memory is None:
            return None
        return {
            "content": memory.content,
            "keywords": memory.keywords,
            "context": memory.context,
            "tags": memory.tags,
        }

    def wrap_user_prompt(self, prompt: str):
        results = self.memory_system.search(prompt.lower(), k=5)
        memory_context_lines = ["<memory_context>"]

        if not results:
            memory_context_lines.append("None")
        else:
            for result in results:
                memory_text = result.get("content")
                if not memory_text:
                    continue
                tags = result.get("tags") or []
                keywords = result.get("keywords") or []
                context = result.get("context") or ""

                meta_parts = []
                if tags:
                    meta_parts.append(f"tags: {', '.join(tags)}")
                if keywords:
                    meta_parts.append(f"keywords: {', '.join(keywords)}")
                if context:
                    meta_parts.append(f"context: {context}")

                if meta_parts:
                    memory_context_lines.append(f"{memory_text} ({'; '.join(meta_parts)})")
                else:
                    memory_context_lines.append(memory_text)

            if len(memory_context_lines) == 1:
                memory_context_lines.append("None")

        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User Prompt: {prompt}")

        return "\n".join(memory_context_lines)
