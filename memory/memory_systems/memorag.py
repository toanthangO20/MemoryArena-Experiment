from __future__ import annotations

import os
from typing import List, Optional

from openai import OpenAI

from .MemoRAG.memorag import MemoRAG as MemoRAGCore


class MemoRAGMemorySystem:
    def __init__(
        self,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        memory_model: str = "gpt-4.1-mini",
        embedding_model: str = "text-embedding-3-small",
        ret_hit: int = 5,
        retrieval_chunk_size: int = 2048,
        api_client: Optional[OpenAI] = None,
    ):
        api_key = api_key or os.getenv("MEMORAG_API_KEY") or os.getenv("OPENAI_API_KEY")
        api_endpoint = api_endpoint or os.getenv("MEMORAG_API_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if api_client is None:
            if not api_key:
                raise ValueError("API key is required for MemoRAG.")
            if api_endpoint:
                api_client = OpenAI(api_key=api_key, base_url=api_endpoint)
            else:
                api_client = OpenAI(api_key=api_key)

        self.user_id = user_id
        self._chunks: List[str] = []

        self._core = MemoRAGCore(
            mem_model_name_or_path=memory_model,
            ret_model_name_or_path=embedding_model,
            retrieval_chunk_size=retrieval_chunk_size,
            api_client=api_client,
            api_key=api_key,
            api_endpoint=api_endpoint,
        )

    def add_chunk(self, chunk: str):
        if not chunk or not chunk.strip():
            return
        self._chunks.append(chunk)
        context = "\n\n".join(self._chunks)
        self._core.memorize(context)

    def wrap_user_prompt(self, prompt: str):
        if not self._chunks:
            memory_context_lines = ["<memory_context>", "None", "</memory_context>", f"User: {prompt}"]
            return "\n".join(memory_context_lines)

        retrieval_results = self._core.retrieve(prompt)
        memory_context_lines = ["<memory_context>"]
        if retrieval_results:
            for chunk in retrieval_results:
                memory_context_lines.append(f"<memory>{chunk}</memory>")
        else:
            memory_context_lines.append("None")
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")
        return "\n".join(memory_context_lines)
        
