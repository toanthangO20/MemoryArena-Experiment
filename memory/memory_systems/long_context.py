import time
from typing import Optional

import tiktoken


class LongContextMemorySystem:
    """
    Simple long-context memory: accumulates text chunks into a running context.
    Keeps the most recent content up to a token budget using the gpt-4o-mini tokenizer.
    """

    def __init__(self, max_tokens: int = 120_000, user_id: Optional[str] = None):
        self.max_tokens = max_tokens
        self.user_id = user_id
        self.context = ""
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

    def add_chunk(self, chunk: str):
        stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {chunk}"
        if self.context:
            self.context = f"{self.context}\n{stamped}"
        else:
            self.context = stamped
        self._truncate()

    def _truncate(self):
        tokens = self.tokenizer.encode(self.context, disallowed_special=())
        if len(tokens) > self.max_tokens:
            self.context = self.tokenizer.decode(tokens[-self.max_tokens :])

    def wrap_user_prompt(self, prompt: str) -> str:
        memory_context_lines = ["<memory_context>"]
        if self.context:
            memory_context_lines.append(self.context)
        else:
            memory_context_lines.append("None")
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")
        return "\n".join(memory_context_lines)
