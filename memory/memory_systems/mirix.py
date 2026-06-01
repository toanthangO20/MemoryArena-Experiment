from mirix import MirixClient
import uuid
import os
from pathlib import Path
import yaml
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class MirixMemorySystem:
    
    def __init__(self, user_id: Optional[str] = None):
        self.client = MirixClient(api_key=os.getenv("MIRIX_API_KEY"))
        self.user_id = user_id if user_id is not None else str(uuid.uuid4())
        config_path = Path(__file__).with_name("mirix_openai.yaml")
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        self.client.initialize_meta_agent(
            config=config
        )

    def add_chunk(self, chunk: str):
        self.client.add(
            user_id=self.user_id,
            messages=[
                {'role': 'user', 'content': chunk}
            ],
            async_add=False
        )
    
    def wrap_user_prompt(self, prompt: str):
        memories = self.client.retrieve_with_conversation(
            user_id=self.user_id,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )

        memory_context_lines = ["<memory_context>"]
        memories_found = False

        if memories.get("memories"):
            for memory_type, data in memories["memories"].items():
                if not data or data.get("total_count", 0) == 0:
                    continue

                # Prefer items, but fall back to recent/relevant shapes Mirix may return
                items = data.get("items", [])
                if memory_type == "episodic" and not items:
                    seen_ids = set()
                    items = []
                    for item in data.get("recent", []) + data.get("relevant", []):
                        item_id = item.get("id")
                        if item_id in seen_ids:
                            continue
                        seen_ids.add(item_id)
                        items.append(item)
                if not items and "recent" in data:
                    items = data.get("recent", [])
                if not items:
                    continue

                for item in items:
                    memories_found = True
                    tag_name = f"{memory_type}_memory"

                    if memory_type == "core":
                        label = item.get("label", "")
                        value = item.get("value", "")
                        content = f"{label}: {value}".strip(": ").strip()
                        if not content:
                            content = item.get("summary", "") or str(item)
                    else:
                        content = (
                            item.get("summary")
                            or item.get("caption")
                            or item.get("name")
                            or item.get("title")
                            or item.get("description")
                            or item.get("value", "")
                        )
                        if not content:
                            content = str(item)

                    memory_context_lines.append(f"<{tag_name}>{content}</{tag_name}>")

        if not memories_found:
            memory_context_lines.append("None")

        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")

        return "\n".join(memory_context_lines)
