import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

try:
    from lightmem.memory.lightmem import LightMemory
except ImportError:
    print("LightMemory not found, please install LightMem")

load_dotenv()

class LightMemMemorySystem:
    def __init__(
        self,
        user_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if user_id is None:
            # randomly generate a user id
            user_id = str(uuid.uuid4())[:4]
        self.user_id = user_id
        if config is None:
            lightmem_data_dir = os.getenv("LIGHTMEM_DATA_DIR")
            logs_root = lightmem_data_dir + "/logs"
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_log_dir = os.path.join(logs_root, run_timestamp)
            os.makedirs(run_log_dir, exist_ok=True)

            api_key = os.getenv("OPENAI_API_KEY")
            api_base_url = "https://api.openai.com/v1"
            llm_model = "gpt-4.1-mini"
            embedding_model_path = "sentence-transformers/all-MiniLM-L6-v2"
            llmlingua_model_path = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

            config = {
                "pre_compress": True,
                "pre_compressor": {
                    "model_name": "llmlingua-2",
                    "configs": {
                        "llmlingua_config": {
                            "model_name": llmlingua_model_path,
                            "device_map": "cuda",
                            "use_llmlingua2": True,
                        },
                    },
                },
                "topic_segment": True,
                "precomp_topic_shared": True,
                "topic_segmenter": {
                    "model_name": "llmlingua-2",
                },
                "messages_use": "user_only",
                "metadata_generate": True,
                "text_summary": True,
                "memory_manager": {
                    "model_name": "openai",
                    "configs": {
                        "model": llm_model,
                        "api_key": api_key,
                        "max_tokens": 16000,
                        "openai_base_url": api_base_url,
                    },
                },
                "extract_threshold": 0.1,
                "index_strategy": "embedding",
                "text_embedder": {
                    "model_name": "huggingface",
                    "configs": {
                        "model": embedding_model_path,
                        "embedding_dims": 384,
                        "model_kwargs": {"device": "cuda"},
                    },
                },
                "retrieve_strategy": "embedding",
                "embedding_retriever": {
                    "model_name": "qdrant",
                    "configs": {
                        "collection_name": self.user_id,
                        "embedding_model_dims": 384,
                        "path": lightmem_data_dir,
                    },
                },
                "update": "offline",
                "logging": {
                    "level": "DEBUG",
                    "file_enabled": True,
                    "log_dir": run_log_dir,
                },
            }

        self.lightmem = LightMemory.from_config(config)

    def add_chunk(self, chunk: str):
        if not chunk or not chunk.strip():
            return None
        timestamp = datetime.now().strftime("%Y-%m-%d")
        messages = [
            {"role": "user", "content": chunk, "time_stamp": timestamp},
            {
                "role": "assistant",
                "content": "Thanks for the information! I will remember this.",
                "time_stamp": timestamp,
            },
        ]
        return self.lightmem.add_memory(
            messages=messages,
            force_segment=True,
            force_extract=True,
        )

    def wrap_user_prompt(self, prompt: str):
        related_memories = self.lightmem.retrieve(prompt, limit=5)
        memory_context_lines = "<memory_context>\n" + related_memories.strip() + "\n</memory_context>\n"
        memory_context_lines += f"User: {prompt}"
        return memory_context_lines

    def _extract_memory_text(self, item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("content", "memory", "text", "summary"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            payload = item.get("payload")
            if isinstance(payload, dict):
                for key in ("content", "memory", "text", "summary"):
                    value = payload.get(key)
                    if isinstance(value, str) and value.strip():
                        return value
        return str(item) if item is not None else ""
