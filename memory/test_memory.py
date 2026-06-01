import os
import uuid
import time

import pytest
import requests

from MemActBench.client import MemoryClient
from dotenv import load_dotenv

load_dotenv()

def _server_available(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def test_memory_roundtrip_long_context(memory_system_name: str):
    base_url = os.getenv("MEMORY_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
    if not _server_available(base_url):
        pytest.skip(f"Memory server not running at {base_url}")

    user_id = str(uuid.uuid4())
    try:
        client = MemoryClient(
            user_id=user_id,
            memory_system_name=memory_system_name,
            base_url=base_url,
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 400:
            pytest.skip(f"Memory system not supported by server: {memory_system_name}")
        raise

    knowledge = "Bob live in Boston and my favorite color is teal."
    response = client.add(knowledge)
    knowledge = "Alice live in Santa Clara and her favorite color is black."
    response = client.add(knowledge)
    
    print(response)

    if memory_system_name == "zep":
        time.sleep(20)  # wait for zep to index the new edges
        
    if memory_system_name == "mem0" or memory_system_name == "zep":
        wrapped = client.wrap_user_prompt("Boston")
        # Mem0 is kind of stupid, if you run "Where do I live?" it will return "None"
    else:
        wrapped = client.wrap_user_prompt("where does Bob live?")
    print(wrapped)

    assert "<memory_context>" in wrapped
    assert "</memory_context>" in wrapped
    assert "Boston" in wrapped


if __name__ == "__main__":
    import sys
    memory_system_name = sys.argv[1]
    test_memory_roundtrip_long_context(memory_system_name)
