from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path
from typing import List, Optional


class GraphRAGMemorySystem:
    def __init__(
        self,
        local_dir: str,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        index_method: str = "standard",
        query_method: str = "global",
        response_type: str = "Multiple Paragraphs",
    ):
        self.user_id = (user_id or uuid.uuid4().hex)[-4:]
        self.root_dir = Path(local_dir).expanduser().resolve() / self.user_id
        self.input_dir = self.root_dir / "input"
        self.api_key = api_key or os.getenv("GRAPHRAG_API_KEY")
        self.index_method = index_method
        self.query_method = query_method
        self.response_type = response_type

        self._chunks: List[str] = []
        print(f"DEBUG: Initializing GraphRAG workspace at {self.root_dir}")
        self._ensure_workspace()
        print("DEBUG: Cleaning existing chunks...")
        self._clean_existing_chunks()
        print("DEBUG: GraphRAG initialization complete.")

    def add_chunk(self, chunk: str):
        if not chunk or not chunk.strip():
            return
        self._chunks.append(chunk)
        self._write_chunk(chunk)
        self._run_index()

    def wrap_user_prompt(self, prompt: str):
        memory_context_lines = ["<memory_context>"]
        if not self._chunks:
            memory_context_lines.append("None")
        else:
            response_text = self._run_query(prompt)
            if response_text:
                memory_context_lines.append(f"<memory>{response_text}</memory>")
            else:
                memory_context_lines.append("None")
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")
        return "\n".join(memory_context_lines)

    def _ensure_workspace(self):
        print(f"DEBUG: Ensuring workspace directories exist...")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy default settings if they exist in the memory_systems directory
        workspace_settings_path = self.root_dir / "settings.yaml"
        if not workspace_settings_path.exists():
            default_settings = Path(__file__).parent / "settings.yaml"
            self._run_cli(["init", "--root", str(self.root_dir)])
            assert default_settings.exists()
            import shutil
            shutil.copy2(default_settings, workspace_settings_path)
    
        if self.api_key:
            print(f"DEBUG: Writing .env file...")
            env_path = self.root_dir / ".env"
            env_path.write_text(f"GRAPHRAG_API_KEY={self.api_key}\n", encoding="utf-8")
        elif not (self.root_dir / ".env").exists():
            raise ValueError("GRAPHRAG_API_KEY is required to run GraphRAG.")

    def _clean_existing_chunks(self):
        if not self.input_dir.exists():
            return
        for path in self.input_dir.glob("*"):
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
        self._chunks = []

    def _write_chunk(self, chunk: str) -> Path:
        filename = f"chunk_{uuid.uuid4().hex}.txt"
        path = self.input_dir / filename
        path.write_text(chunk, encoding="utf-8")
        return path

    def _run_index(self):
        # Check if there are any files in the input directory
        if not any(self.input_dir.iterdir()):
            print("DEBUG: No files in input directory, skipping indexing.")
            return
        args = ["index", "--root", str(self.root_dir)]
        if self.index_method:
            args.extend(["--method", self.index_method])
        self._run_cli(args)

    def _run_query(self, prompt: str) -> str:
        args = [
            "query",
            "--root",
            str(self.root_dir),
            "--method",
            self.query_method,
            "--query",
            prompt,
            "--response-type",
            self.response_type,
        ]
        return self._run_cli(args)

    def _run_cli(self, args: List[str]) -> str:
        cmd = ["graphrag"] + args
        print(f"DEBUG: Executing CLI command: {' '.join(cmd)}")
        env = os.environ.copy()
        if self.api_key:
            env["GRAPHRAG_API_KEY"] = self.api_key
        try:
            # Simplified: don't capture output, don't check return code.
            # This allows the command to print directly to the terminal.
            subprocess.run(
                cmd,
                cwd=str(self.root_dir),
                env=env,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("graphrag CLI not found. Install graphrag first.") from exc
        
        print(f"DEBUG: CLI command finished.")
        return "done"
