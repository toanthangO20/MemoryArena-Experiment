from __future__ import annotations

import os
import uuid
from typing import List, Optional, Sequence, Tuple


class GraphRAGMemorySystem:
    def __init__(
        self,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        top_k: int = 5,
        start_k: int = 1,
        max_depth: int = 2,
        edges: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        self.user_id = user_id or uuid.uuid4().hex
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.top_k = top_k
        self.start_k = start_k
        self.max_depth = max_depth
        self.edges = list(edges) if edges is not None else [("user_id", "user_id")]

        self._chunks: List[str] = []
        self._vector_store = None
        self._retriever = None
        self._embeddings = None

    def add_chunk(self, chunk: str):
        if not chunk or not chunk.strip():
            return
        self._chunks.append(chunk)
        doc = self._make_document(chunk)
        if self._vector_store is None:
            self._vector_store = self._get_vector_store(initial_docs=[doc])
        else:
            self._vector_store.add_documents([doc])
        self._retriever = None

    def wrap_user_prompt(self, prompt: str) -> str:
        memory_context_lines = ["<memory_context>"]
        if not self._chunks:
            memory_context_lines.append("None")
        else:
            retriever = self._get_retriever()
            docs = retriever.invoke(prompt)
            if docs:
                for doc in docs:
                    memory_context_lines.append(f"<memory>{doc.page_content}</memory>")
            else:
                memory_context_lines.append("None")
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")
        return "\n".join(memory_context_lines)

    def _make_document(self, chunk: str):
        try:
            from langchain_core.documents import Document
        except ImportError as exc:
            raise RuntimeError("langchain-core is required for GraphRAG retrieval") from exc
        return Document(page_content=chunk, metadata={"user_id": self.user_id})

    def _get_vector_store(self, initial_docs: Optional[List[object]] = None):
        if self._vector_store is not None:
            return self._vector_store
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for LangChain GraphRAG.")
        try:
            from langchain_core.vectorstores import InMemoryVectorStore
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "langchain-openai and langchain-core are required for GraphRAG retrieval"
            ) from exc
        self._embeddings = self._embeddings or OpenAIEmbeddings(api_key=self.api_key)
        if initial_docs:
            self._vector_store = InMemoryVectorStore.from_documents(
                documents=initial_docs,
                embedding=self._embeddings,
            )
        else:
            self._vector_store = InMemoryVectorStore(embedding=self._embeddings)
        return self._vector_store

    def _get_retriever(self):
        if self._retriever is not None:
            return self._retriever
        try:
            from graph_retriever.strategies import Eager
            from langchain_graph_retriever import GraphRetriever
        except ImportError as exc:
            raise RuntimeError("langchain-graph-retriever is required for GraphRAG.") from exc
        strategy = Eager(k=self.top_k, start_k=self.start_k, max_depth=self.max_depth)
        self._retriever = GraphRetriever(
            store=self._get_vector_store(),
            edges=self.edges,
            strategy=strategy,
        )
        return self._retriever
