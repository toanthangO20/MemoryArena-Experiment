import math
import os
import re
from typing import List, Optional, Sequence

import tiktoken


class RAGMemorySystem:
    """
    RAG-style memory with either BM25 or embedding retrieval.
    Stores chunks up to max_tokens and retrieves the most relevant ones.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        max_tokens: int = 2048,
        top_k: int = 3,
        user_id: Optional[str] = None,
    ):
        self.retrieval_method = self._normalize_retrieval_method(retrieval_method)
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.user_id = user_id
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

        self._chunks: List[str] = []

        # BM25 state
        self._bm25 = None
        self._bm25_corpus: List[List[str]] = []

        # Embedding state
        self._embedding_model = "text-embedding-3-small"
        self._embeddings: List[List[float]] = []
        self._embedding_client = None

    def add_chunk(self, chunk: str):
        for piece in self._split_chunk(chunk):
            if not piece.strip():
                continue
            self._chunks.append(piece)
            if self.retrieval_method == "bm25":
                self._add_bm25_doc(piece)
            else:
                self._add_embedding_doc(piece)

    def wrap_user_prompt(self, prompt: str) -> str:
        memory_context_lines = ["<memory_context>"]

        if not self._chunks:
            memory_context_lines.append("None")
        else:
            if self.retrieval_method == "bm25":
                results = self._bm25_retrieve(prompt)
            else:
                results = self._embedding_retrieve(prompt)

            if results:
                for chunk in results:
                    memory_context_lines.append(f"<memory>{chunk}</memory>")
            else:
                memory_context_lines.append("None")

        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")
        return "\n".join(memory_context_lines)

    def _split_chunk(self, chunk: str) -> List[str]:
        tokens = self.tokenizer.encode(chunk, disallowed_special=())
        if len(tokens) <= self.max_tokens:
            return [chunk]
        pieces = []
        for start in range(0, len(tokens), self.max_tokens):
            piece_tokens = tokens[start : start + self.max_tokens]
            pieces.append(self.tokenizer.decode(piece_tokens))
        return pieces

    def _normalize_retrieval_method(self, retrieval_method: str) -> str:
        method = (retrieval_method or "").strip().lower()
        if method in {"bm25", "bm-25"}:
            return "bm25"
        if method in {"embedding", "embeddings", "text-embedding-3-small", "text-embedding-small-3"}:
            return "embedding"
        raise ValueError(f"Unsupported retrieval_method: {retrieval_method}")

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _add_bm25_doc(self, text: str):
        tokens = self._tokenize(text)
        self._bm25_corpus.append(tokens)
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank_bm25 package is required for BM25 retrieval") from exc
        self._bm25 = BM25Okapi(self._bm25_corpus)

    def _bm25_retrieve(self, query: str) -> List[str]:
        if self._bm25 is None:
            return []
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        scores = self._bm25.get_scores(query_terms)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: self.top_k]
        if not ranked:
            return []
        # Return top-ranked docs regardless of score sign.
        return [self._chunks[idx] for idx, _score in ranked]

    def _init_embedding_client(self):
        if self._embedding_client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is required for embedding retrieval") from exc
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for embedding retrieval")
        self._embedding_client = OpenAI(api_key=api_key)

    def _add_embedding_doc(self, text: str):
        embedding = self._get_embedding(text)
        self._embeddings.append(embedding)

    def _get_embedding(self, text: str) -> List[float]:
        self._init_embedding_client()
        response = self._embedding_client.embeddings.create(
            model=self._embedding_model,
            input=[text],
        )
        return response.data[0].embedding

    def _embedding_retrieve(self, query: str) -> List[str]:
        if not self._embeddings:
            return []
        query_embedding = self._get_embedding(query)
        scored = []
        query_norm = self._vector_norm(query_embedding)
        for idx, embedding in enumerate(self._embeddings):
            score = self._cosine_similarity(query_embedding, query_norm, embedding)
            scored.append((score, idx))
        top = sorted(scored, key=lambda item: item[0], reverse=True)[: self.top_k]
        return [self._chunks[idx] for score, idx in top if score > 0]

    def _vector_norm(self, vector: Sequence[float]) -> float:
        return math.sqrt(sum(value * value for value in vector))

    def _cosine_similarity(
        self,
        query_vector: Sequence[float],
        query_norm: float,
        doc_vector: Sequence[float],
    ) -> float:
        denom = query_norm * self._vector_norm(doc_vector)
        if denom <= 0:
            return 0.0
        dot = sum(q * d for q, d in zip(query_vector, doc_vector))
        return dot / denom
