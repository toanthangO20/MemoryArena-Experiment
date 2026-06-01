import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI
import re
from typing import Generator

try:
    from zep_cloud.client import Zep
    from zep_cloud.types import Message

except ImportError:
    print("Zep not found, please install zep-cloud")
    
class ZepMemorySystem:
    def __init__(
        self,
        user_id: Optional[str] = None,
        chunk_size=500,
        chunk_overlap=50,
    ):
       
        self.user_id = user_id if user_id is not None else str(uuid.uuid4()) # must have user_id
        self.zep_client=Zep(api_key=os.getenv("ZEP_API_KEY"))
        self.zep_client.user.add(user_id=self.user_id)

        thread_id = uuid.uuid4().hex # A new thread identifier
        self.zep_client.thread.create(
            thread_id=thread_id,
            user_id=self.user_id,
        )
        self.thread_id = thread_id
                
        # self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
     
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap

    def _split_long_paragraph(
        self,
        paragraph: str,
        chunk_size: int,
        chunk_overlap: int
        ) -> Generator[str, None, None]:
        """Split a long paragraph by sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                if current_chunk:
                    yield current_chunk.strip()
                    if chunk_overlap > 0:
                        overlap = current_chunk[-chunk_overlap:]
                        first_space = overlap.find(' ')
                        if first_space > 0:
                            current_chunk = overlap[first_space + 1:] + " "
                        else:
                            current_chunk = ""
                    else:
                        current_chunk = ""
            current_chunk += sentence + " "

        if current_chunk.strip():
            yield current_chunk.strip()

    def _split_chunk(
        self,
        text: str,
        ) -> Generator[tuple[int, str], None, None]:
        """
        Split a document into chunks with configurable size and overlap.

        Args:
            text: The full document text
            chunk_size: Maximum characters per chunk (default 6000 to leave room for context)
            chunk_overlap: Characters to overlap between chunks for continuity

        Yields:
            Tuple of (chunk_index, chunk_text)
        """
        chunk_size=self.chunk_size
        chunk_overlap=self.chunk_overlap
        
        if not text:
            return

        text = text.strip()
        paragraphs = text.split('\n\n')

        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph exceeds chunk_size, yield current chunk
            if len(current_chunk) + len(paragraph) + 2 > chunk_size:
                if current_chunk:
                    yield (chunk_index, current_chunk.strip())
                    chunk_index += 1

                    # Start new chunk with overlap from previous
                    if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                        overlap_text = current_chunk[-chunk_overlap:]
                        first_space = overlap_text.find(' ')
                        if first_space > 0:
                            overlap_text = overlap_text[first_space + 1:]
                        current_chunk = overlap_text + "\n\n"
                    else:
                        current_chunk = ""

                # Handle single paragraphs longer than chunk_size
                if len(paragraph) > chunk_size:
                    for sub_chunk in self._split_long_paragraph(paragraph, chunk_size, chunk_overlap):
                        yield (chunk_index, sub_chunk)
                        chunk_index += 1
                    current_chunk = ""
                else:
                    current_chunk += paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Yield final chunk
        if current_chunk.strip():
            yield (chunk_index, current_chunk.strip())

    def add_chunk(self, chunk: str):
        if len(chunk) <= 10000:
            self.zep_client.thread.add_messages(
                thread_id=self.thread_id,
                messages=[Message(
                    # name='User',
                    role="user",
                    content=chunk
                )]
            )
            # episode = self.zep_client.graph.add(
            # user_id=self.user_id,
            # type="text",
            # data=chunk
            # )
            # all_user_edges = self.zep_client.graph.edge.get_by_user_id(user_id=self.user_id)

        else:
            chunk_data=self._split_chunk(chunk)
            stats={"successful":0}
            for idx, chunk_data in chunk_data:
                # episode = self.zep_client.graph.add(
                #     user_id=self.user_id,
                #     type="text",
                #     data=chunk_data
                # )
                self.zep_client.thread.add_messages(
                    thread_id=self.thread_id,
                    messages=[Message(
                        role="user",
                        content=chunk_data
                    )]
                )
                stats["successful"] += 1

    def wrap_user_prompt(self, prompt: str):
        memory_context_lines = ["<memory_context>"]
        
        results = self.zep_client.graph.search(user_id=self.user_id, query=prompt, scope="edges")        

        relevant_edges = results.edges
        if relevant_edges is not None:
            formatted_facts = []
            for edge in relevant_edges:
                valid_at = edge.valid_at if edge.valid_at is not None else "date unknown"
                invalid_at = edge.invalid_at if edge.invalid_at is not None else "present"
                try:
                    fact=edge.fact
                except Exception as e: 
                    print("Error in edge.fact:", edge) # if there is no fact then skip this edge
                    continue
                formatted_fact = f"{fact} (Date range: {valid_at} - {invalid_at})"
                formatted_facts.append(formatted_fact)
        
            for fact in formatted_facts:
                memory_context_lines.append(fact)
        else:
            memory_context_lines.append("None")
            
        memory_context_lines.append("</memory_context>")
        memory_context_lines.append(f"User: {prompt}")
        
        return "\n".join(memory_context_lines)


       