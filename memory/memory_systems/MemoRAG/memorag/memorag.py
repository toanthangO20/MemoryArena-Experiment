import json
import os
from itertools import chain
from typing import List, Optional, Union

import logging
import tiktoken
from openai import OpenAI
from semantic_text_splitter import TextSplitter

from .prompt import en_prompts, zh_prompts
from .retrieval import FaissIndex, DenseRetriever

logger = logging.getLogger(__name__)          

class Model:
    def __init__(
        self, 
        model_name_or_path: str, 
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_client=None,
    ):  
        self.model_name_or_path = model_name_or_path
        self.client = api_client
        if self.client is None:
            if api_endpoint:
                self.client = OpenAI(base_url=api_endpoint, api_key=api_key)
            else:
                self.client = OpenAI(api_key=api_key)
        logger.info(f"API model configured for {model_name_or_path}")

    def _api_generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float = None,
        top_p: float = None,
        repetition_penalty: float = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name_or_path,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content or ""

    def generate(
        self, 
        prompts: Union[str, List[str]], 
        batch_size: int = 1, 
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = False,
        repetition_penalty:float=1.0
    ) -> Union[str, List[str]]:

        if isinstance(prompts, str):
            prompts = [prompts]

        outputs = []
        for prompt in prompts:
            outputs.append(
                self._api_generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            )
        return outputs


class Memory(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = None
        self.memo_type = "api"

        if self.model_name_or_path.lower().find("chinese") != -1:
            self.prompts = zh_prompts
        else:
            self.prompts = en_prompts

    def memorize(
        self, 
        context, 
        max_length=None,
        reload_model:bool=True
    ):
        self.memory = context

    def reset(
        self
    ) -> None:
        self.memory = None

    def answer(
        self,
        query, max_new_tokens=128) -> str:
        return self.generate(self.prompts["qa"], query, max_new_tokens=max_new_tokens)[0]

    def recall(
        self,
        query, max_new_tokens=128) -> str:
        return self.generate(self.prompts["span"], query, max_new_tokens=max_new_tokens)[0]

    def rewrite(
        self,
        query, max_new_tokens=128) -> str:
        return self.generate(self.prompts["sur"], query, max_new_tokens=max_new_tokens)[0]

    def summarize(
        self, max_new_tokens:int=512) -> str:
        return self.generate(self.prompts["sum"], max_new_tokens=max_new_tokens)[0]

    def generate(
        self, 
        instruct: Union[str, List[str]], 
        query: str = "",  
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = False,
        with_cache: bool = True
    ) -> List[str]:
        if not self.memory:
            raise ValueError("Memory is not initialized. Please ensure that memory has been formed before using generate.")

        if isinstance(instruct, str):
            instruct = [instruct]

        outputs = []
        context_prompt = self.prompts["context"].format(context=self.memory)
        for inst in instruct:
            if query:
                inst_text = inst.format(question=query)
            else:
                inst_text = inst
            full_prompt = f"{context_prompt}\n\n{inst_text}"
            outputs.append(
                self._api_generate(
                    full_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )
        return outputs
    
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"memory": self.memory}, f, ensure_ascii=False, indent=2)
        
    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.memory = payload.get("memory")
        

class MemoRAG:
    def __init__(
        self, 
        mem_model_name_or_path: str, 
        ret_model_name_or_path: str = None,
        ret_hit:int=3,
        retrieval_chunk_size:int=512,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_client=None,
        cache_dir: Optional[str] = None,
        **_ignored,
    ):

        if mem_model_name_or_path.lower().find("chinese") != -1:
            self.prompts = zh_prompts
            retrieval_chunk_size = 2048
        else:
            self.prompts = en_prompts

        self.mem_model = Memory(
            mem_model_name_or_path,
            api_endpoint=api_endpoint,
            api_key=api_key,
            api_client=api_client,
        )

        self.retriever = DenseRetriever(
            ret_model_name_or_path,
            hits=ret_hit,
            cache_dir=cache_dir,
            api_client=api_client,
            api_endpoint=api_endpoint,
            api_key=api_key,
        )

        self.text_splitter = TextSplitter.from_tiktoken_model(
            "gpt-3.5-turbo", retrieval_chunk_size)

    def memorize(self, context: str, save_dir: str = None, print_stats: bool = False):
        self.retriever.remove_all()

        self.mem_model.memorize(context)
        self.retrieval_corpus = self.text_splitter.chunks(context)
        self.retriever.add(self.retrieval_corpus)

        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.mem_model.save(os.path.join(save_dir, "memory.bin"))
            self.retriever._index.save(os.path.join(save_dir, "index.bin"))
            with open(os.path.join(save_dir, "chunks.json"), "w") as f:
                json.dump(self.retrieval_corpus, f, ensure_ascii=False, indent=2)
            if print_stats:
                self._print_stats(save_dir, context)

    def _print_stats(self, save_dir: str, context: str=None):
        memory_path = os.path.join(save_dir, "memory.bin")
        memory_size_kb = os.path.getsize(memory_path) / 1024
        print(f"Memory file size: {memory_size_kb:.1f} KB")

        encoding = tiktoken.get_encoding("cl100k_base")
        if context:
            encoded_context = encoding.encode(context)
            print(f"Encoded context length: {len(encoded_context)} tokens")
        print(f"Number of chunks in retrieval corpus: {len(self.retrieval_corpus)}")


    def load(self, save_dir: str, print_stats: bool = False):
        self.mem_model.load(os.path.join(save_dir, "memory.bin"))
        _index = FaissIndex(self.retriever.device)
        _index.load(os.path.join(save_dir, "index.bin"))
        self.retriever._index = _index
        self.retrieval_corpus = json.load(open(os.path.join(save_dir, "chunks.json")))
        if print_stats:
            self._print_stats(save_dir)
            
    def __call__(
        self, 
        query: str = None, 
        context: str = None, 
        task_type: str = "retrieval", 
        reset_each_call: bool = False,
        use_memory_answer: bool = False
    ):
        if reset_each_call:
            self.mem_model.reset()
            self.retriever.remove_all()

        if not self.mem_model.memory:
            if not context:
                raise ValueError("Please provide your input context...")
            self.memorize(context)

        if task_type in {"retrieval", "memorag"}:
            return self.retrieve(query, use_memory_answer=use_memory_answer)
        else:
            raise NotImplementedError(f"Task type '{task_type}' is not supported.")

    def retrieve(self, query: str, use_memory_answer: bool = False):
        text_spans = self.mem_model.recall(query)
        surrogate_queries = self.mem_model.rewrite(query)
        retrieval_query, potential_answer = self._prepare_retrieval_query(query, text_spans, surrogate_queries, use_memory_answer)

        retrieval_results = self._retrieve(retrieval_query)

        if potential_answer:
            retrieval_results.append(f"The answer might be {potential_answer}.")

        return retrieval_results

    def _prepare_retrieval_query(self, query, text_spans, surrogate_queries, use_memory_answer):
        retrieval_query = text_spans.split("\n") + surrogate_queries.split("\n")
        retrieval_query = [q for q in retrieval_query if len(q.split()) > 3]
        potential_answer = None
        if use_memory_answer:
            potential_answer = self.mem_model.answer(query)
            retrieval_query.append(potential_answer)
        retrieval_query.append(query)
        return retrieval_query, potential_answer

    def _retrieve(self, retrieval_query):
        topk_scores, topk_indices = self.retriever.search(queries=retrieval_query)
        topk_indices = list(chain(*[topk_index.tolist() for topk_index in topk_indices]))
        topk_indices = sorted(set([x for x in topk_indices if x > -1]))
        return [self.retrieval_corpus[i].strip() for i in topk_indices]
