import json
from dataclasses import dataclass, asdict
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

import httpx
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd

from toyaikit.pricing import PricingConfig
from chunking import extract_text_from_pdf, chunk_sliding_window
from minsearch import VectorSearch
from utils import instructions, prompt_template


@dataclass
class RAGConfig:
    """Container for all reusable RAG components (embeddings, vector index, LLM)."""
    embedding_model_name: str = "multi-qa-distilbert-cos-v1"
    llm_model_name: str = "gpt-4o-mini"
    def __post_init__(self):
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.vindex = VectorSearch(keyword_fields=[])
        self.instructions = instructions
        

default_config = RAGConfig()
pricing = PricingConfig()
_client_local = threading.local()

@dataclass
class RAGResult:
    """Structured output capturing the answer text plus token and cost metadata."""
    question: str
    answer: str
    context: List[dict]
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

class ProcessChunks():
    """Helper for generating chunks and loading them into the shared vector index."""

    def __init__(self, config=default_config):
        self.config = config
        self.embedding_model = config.embedding_model
        self.vindex = config.vindex


    def get_chunks(self, size:int, step:int, start_page:int, end_page:Optional[int]=None) -> List:
        """Slice PDF pages into overlapping windows."""
        pages = extract_text_from_pdf(start_page=start_page, end_page=end_page)
        chunks = chunk_sliding_window(pages, size, step)
        return chunks


    def embed_chunks(self, chunks):
        """Encode each chunk into vector space."""
        embeddings = []
        for chunk in tqdm(chunks):
            vector = self.embedding_model.encode(chunk["text"])
            embeddings.append(vector)
        
        return np.array(embeddings)
    
    def index_chunks(self, embeddings:np.array, chunks:List[dict]):
        """Store embeddings and associated metadata in the vector index."""
        return self.vindex.fit(embeddings, chunks)

def search(user_query:str, config=default_config):
    """Encode a query and fetch the top matches from the configured vector index."""

    embedding_model = config.embedding_model
    vindex = config.vindex

    user_query_embedding = embedding_model.encode(user_query)
    search_results = vindex.search(user_query_embedding, num_results=10)
    return search_results

def build_prompt(user_query, search_results):
    """Project retrieved chunks into the downstream LLM prompt template."""
    search_json = json.dumps(search_results)
    prompt = prompt_template.format(user_question=user_query, context=search_json).strip()
    return prompt


def ask_llm(user_query, config=default_config):
    """Send the prepared prompt to the LLM client and return the raw SDK response."""
    model = config.llm_model_name
    instructions = config.instructions

    http_client = None
    client = getattr(_client_local, "client", None)
    if client is None:
        http_client = httpx.Client()
        client = OpenAI(http_client=http_client)
        _client_local.client = client

    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_query}]
    
    try:
        response = client.responses.create(
            model=model,
            input=messages
        )
        return response
    finally:
        if http_client is not None:
            http_client.close()
            del _client_local.client

def calculate_cost(model, input_tokens, output_tokens):
    """Convert token counts into dollar costs using the configured pricing table."""
    cost = pricing.calculate_cost(model, input_tokens, output_tokens)
    input_cost = cost.input_cost
    output_cost = cost.output_cost
    total_cost = cost.total_cost
    return input_cost, output_cost, total_cost


def run_rag(user_query, config=default_config) -> RAGResult:
    """Execute retrieval + generation and return a RAGResult with token/cost stats."""
    model = config.llm_model_name
    
    search_results = search(user_query, config)
    user_prompt = build_prompt(user_query, search_results)
    response = ask_llm(user_prompt, config)
    answer_text = response.output_text

    usage = getattr(response, "usage")
    input_tokens = getattr(usage, "input_tokens")
    output_tokens = getattr(usage, "output_tokens")

    cost = calculate_cost(model, input_tokens, output_tokens)
    input_cost = float(cost[0])
    output_cost = float(cost[1])
    total_cost = float(cost[2])

    results = RAGResult(
        question=user_query,
        answer=answer_text,
        context=search_results,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost, 
        total_cost=total_cost
    )
    return results


def map_progress(pool, seq, f):
    """Map function f over seq using the provided executor pool while
    displaying a tqdm progress bar. Returns a list of results in submission order.
    """
    results = []
    
    with tqdm(total=len(seq)) as progress:
        futures = []
    
        for el in seq:
            future = pool.submit(f, el)
            future.add_done_callback(lambda p: progress.update())
            futures.append(future)

        for future in futures:
            result = future.result()
            results.append(result)
        
        return results


def run_rag_concurrent(path_to_ground_truth:str, outpath:str):
    ground_truth = pd.read_csv(path_to_ground_truth)
    records = ground_truth[["question", "summary_answer"]].to_dict(orient="records")
    questions = [r["question"] for r in records]
    summary_answers = [r["summary_answer"] for r in records]

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = map_progress(pool, questions, run_rag)
    
    json_rows=[]
    for summary_answer, rag_result in zip(summary_answers, results):
        data = asdict(rag_result)
        data["reference_answer"] = summary_answer
        json_rows.append(data)

    Path(outpath).write_text(json.dumps(json_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return results

def main(path_to_ground_truth="sample_gt.csv", outpath="rag_results_1000_833.json", size=1000, step=833, start_page=1):
    """Build the index and run the RAG batch."""
    pc = ProcessChunks(config=default_config)
    chunks = pc.get_chunks(size=size, step=step, start_page=start_page)
    embeddings = pc.embed_chunks(chunks)
    pc.index_chunks(embeddings, chunks)

    results = run_rag_concurrent(
        path_to_ground_truth=path_to_ground_truth,
        outpath=outpath,
    )
    print(f"Wrote {len(results)} results to {outpath}")


if __name__ == "__main__":
    main()
