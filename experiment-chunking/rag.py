import json
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
        self.client = OpenAI()
        self.instructions = instructions
        

default_config = RAGConfig()
pricing = PricingConfig()

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

def ask_llm(user_query, config=default_config, model="gpt-4o-mini"):
    """Send the prepared prompt to the LLM client and return the raw SDK response."""

    client = config.client
    instructions = config.instructions

    messages = []


    messages.append({
        "role": "system",
        "content": instructions
    })

    messages.append({
        "role": "user",
        "content": user_query
    })

    response = client.responses.create(
        model=model,
        input=messages
    )

    return response

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
    input_cost = cost[0]
    output_cost = cost[1]
    total_cost = cost [2]

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

