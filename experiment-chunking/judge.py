from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import threading
from typing import Any, Optional
import argparse

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field

from utils import calculate_cost, judge_instructions, map_progress

_client_local = threading.local()

def _get_client() -> OpenAI:
    """Return a thread-local OpenAI client, creating one if needed."""
    client = getattr(_client_local, "client", None)
    if client is None:
        http_client = httpx.Client()
        client = OpenAI(http_client=http_client)
        _client_local.client = client
        _client_local.http_client = http_client
    return client

def _close_thread_client() -> None:
    """Clean up the thread-local HTTP client."""
    http_client = getattr(_client_local, "http_client", None)
    if http_client is not None:
        http_client.close()
        delattr(_client_local, "http_client")
    if hasattr(_client_local, "client"):
        delattr(_client_local, "client")

@dataclass
class JudgeConfig():
    llm_model_name: str = "gpt-5-nano"


class CheckName(str, Enum):
    answer_relevant = "answer_relevant"
    completeness = "completeness"
    grounding_accuracy = "grounded_accuracy"
    context_utilization = "context_utilization"
    chunk_coverage = "chunk_coverage"
    consistency = "consistency"
    faithful_to_source = "faithful_to_source"
    focused = "focused"
    uncertainty_handling = "uncertainty_handling"
  

CHECK_DESCRIPTIONS = {
    CheckName.answer_relevant: "The answer directly address the user's question.",
    CheckName.completeness: "The answer cover all key points requested.",
    CheckName.grounding_accuracy: "All claims are supported by retrieved chunks (no hallucinations).",
    CheckName.context_utilization: "The answer utilized the provided snippets well and kept generic knowledge to a minimal.",
    CheckName.chunk_coverage: "The answer utilized multiple chunks effectively and rarely missed relevant information.",
    CheckName.consistency: "The answer avoids conflicting statements and contradictions.",
    CheckName.faithful_to_source: "The wording or paraphasing in the answer does not misrepresent the tone or meaning of the source.",
    CheckName.focused: "The response is focused and free of fluff.",
    CheckName.uncertainty_handling: "The model explictly indicates uncertainty when chunks lack coverage."
}

class EvaluationCheck(BaseModel):
    check_name: CheckName = Field(description="The type of evaluation check")
    reasoning: str = Field(description="The reasoning behind the check result")
    check_pass: bool = Field(description="Whether the check passed (True) or failed (False)")
    
class EvaluationChecklist(BaseModel):
    checklist: list[EvaluationCheck] = Field(description="List of all evaluation checks")
    summary: str = Field(description="Evaluation summary")

def run_judge_structured(
    instructions: str,
    user_query: str,
    output_format: type[BaseModel],
    config: Optional[JudgeConfig] = None,
) -> tuple[EvaluationChecklist, Any]:
    """Call the LLM judge and parse the response into the requested structure."""

    if config is None:
        config = JudgeConfig()
        model = config.llm_model_name

    client = _get_client()
   
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_query}]
    
    try:
        response = client.responses.parse(
            model=model,
            input=messages,
            text_format=output_format
        )
        return (response.output_parsed, response.usage)
    finally:
        _close_thread_client()

def run_eval(row: dict[str, Any]) -> tuple[dict[str, Any], tuple[EvaluationChecklist, Any]]:
    """Run the judge with structured output."""

    user_query = f"""
        <QUESTION>{row['question']}</QUESTION>
        <ANSWER>{row['answer']}</ANSWER>
        <CONTEXT>{row["context"]}</CONTEXT>
        """.strip()
    
    output = run_judge_structured(instructions=judge_instructions, user_query=user_query, output_format=EvaluationChecklist)
    return row, output

def run_eval_concurrent(
    doc: list[dict[str, Any]],
    outpath: str,
    config: Optional[JudgeConfig] = None,
) -> list[dict[str, Any]]:
    """Evaluate many rows in parallel and persist the aggregated scoring table."""

    if config is None:
        config = JudgeConfig()
        model = config.llm_model_name

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = map_progress(pool, doc, run_eval)

    all_checks = []
    for original_row, result in results:
        checklist, usage = result
        checks = checklist.checklist
        cost = calculate_cost(model, usage.input_tokens, usage.output_tokens)
        checks_formatted = {
            'question': original_row['question'],
            'input_tokens': usage.input_tokens,
            'output_tokens': usage.output_tokens, 
            'total_tokens': usage.total_tokens,
            'input_cost': float(cost[0]),
            'output_cost': float(cost[1]),
            'total_cost': float(cost[2])
        }
        for check in checks:
            checks_formatted[check.check_name] = check.check_pass
        all_checks.append(checks_formatted)

    Path(outpath).write_text(json.dumps(all_checks, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_checks


def main():

    parser = argparse.ArgumentParser(description="Run the structured LLM judge.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSON file containing ground truth data.",
    )
    parser.add_argument(
        "--outpath",
        required=True,
        help="Path to write the aggregated judge results as JSON.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        doc = json.load(f)

    run_eval_concurrent(doc=doc, outpath=args.outpath)


if __name__ == "__main__":
    main()
