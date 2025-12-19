from tqdm import tqdm
from toyaikit.pricing import PricingConfig

instructions = """ 

You are an AI Researcher with 10 years of experience as a senior AI Engineer. 
You are instructing a course based on the "AI Engineering book" published by Chip Huyen.
Answer the QUESTION and base your response ONLY on the CONTEXT provided by this book.
""".strip()

prompt_template = """ 
<QUESTION>
{user_question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
"""


judge_instructions = """
Use this checklist to evaluate the quality of the answer of a LLM
(<ANSWER>) to a user question (<QUESTION>). In <CONTEXT> is the context that the LLM used to create the response.

For each item of the checklist, check if the condition is met. 

Checklist:

- answer_relevant: The answer directly address the user's question.
- completeness: The answer cover all key points requested.
- grounding_accuracy: All claims are supported by retrieved chunks (no hallucinations).
- context_utilization: The answer utilized the provided snippets well and kept generic knowledge to a minimal.
- chunk_coverage: The answer utilized multiple chunks effectively and rarely missed relevant information.
- consistency: The answer avoids conflicting statements and contradictions.  
- focused: The response is focused and free of fluff.
- uncertainty_handling: The model explictly indicates uncertainty when chunks lack coverage.

Output true/false for each check and provide a short explanation for your judgment.
""".strip()

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

def calculate_cost(model, input_tokens, output_tokens):
    """Convert token counts into dollar costs using the configured pricing table."""
    pricing = PricingConfig()
    cost = pricing.calculate_cost(model, input_tokens, output_tokens)
    input_cost = cost.input_cost
    output_cost = cost.output_cost
    total_cost = cost.total_cost
    return input_cost, output_cost, total_cost