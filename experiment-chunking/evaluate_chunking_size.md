# Test Chunk Size
I want to evaluate chunk size and each a LLM judge to evaluate which chunk size leads to the best answers based on a number of llm generated questions.

- [x] Extract text from pdf 
- [x] Merge all pages as one list
- [x] Create a question list for every 400 characters
- [x] Test with a sample
    - [x] Create RAG pipeline for testing; this is a synchronous testing pipeline
- [ ] Write asychronous pipeline
    - [ ] Generate reponses for 300/250 size-step chunks
    - [ ] Repeat above for 600/500 size-step chunks
    - [ ] Repeat above for 100/833 size-step chunks

## Synchronous pipeline (test)
chunk -> embed -> index -> search -> build prompt -> ask llm

Returns:
``` 
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
``` 

## Asynchronous pipeline 
- 

### Notes
1. Organize packages into standard, external, and local libraries.
2. `getattr(obj, "attr_name", default) `fetches an attribute dynamically—like obj.attr_name but with a fallback. If obj lacks that attribute, it returns default instead of raising AttributeError. Handy when the attribute name is computed at runtime or optional, e.g., usage = getattr(response, "usage", None) pulls the usage field if it exists; otherwise you get None.
    - ResponseUsage is an object, not a dict, so it doesn’t implement .get. Stick with attribute access:

    ``` 
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None) 
    ```
3. `getattr` works on any Python object’s attributes (e.g., obj.foo) and uses attribute lookup rules—descriptors, properties, __getattr__, etc. It defaults to raising AttributeError unless you provide a fallback. .get is a dictionary (or dict-like) method that looks up keys (e.g., mapping["foo"]), returning None or the supplied default when the key is missing. In short: getattr → attributes, works for any object; .get → dictionary keys only, no descriptors.
4. A smoke test is a lightweight sanity check you run before deeper testing—just enough to confirm the basic pipeline works without crashing. For RAG, that means a small end-to-end run (e.g., index a few pages, answer a couple of questions) to verify chunking, retrieval, and the LLM call all succeed. If the smoke test fails, you fix the fundamentals before spending time on heavier evaluations, async fan-out, or full regression suites.
5. A blocking loop is a loop that runs synchronously—each iteration must finish before the next starts, and nothing else can execute until it completes. In other words, the loop “blocks” the thread, so long-running work (like API calls) holds up the entire process until it finishes. That’s what you have now with for question in tqdm(...): run_rag(question)—each RAG call blocks the loop until the LLM returns.