"""
inference.py
============
Three inference strategies for Polymarket prediction:

  1. baseline    — Vanilla LLM, no search. Pure parametric knowledge.
  2. rag         — Vanilla LLM + pre-retrieved Tavily context stuffed in prompt.
  3. search-r1   — Search-R1 checkpoint with interleaved search-and-reason.

All three return the same dict structure so evaluation is uniform.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SEARCH_TURNS = 4       # max search calls per question (paper's B=4)
MAX_NEW_TOKENS   = 512     # max tokens per generation step

# ── Model IDs ────────────────────────────────────────────────────────────────

MODELS = {
    "search-r1": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo",
    "baseline":  "Qwen/Qwen2.5-3B-Instruct",
    "rag":       "Qwen/Qwen2.5-3B-Instruct",   # same model, different prompting
}

# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

# Search-R1's native template (Table 1 from the paper).
# Adapted for prediction markets: asks for Yes/No answer.
SEARCH_R1_TEMPLATE = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if you "
    "find you lack some knowledge, you can call a search engine by "
    "<search> query </search>, and it will return the top searched results "
    "between <information> and </information>. You can search as many times as "
    "you want. If you find no further external knowledge needed, you can "
    "directly provide the answer inside <answer> and </answer> without "
    "detailed illustrations. For example, <answer> Yes </answer>. "
    "Question: {question}"
)

# ── Approach B (probability) for Search-R1 — uncomment to swap ───────────────
# SEARCH_R1_TEMPLATE = (
#     "Answer the given question. You must conduct reasoning inside <think> and "
#     "</think> first every time you get new information. After reasoning, if you "
#     "find you lack some knowledge, you can call a search engine by "
#     "<search> query </search>, and it will return the top searched results "
#     "between <information> and </information>. You can search as many times as "
#     "you want. If you find no further external knowledge needed, provide your "
#     "estimated probability (0-100) that the answer is Yes inside <answer> and "
#     "</answer> without detailed illustrations. For example, <answer> 75 </answer>. "
#     "Question: {question}"
# )

# Baseline: no search capability at all.
BASELINE_TEMPLATE = (
    "You are a prediction market forecaster. Given the following question about "
    "a future event, predict whether the outcome will be Yes or No. Think step "
    "by step, then provide your final answer.\n\n"
    "Question: {question}\n\n"
    "Provide your answer as exactly \"Yes\" or \"No\" inside <answer> tags. "
    "For example: <answer> Yes </answer>"
)

# ── Approach B (probability) for Baseline — uncomment to swap ────────────────
# BASELINE_TEMPLATE = (
#     "You are a prediction market forecaster. Given the following question about "
#     "a future event, estimate the probability (0-100) that the answer is Yes. "
#     "Think step by step, then provide your probability estimate.\n\n"
#     "Question: {question}\n\n"
#     "Provide your probability as a number (0-100) inside <answer> tags. "
#     "For example: <answer> 75 </answer>"
# )

# RAG: same model as baseline, but we inject retrieved context into the prompt.
RAG_TEMPLATE = (
    "You are a prediction market forecaster. You have access to the following "
    "recent search results relevant to the question. Use them to inform your "
    "prediction.\n\n"
    "Search Results:\n{context}\n\n"
    "Question: {question}\n\n"
    "Based on the search results and your own knowledge, predict whether the "
    "outcome will be Yes or No. Think step by step, then provide your final "
    "answer as exactly \"Yes\" or \"No\" inside <answer> tags. "
    "For example: <answer> Yes </answer>"
)

# ── Approach B (probability) for RAG — uncomment to swap ─────────────────────
# RAG_TEMPLATE = (
#     "You are a prediction market forecaster. You have access to the following "
#     "recent search results relevant to the question. Use them to inform your "
#     "prediction.\n\n"
#     "Search Results:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "Based on the search results and your own knowledge, estimate the "
#     "probability (0-100) that the answer is Yes. Think step by step, then "
#     "provide your probability as a number (0-100) inside <answer> tags. "
#     "For example: <answer> 75 </answer>"
# )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str):
    """
    Load model + tokenizer via HuggingFace transformers.

    For vLLM (faster batched inference), replace with:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_name, gpu_memory_utilization=0.8)
    and adapt generate_text() accordingly.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"📦  Loading model: {model_name}")
    print(f"    (First download may take a few minutes…)")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"    ✓ Loaded on {model.device}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT GENERATION (shared by all modes)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_text(model, tokenizer, prompt: str, stop_strings: list[str]) -> str:
    """
    Generate text, stopping at the first occurrence of any stop string.
    Returns generated text only (prompt excluded).
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3584).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Truncate at the earliest stop string
    earliest_pos = len(generated)
    for stop in stop_strings:
        idx = generated.find(stop)
        if idx != -1 and idx + len(stop) <= earliest_pos:
            earliest_pos = idx + len(stop)
    generated = generated[:earliest_pos]

    return generated


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def extract_between(text: str, start_tag: str, end_tag: str) -> str | None:
    """Extract text between the LAST occurrence of start_tag … end_tag."""
    start = text.rfind(start_tag)
    if start == -1:
        return None
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        return None
    return text[start:end].strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 1: BASELINE (no search)
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline(question: str, model, tokenizer) -> dict:
    """Vanilla LLM inference — no search, no retrieved context."""
    prompt = BASELINE_TEMPLATE.format(question=question)
    eos = tokenizer.eos_token or "<|endoftext|>"
    generated = generate_text(model, tokenizer, prompt, stop_strings=["</answer>", eos])

    answer = extract_between(generated, "<answer>", "</answer>")

    return {
        "prediction": answer,
        "full_trace": generated,
        "num_searches": 0,
        "search_queries": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 2: RAG (pre-retrieved context, vanilla model)
# ═══════════════════════════════════════════════════════════════════════════════

def run_rag(question: str, model, tokenizer, search_fn) -> dict:
    """
    Standard RAG: search ONCE using the question as query, stuff results
    into the prompt, then let the vanilla model reason over them.

    This isolates the value of *having* search results from the value of
    *learning to search interactively* (which is what Search-R1 adds).
    """
    # Single pre-retrieval using the question itself as the query
    print(f"    🔍 RAG pre-retrieval: \"{question[:80]}\"")
    context = search_fn(question)

    prompt = RAG_TEMPLATE.format(question=question, context=context)
    eos = tokenizer.eos_token or "<|endoftext|>"
    generated = generate_text(model, tokenizer, prompt, stop_strings=["</answer>", eos])

    answer = extract_between(generated, "<answer>", "</answer>")

    return {
        "prediction": answer,
        "full_trace": f"[RAG Context]\n{context}\n\n[Generation]\n{generated}",
        "num_searches": 1,
        "search_queries": [question],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE 3: SEARCH-R1 (interleaved search-and-reason)
# ═══════════════════════════════════════════════════════════════════════════════

def run_search_r1(question: str, model, tokenizer, search_fn) -> dict:
    """
    Search-R1's multi-turn inference loop (Algorithm 1 from the paper).
    The model autonomously decides when to search and what to search for.
    """
    prompt = SEARCH_R1_TEMPLATE.format(question=question)
    full_response = ""
    num_searches = 0
    search_queries = []
    eos = tokenizer.eos_token or "<|endoftext|>"

    for turn in range(MAX_SEARCH_TURNS + 1):
        current_input = prompt + full_response
        generated = generate_text(
            model, tokenizer, current_input,
            stop_strings=["</search>", "</answer>", eos],
        )
        full_response += generated

        if "</search>" in generated:
            query = extract_between(full_response, "<search>", "</search>")
            if query:
                search_queries.append(query)
                num_searches += 1
                print(f"    🔍 Search {num_searches}: \"{query[:80]}\"")

                results = search_fn(query)
                full_response += f"\n<information>{results}</information>\n"
            else:
                full_response += "\nMy action is not correct. Let me rethink.\n"

        elif "</answer>" in generated:
            break
        else:
            # Model stopped without valid action — force an answer
            full_response += "\n<answer>"
            forced = generate_text(
                model, tokenizer, prompt + full_response,
                stop_strings=["</answer>"],
            )
            full_response += forced + "</answer>"
            break

    answer = extract_between(full_response, "<answer>", "</answer>")

    return {
        "prediction": answer,
        "full_trace": full_response,
        "num_searches": num_searches,
        "search_queries": search_queries,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

RUNNERS = {
    "baseline":  run_baseline,
    "rag":       run_rag,
    "search-r1": run_search_r1,
}

def run_inference(mode: str, question: str, model, tokenizer, search_fn=None) -> dict:
    """Dispatch to the right inference function based on mode."""
    if mode == "baseline":
        return run_baseline(question, model, tokenizer)
    elif mode == "rag":
        return run_rag(question, model, tokenizer, search_fn)
    elif mode == "search-r1":
        return run_search_r1(question, model, tokenizer, search_fn)
    else:
        raise ValueError(f"Unknown mode: {mode}")