"""
main.py
=======
Entry point for the Search-R1 Polymarket evaluation.

Runs one of three modes on the dataset and produces detailed logs + summary:
    python main.py --mode search-r1  --dataset polymarket_data/polymarket_dataset.jsonl --tavily-key tvly-XXX
    python main.py --mode rag        --dataset polymarket_data/polymarket_dataset.jsonl --tavily-key tvly-XXX
    python main.py --mode baseline   --dataset polymarket_data/polymarket_dataset.jsonl --tavily-key tvly-XXX

Modes:
    baseline   Vanilla Qwen2.5-3B-Instruct, no search (pure parametric knowledge)
    rag        Vanilla Qwen2.5-3B-Instruct + one-shot Tavily search per question
    search-r1  Search-R1-3B with interleaved multi-turn search (model decides when to search)
"""

import argparse
import json
import os
import time
from datetime import datetime

from search import TavilySearch, NullSearch
from inference import load_model, run_inference, MODELS
from evaluate import parse_prediction, compute_summary


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str, limit: int | None = None) -> list[dict]:
    """Load the JSONL dataset produced by scrape_polymarket.py."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    print(f"📂  Loaded {len(rows)} markets from {path}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM prediction accuracy on resolved Polymarket events"
    )
    parser.add_argument(
        "--mode", required=True, choices=["baseline", "rag", "search-r1"],
        help="baseline = no search | rag = pre-retrieved context | search-r1 = interleaved search"
    )
    parser.add_argument("--dataset", required=True, help="Path to polymarket_dataset.jsonl")
    parser.add_argument("--tavily-key", default=None, help="Tavily API key (required for rag and search-r1)")
    parser.add_argument("--model", default=None, help="Override the HuggingFace model ID")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N markets")
    parser.add_argument("--out-dir", default="./eval_results", help="Output directory")
    args = parser.parse_args()

    mode = args.mode

    # ── Resolve model ────────────────────────────────────────────────────────
    model_name = args.model or MODELS[mode]

    # ── Setup search backend ─────────────────────────────────────────────────
    if mode in ("rag", "search-r1"):
        if not args.tavily_key and not os.environ.get("TAVILY_API_KEY"):
            parser.error(f"--tavily-key is required for mode '{mode}'")
        search_backend = TavilySearch(api_key=args.tavily_key)
    else:
        search_backend = NullSearch()

    # ── Load data + model ────────────────────────────────────────────────────
    dataset = load_dataset(args.dataset, limit=args.limit)
    model, tokenizer = load_model(model_name)

    # ── Prepare output ───────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, f"log_{mode}_{timestamp}.jsonl")
    log_file = open(log_path, "w", encoding="utf-8")

    print(f"\n{'═'*70}")
    print(f"  Mode:    {mode}")
    print(f"  Model:   {model_name}")
    print(f"  Markets: {len(dataset)}")
    print(f"  Log:     {log_path}")
    print(f"{'═'*70}\n")

    # ── Run evaluation ───────────────────────────────────────────────────────
    results = []

    for i, market in enumerate(dataset):
        question = market["question"]
        gt       = market["ground_truth"]           # 1 = Yes won, 0 = No won
        gt_label = market["ground_truth_label"]

        print(f"[{i+1}/{len(dataset)}] {question[:80]}…")
        print(f"    Ground truth: {gt_label} ({gt})")

        t0 = time.time()

        # Run inference
        result = run_inference(
            mode=mode,
            question=question,
            model=model,
            tokenizer=tokenizer,
            search_fn=search_backend.search if mode != "baseline" else None,
        )

        elapsed = time.time() - t0

        # Parse + evaluate
        eval_result = parse_prediction(result["prediction"], gt)

        print(f"    Prediction:   {eval_result['parsed']}")
        print(f"    Correct:      {'✓' if eval_result['correct'] else '✗'}")
        print(f"    Brier:        {eval_result['brier_score']:.4f}")
        if result["num_searches"] > 0:
            print(f"    Searches:     {result['num_searches']}")
        print(f"    Time:         {elapsed:.1f}s\n")

        # Build log entry
        entry = {
            "market_id":         market["market_id"],
            "question":          question,
            "ground_truth":      gt,
            "ground_truth_label": gt_label,
            "prediction_raw":    result["prediction"],
            "predicted_prob":    eval_result["predicted_prob"],
            "brier_score":       eval_result["brier_score"],
            "correct":           eval_result["correct"],
            "parsed":            eval_result["parsed"],
            "num_searches":      result["num_searches"],
            "search_queries":    result["search_queries"],
            "full_trace":        result["full_trace"][:5000],
            "elapsed_seconds":   round(elapsed, 1),
        }
        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log_file.flush()
        results.append(entry)

    log_file.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = compute_summary(results, mode, model_name)
    summary["tavily_calls_used"] = search_backend.total_calls

    summary_path = os.path.join(args.out_dir, f"summary_{mode}_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'═'*70}")
    print(f"  RESULTS — {mode}")
    print(f"{'═'*70}")
    print(f"  Markets evaluated:       {summary['num_markets']}")
    print(f"  Accuracy:                {summary['accuracy']:.1%}  "
          f"({int(summary['accuracy'] * summary['num_markets'])}/{summary['num_markets']})")
    print(f"  Avg Brier Score:         {summary['avg_brier_score']:.4f}")
    print(f"  Accuracy (Yes markets):  {summary['accuracy_on_yes']:.1%}  "
          f"(n={summary['num_yes_markets']})")
    print(f"  Accuracy (No markets):   {summary['accuracy_on_no']:.1%}  "
          f"(n={summary['num_no_markets']})")
    print(f"  Avg searches/market:     {summary['avg_searches_per_market']}")
    print(f"  Parse failures:          {summary['parse_failures']}")
    if search_backend.total_calls:
        print(f"  Tavily API calls used:   {search_backend.total_calls}")
    print(f"\n  Log:     {log_path}")
    print(f"  Summary: {summary_path}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()