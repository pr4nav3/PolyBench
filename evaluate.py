"""
evaluate.py
===========
Prediction parsing and metric computation.

Supports two approaches:
  A (default): Binary Yes/No → accuracy + Brier score
  B (commented): Probability 0-100 → calibrated Brier score
"""

from __future__ import annotations
import re


# ═══════════════════════════════════════════════════════════════════════════════
#  APPROACH A: Binary Yes/No parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_prediction(prediction: str | None, ground_truth: int) -> dict:
    """
    Parse a Yes/No prediction into evaluation metrics.

    Args:
        prediction:   raw string from <answer> tags (may be None)
        ground_truth: 1 if outcomes[0] (usually "Yes") won, 0 otherwise

    Returns:
        dict with predicted_prob, brier_score, correct, parsed
    """
    if prediction is None:
        return {
            "predicted_prob": 0.5,
            "brier_score": 0.25,
            "correct": False,
            "parsed": "PARSE_FAILED",
        }

    clean = prediction.strip().lower().rstrip(".")

    # Direct match (the model gave just "Yes" or "No")
    if clean in ("yes", '"yes"', "'yes'"):
        prob = 1.0
    elif clean in ("no", '"no"', "'no'"):
        prob = 0.0
    else:
        # Fuzzy: use word-boundary regex to avoid matching 'no' inside
        # 'know', 'cannot', 'innovation', etc.
        has_yes = bool(re.search(r"\byes\b", clean))
        has_no  = bool(re.search(r"\bno\b", clean))

        if has_yes and not has_no:
            prob = 1.0
        elif has_no and not has_yes:
            prob = 0.0
        else:
            # Both present, or neither — ambiguous, default to 50/50
            prob = 0.5

    brier = (prob - ground_truth) ** 2
    correct = (prob >= 0.5) == (ground_truth == 1)

    return {
        "predicted_prob": prob,
        "brier_score": brier,
        "correct": correct,
        "parsed": prediction.strip(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  APPROACH B: Probability parsing (uncomment to use)
# ═══════════════════════════════════════════════════════════════════════════════

# def parse_prediction(prediction: str | None, ground_truth: int) -> dict:
#     """
#     Parse a probability (0-100) from the model's answer.
#     Replace the function above with this one for calibrated Brier scores.
#     """
#     if prediction is None:
#         return {
#             "predicted_prob": 0.5,
#             "brier_score": 0.25,
#             "correct": False,
#             "parsed": "PARSE_FAILED",
#         }
#
#     numbers = re.findall(r"(\d+(?:\.\d+)?)", prediction.strip())
#     if not numbers:
#         return {
#             "predicted_prob": 0.5,
#             "brier_score": 0.25,
#             "correct": False,
#             "parsed": prediction.strip(),
#         }
#
#     prob = float(numbers[0])
#     if prob > 1.0:
#         prob /= 100.0
#     prob = max(0.0, min(1.0, prob))
#
#     brier = (prob - ground_truth) ** 2
#     correct = (prob >= 0.5) == (ground_truth == 1)
#
#     return {
#         "predicted_prob": prob,
#         "brier_score": brier,
#         "correct": correct,
#         "parsed": prediction.strip(),
#     }


# ═══════════════════════════════════════════════════════════════════════════════
#  AGGREGATE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summary(results: list[dict], mode: str, model_name: str) -> dict:
    """
    Compute aggregate metrics from a list of per-market result dicts.
    Each result dict must have: correct, brier_score, num_searches.
    """
    n = len(results)
    if n == 0:
        return {"error": "no results"}

    total_correct  = sum(r["correct"] for r in results)
    total_brier    = sum(r["brier_score"] for r in results)
    total_searches = sum(r["num_searches"] for r in results)

    # Breakdown by ground truth label
    yes_results = [r for r in results if r["ground_truth"] == 1]
    no_results  = [r for r in results if r["ground_truth"] == 0]

    def safe_acc(subset):
        return round(sum(r["correct"] for r in subset) / max(len(subset), 1), 4)

    return {
        "mode":                    mode,
        "model":                   model_name,
        "num_markets":             n,
        "accuracy":                round(total_correct / n, 4),
        "avg_brier_score":         round(total_brier / n, 4),
        "accuracy_on_yes":         safe_acc(yes_results),
        "accuracy_on_no":          safe_acc(no_results),
        "num_yes_markets":         len(yes_results),
        "num_no_markets":          len(no_results),
        "total_searches":          total_searches,
        "avg_searches_per_market": round(total_searches / n, 2),
        "parse_failures":          sum(1 for r in results if r.get("parsed") == "PARSE_FAILED"),
    }