"""
scrape_polymarket.py
====================
Scrape resolved binary markets from Polymarket's Gamma API and store
as a clean JSONL + CSV dataset for Search-R1 evaluation.

Filters:
  - closed=true (resolved markets only)
  - Binary outcomes (exactly 2 outcomes)
  - Clearly resolved: winning outcome price >= 0.95
  - Clean "Will …?" question format (natural language, not sports spreads)
  - Volume > $10,000 (meaningful liquidity)
  - Resolved AFTER 2025-01-01 (safely past Qwen-2.5's knowledge cutoff)

Ground truth derivation:
  When a Polymarket market resolves, the winning outcome token → $1.00
  and the losing one → $0.00.  For closed markets the `outcomePrices`
  field holds the final (post-resolution) prices.  We treat the outcome
  whose final price >= 0.95 as the winner.

Usage:
    python scrape_polymarket.py                 # defaults: 200 target rows
    python scrape_polymarket.py --target 300    # collect 300 rows
    python scrape_polymarket.py --min-volume 50000
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE = "https://gamma-api.polymarket.com/markets"
PAGE_SIZE = 100          # max the API tends to return per page
RATE_LIMIT_SLEEP = 1.0   # seconds between paginated requests

# Qwen-2.5 knowledge cutoff context:
#   - Official sources list "end of 2023" to "September 2024" depending
#     on the variant.  Some user reports suggest June 2024.
#   - To be safe we only keep markets that resolved AFTER 2025-01-01,
#     giving a comfortable buffer past any plausible training data.
CUTOFF_DATE = "2025-01-01T00:00:00Z"

# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_json(url: str, max_retries: int = 3) -> list | dict:
    """GET a URL and return parsed JSON with simple retry logic."""
    for attempt in range(max_retries):
        try:
            req = Request(url, headers={"Accept": "application/json",
                                        "User-Agent": "SearchR1-DataCollector/1.0"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            if e.code == 429:  # rate-limited
                wait = 2 ** (attempt + 1)
                print(f"  ⚠ Rate-limited (429). Waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"  ✗ HTTP {e.code} on attempt {attempt+1}/{max_retries}")
                time.sleep(RATE_LIMIT_SLEEP)
        except (URLError, TimeoutError) as e:
            print(f"  ✗ Network error: {e}  (attempt {attempt+1}/{max_retries})")
            time.sleep(RATE_LIMIT_SLEEP)
    print(f"  ✗ Failed after {max_retries} retries for: {url[:120]}…")
    return []


def is_clean_question(q: str) -> bool:
    """
    Return True if the question looks like a natural-language prediction
    question rather than a sports spread / over-under / jargon market.

    Positive signals:  starts with Will/Who/What/Is/Does/Did/Has/Can,
                       or contains a '?'
    Negative signals:  contains O/U, spread notation, point-spread
                       patterns like '+4.5', moneyline jargon.
    """
    q_stripped = q.strip()

    # Must contain a question mark
    if "?" not in q_stripped:
        return False

    # Reject sports-spread / over-under jargon
    spread_patterns = [
        r"\bO/U\b",            # Over/Under
        r"\bover/under\b",
        r"[+-]\d+\.5",         # point spreads like +4.5, -3.5
        r"\bspread\b",
        r"\bmoneyline\b",
        r"\btotal points\b",
        r"\bML\b",
    ]
    for pat in spread_patterns:
        if re.search(pat, q_stripped, re.IGNORECASE):
            return False

    return True


def extract_ground_truth(outcomes_str: str, prices_str: str):
    """
    Parse outcome labels and final prices.
    Return (ground_truth_binary, ground_truth_label, prices_list)
    or None if the market isn't cleanly resolved.

    ground_truth_binary = 1 if outcomes[0] won, else 0
    """
    try:
        outcomes = json.loads(outcomes_str)
        prices = json.loads(prices_str)
    except (json.JSONDecodeError, TypeError):
        return None

    if len(outcomes) != 2 or len(prices) != 2:
        return None

    p0 = float(prices[0])
    p1 = float(prices[1])

    # Check one side clearly won (>= 0.95)
    if p0 >= 0.95 and p1 < 0.5:
        return (1, outcomes[0], [p0, p1])
    elif p1 >= 0.95 and p0 < 0.5:
        return (0, outcomes[1], [p0, p1])
    else:
        return None  # ambiguous / 50-50 / not fully resolved


def passes_date_filter(market: dict, cutoff_iso: str) -> bool:
    """Return True if the market's end date (resolution) is after the cutoff."""
    end_str = market.get("endDate") or market.get("endDateIso")
    if not end_str:
        return False
    try:
        # Handle both ISO datetime and date-only formats
        if "T" in end_str:
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        else:
            end_dt = datetime.fromisoformat(end_str + "T00:00:00+00:00")
        cutoff_dt = datetime.fromisoformat(cutoff_iso.replace("Z", "+00:00"))
        return end_dt >= cutoff_dt
    except ValueError:
        return False


# ── Main collection loop ─────────────────────────────────────────────────────

def collect_markets(target: int, min_volume: float) -> list[dict]:
    """
    Paginate through the Gamma API, filter, and return clean rows.
    """
    collected = []
    seen_ids = set()
    offset = 0
    pages_fetched = 0
    total_scanned = 0

    print(f"🎯  Target: {target} clean markets")
    print(f"💰  Min volume: ${min_volume:,.0f}")
    print(f"📅  Resolution after: {CUTOFF_DATE}")
    print(f"{'─'*60}")

    while len(collected) < target:
        params = urlencode({
            "closed": "true",
            "limit": PAGE_SIZE,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        })
        url = f"{API_BASE}?{params}"

        print(f"\n📡  Fetching page {pages_fetched+1}  (offset={offset}) …")
        markets = fetch_json(url)

        if not markets:
            print("  ℹ No more results. Stopping pagination.")
            break

        pages_fetched += 1
        total_scanned += len(markets)

        for m in markets:
            mid = m.get("id")
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            # ── Filter pipeline ──────────────────────────────────────
            # 1. Must be closed
            if not m.get("closed"):
                continue

            # 2. Volume threshold
            vol = float(m.get("volume") or m.get("volumeNum") or 0)
            if vol < min_volume:
                continue

            # 3. Date filter (resolved after Qwen-2.5 cutoff)
            if not passes_date_filter(m, CUTOFF_DATE):
                continue

            # 4. Clean natural-language question
            question = m.get("question", "")
            if not is_clean_question(question):
                continue

            # 5. Binary + clearly resolved ground truth
            gt = extract_ground_truth(
                m.get("outcomes", "[]"),
                m.get("outcomePrices", "[]"),
            )
            if gt is None:
                continue

            gt_binary, gt_label, final_prices = gt

            # ── Build clean row ──────────────────────────────────────
            row = {
                "market_id":          mid,
                "question":           question,
                "description":        (m.get("description") or "")[:2000],  # cap length
                "outcomes":           json.loads(m["outcomes"]),
                "end_date":           m.get("endDate", ""),
                "volume":             round(vol, 2),
                "ground_truth":       gt_binary,
                "ground_truth_label": gt_label,
                "final_prices":       final_prices,
                "slug":               m.get("slug", ""),
            }
            collected.append(row)

            if len(collected) % 25 == 0 or len(collected) == target:
                print(f"  ✓ Collected {len(collected)}/{target}")

            if len(collected) >= target:
                break

        offset += PAGE_SIZE
        time.sleep(RATE_LIMIT_SLEEP)

        # Safety valve: don't paginate forever
        if pages_fetched > 100:
            print("  ⚠ Hit 100-page safety limit. Stopping.")
            break

    print(f"\n{'─'*60}")
    print(f"✅  Done. Scanned {total_scanned} markets → kept {len(collected)}")
    return collected


# ── Output ───────────────────────────────────────────────────────────────────

def save_dataset(rows: list[dict], out_dir: str):
    """Save as both JSONL and CSV."""
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "polymarket_dataset.jsonl")
    csv_path   = os.path.join(out_dir, "polymarket_dataset.csv")

    # ── JSONL ────────────────────────────────────────────────────────
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"📄  JSONL → {jsonl_path}")

    # ── CSV ──────────────────────────────────────────────────────────
    csv_fields = [
        "market_id", "question", "description", "outcomes",
        "end_date", "volume", "ground_truth", "ground_truth_label",
        "final_prices", "slug",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            # Serialize lists to JSON strings for CSV
            csv_row["outcomes"]     = json.dumps(csv_row["outcomes"])
            csv_row["final_prices"] = json.dumps(csv_row["final_prices"])
            writer.writerow(csv_row)
    print(f"📊  CSV  → {csv_path}")

    # ── Summary stats ────────────────────────────────────────────────
    n = len(rows)
    yes_count = sum(1 for r in rows if r["ground_truth"] == 1)
    no_count  = n - yes_count
    volumes   = [r["volume"] for r in rows]
    print(f"\n📈  Dataset summary:")
    print(f"    Total markets:  {n}")
    print(f"    Yes outcomes:   {yes_count}  ({100*yes_count/n:.1f}%)")
    print(f"    No  outcomes:   {no_count}  ({100*no_count/n:.1f}%)")
    print(f"    Volume range:   ${min(volumes):,.0f} – ${max(volumes):,.0f}")
    print(f"    Median volume:  ${sorted(volumes)[n//2]:,.0f}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape resolved Polymarket markets for Search-R1 evaluation"
    )
    parser.add_argument(
        "--target", type=int, default=200,
        help="Target number of markets to collect (default: 200)"
    )
    parser.add_argument(
        "--min-volume", type=float, default=10_000,
        help="Minimum trading volume in USD (default: 10000)"
    )
    parser.add_argument(
        "--out-dir", type=str, default="./polymarket_data",
        help="Output directory (default: ./polymarket_data)"
    )
    parser.add_argument(
        "--cutoff-date", type=str, default=CUTOFF_DATE,
        help=f"Only keep markets resolved after this ISO date (default: {CUTOFF_DATE})"
    )
    args = parser.parse_args()

    # Update module-level cutoff date if user overrode it
    import scrape_polymarket
    scrape_polymarket.CUTOFF_DATE = args.cutoff_date

    rows = collect_markets(target=args.target, min_volume=args.min_volume)

    if not rows:
        print("❌  No markets matched the filters. Try lowering --min-volume or adjusting --cutoff-date.")
        sys.exit(1)

    save_dataset(rows, args.out_dir)


if __name__ == "__main__":
    main()