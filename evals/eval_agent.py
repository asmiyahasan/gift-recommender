"""
Agent eval: does the full agent follow the rules?

Runs the real agent (Claude + tools) on every test persona in
evals/cases.json, parses the 3 recommendations out of its reply, and checks
them against the product database.

Hard checks (a case passes only if all pass):
  format       - 3 numbered picks in the "**1. Name — $Price**" format
  three_picks  - exactly 3 different products, each with a product URL
  real         - every URL exists in the catalogue (no made-up products)
  prices       - the price the agent states matches the catalogue
  budget       - every pick within budget; one pick up to 10% over is
                 allowed as a "stretch", as the system prompt permits
  exclusions   - nothing matching the case's exclude_keywords
  searched     - the agent called semantic_search at least once

Also reported, not pass/fail: how many picks match the case's "relevant"
patterns, tokens used, estimated cost, and time taken.

This calls the Anthropic API (roughly 1-3 cents per case with Sonnet).

Usage (from anywhere):
  python evals/eval_agent.py                  # all cases
  python evals/eval_agent.py --limit 3        # first 3 cases, for a quick check
  python evals/eval_agent.py --case runner --verbose
"""

import argparse
import re
import sqlite3
import time
import traceback

from common import is_excluded, is_relevant, load_cases, pct, save_results
from agent.agent import agent

STRETCH = 0.10                           # allowed overshoot for one "stretch" pick
PRICE_PER_MTOK = {"input": 3.0, "output": 15.0}   # Claude Sonnet, USD per million tokens
CHECKS = ["format", "three_picks", "real", "prices", "budget", "exclusions", "searched"]

HEADER_RE = re.compile(
    r"\*\*\s*(\d)\.\s*(?P<name>.+?)\s*[—–-]+\s*\$\s*(?P<price>[\d,]+(?:\.\d+)?)\s*\*\*"
)
URL_RE = re.compile(r"https?://(?:www\.)?jbhifi\.com\.au/products/[\w\-%.]+")


def load_catalogue():
    conn = sqlite3.connect("data/products.db")
    rows = conn.execute("SELECT url, name, price, category FROM products").fetchall()
    conn.close()
    return {norm_url(u): {"name": n, "price": p, "category": c} for u, n, p, c in rows}


def norm_url(url):
    return url.split("?")[0].split("#")[0].rstrip("/.)").replace("://jbhifi", "://www.jbhifi")


def parse_picks(text):
    """Return [{stated_name, stated_price, url}] for each numbered pick."""
    headers = list(HEADER_RE.finditer(text))
    picks = []
    for i, h in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        url = URL_RE.search(text, h.end(), end)
        picks.append({
            "stated_name": h.group("name").strip(),
            "stated_price": float(h.group("price").replace(",", "")),
            "url": norm_url(url.group(0)) if url else None,
        })
    return picks


def check_budget(prices, case):
    lo, hi = case["min_price"], case["max_price"]
    under = [p for p in prices if p < lo]
    over = [p for p in prices if p > hi]
    if not under and not over:
        return True, "ok"
    if not under and len(over) == 1 and over[0] <= hi * (1 + STRETCH):
        return True, "stretch"
    return False, f"{len(under)} under, {len(over)} over"


def run_case(case, catalogue):
    start = time.time()
    result = agent.invoke({"messages": [{"role": "user", "content": case["prompt"]}]})
    elapsed = time.time() - start

    messages = result["messages"]
    reply = messages[-1].content
    if isinstance(reply, list):  # content blocks
        reply = "".join(b.get("text", "") for b in reply if isinstance(b, dict))

    tool_calls = [c["name"] for m in messages for c in (getattr(m, "tool_calls", None) or [])]
    tokens = {"input": 0, "output": 0}
    for m in messages:
        usage = getattr(m, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0)
        tokens["output"] += usage.get("output_tokens", 0)
    cost = sum(tokens[k] * PRICE_PER_MTOK[k] for k in tokens) / 1e6

    picks = parse_picks(reply)
    for p in picks:
        p["catalogue"] = catalogue.get(p["url"]) if p["url"] else None

    urls = [p["url"] for p in picks if p["url"]]
    found = [p for p in picks if p["catalogue"]]
    real_prices = [p["catalogue"]["price"] for p in found]
    budget_ok, budget_note = check_budget(real_prices, case)

    checks = {
        "format": len(picks) == 3,
        "three_picks": len(picks) == 3 and len(urls) == 3 and len(set(urls)) == 3,
        "real": bool(picks) and len(found) == len(picks),
        "prices": bool(found) and all(abs(p["stated_price"] - p["catalogue"]["price"]) < 0.5 for p in found),
        "budget": bool(found) and budget_ok,
        "exclusions": not any(is_excluded(p["catalogue"]["name"], case) for p in found),
        "searched": "semantic_search" in tool_calls,
    }

    return {
        "id": case["id"],
        "passed": all(checks.values()),
        "checks": checks,
        "budget_note": budget_note,
        "relevant_picks": sum(is_relevant(p["catalogue"]["name"], case) for p in found)
        if case.get("relevant") else None,
        "picks": picks,
        "tool_calls": tool_calls,
        "tokens": tokens,
        "cost_usd": round(cost, 4),
        "seconds": round(elapsed, 1),
        "note": case.get("note"),
        "reply": reply,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", action="append", help="only run this case id (repeatable)")
    ap.add_argument("--limit", type=int, help="only run the first N cases")
    ap.add_argument("--verbose", action="store_true", help="print each reply and its parsed picks")
    args = ap.parse_args()

    cases = load_cases(args.case)[: args.limit]
    catalogue = load_catalogue()

    rows = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} ...", end=" ", flush=True)
        try:
            r = run_case(case, catalogue)
        except Exception as e:  # keep going; one API hiccup shouldn't sink the run
            r = {"id": case["id"], "passed": False, "error": f"{type(e).__name__}: {e}",
                 "traceback": traceback.format_exc(),
                 "checks": {c: False for c in CHECKS}, "cost_usd": 0, "seconds": 0, "note": case.get("note")}
        rows.append(r)
        print("error" if "error" in r else ("pass" if r["passed"] else "FAIL"))
        if args.verbose and "reply" in r:
            print(r["reply"], "\n")
            for p in r["picks"]:
                cat = p["catalogue"]
                print(f"  -> {p['stated_name'][:50]} ${p['stated_price']} | "
                      f"{'catalogue $' + str(cat['price']) if cat else 'NOT IN CATALOGUE'}")
            print()

    n = len(rows)
    print(f"\n{'case':26} " + " ".join(f"{c[:6]:>6}" for c in CHECKS) + "  relevant   cost")
    print("-" * 100)
    for r in rows:
        marks = " ".join(f"{'✓' if r['checks'][c] else '✗':>6}" for c in CHECKS)
        rel = r.get("relevant_picks")
        rel = "-" if rel is None else f"{rel}/3"
        extra = f"  ({r['budget_note']})" if r.get("budget_note") not in (None, "ok") else ""
        extra += f"  ERROR {r['error'][:40]}" if "error" in r else ""
        print(f"{r['id']:26} {marks}  {rel:>8}  ${r['cost_usd']:.3f}{extra}")
    print("-" * 100)

    summary = {
        "cases": n,
        "passed": sum(r["passed"] for r in rows),
        "pass_rate": sum(r["passed"] for r in rows) / n,
        "check_rates": {c: sum(r["checks"][c] for r in rows) / n for c in CHECKS},
        "relevant_pick_rate": (lambda rs: sum(rs) / (3 * len(rs)) if rs else None)(
            [r["relevant_picks"] for r in rows if r.get("relevant_picks") is not None]),
        "errors": sum("error" in r for r in rows),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 3),
        "avg_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
    }
    print(f"Passed {summary['passed']}/{n} ({pct(summary['pass_rate']).strip()})   " +
          "   ".join(f"{c} {pct(v).strip()}" for c, v in summary["check_rates"].items()))
    if summary["relevant_pick_rate"] is not None:
        print(f"Picks matching the case's relevant products: {pct(summary['relevant_pick_rate']).strip()}")
    print(f"Cost ~${summary['total_cost_usd']}   avg {summary['avg_seconds']}s per case")

    for r in rows:
        if r.get("note"):
            print(f"\nReview by hand — {r['id']}: {r['note']}")

    path = save_results("agent", {"summary": summary, "cases": rows})
    print(f"\nSaved {path.relative_to(path.parents[2])} (includes every reply)")


if __name__ == "__main__":
    main()
