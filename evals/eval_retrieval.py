"""
Retrieval eval: how good is semantic search on its own?

For each test persona in evals/cases.json, this queries the ChromaDB index
directly (no LLM, no API calls) and checks whether products matching the
case's "relevant" patterns come back near the top. Each result is a product
group; it counts as relevant if any of its variants' names match.

Metrics (over the top K results, ignoring budget):
  hit@3      - at least one relevant product in the top 3
  hit@K      - at least one relevant product in the top K
  precision  - share of the top K that are relevant
  MRR        - 1 / rank of the first relevant product (0 if none in top K)

It also runs the agent's actual semantic_search tool with each case's budget
and reports how many products it returns, how many of those are relevant
(counting only in-budget, non-excluded variants), and any variant it returns
that is outside the budget.

Usage (from anywhere):
  python evals/eval_retrieval.py
  python evals/eval_retrieval.py --k 5 --verbose
  python evals/eval_retrieval.py --case runner --case sim_racer
"""

import argparse
import json

from common import is_excluded, is_relevant, load_cases, load_variant_names, pct, save_results
from agent.tools import get_collection, semantic_search

TOOL_N_RESULTS = 8  # the tool's default, which is what the agent normally gets


def rank_case(collection, case, k, variant_names):
    res = collection.query(
        query_texts=[case["prompt"]],
        n_results=k,
        include=["metadatas", "distances"],
    )
    metas = res["metadatas"][0]
    ranked = [m.get("name", "") for m in metas]
    flags = [
        any(is_relevant(n, case) for n in variant_names.get(m.get("group_id"), [m.get("name", "")]))
        for m in metas
    ]
    first = next((i for i, f in enumerate(flags) if f), None)
    return {
        "top": [{"name": n, "relevant": f} for n, f in zip(ranked, flags)],
        "hit@3": any(flags[:3]),
        "hit@k": any(flags),
        "precision": sum(flags) / len(flags) if flags else 0.0,
        "mrr": 1 / (first + 1) if first is not None else 0.0,
    }


def budget_case(case):
    raw = semantic_search.invoke({
        "query": case["prompt"],
        "min_price": case["min_price"],
        "max_price": case["max_price"],
        "n_results": TOOL_N_RESULTS,
    })
    products = json.loads(raw)
    variants = [v for p in products for v in p["variants"]]
    over = [v for v in variants if not case["min_price"] <= v["price"] <= case["max_price"]]
    return {
        "returned": len(products),
        "relevant_in_budget": sum(
            any(is_relevant(v["name"], case) and not is_excluded(v["name"], case) for v in p["variants"])
            for p in products
        ),
        "budget_violations": len(over),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=10, help="how many top results to score (default 10)")
    ap.add_argument("--case", action="append", help="only run this case id (repeatable)")
    ap.add_argument("--verbose", action="store_true", help="print the ranked results for each case")
    args = ap.parse_args()

    cases = [c for c in load_cases(args.case) if c.get("relevant")]
    collection = get_collection()
    variant_names = load_variant_names()

    rows = []
    for case in cases:
        r = {"id": case["id"], **rank_case(collection, case, args.k, variant_names), "tool": budget_case(case)}
        rows.append(r)
        if args.verbose:
            print(f"\n{case['id']}: {case['prompt']}")
            for i, t in enumerate(r["top"], 1):
                print(f"  {i:2}. {'✓' if t['relevant'] else ' '} {t['name']}")

    n = len(rows)
    summary = {
        "cases": n,
        "k": args.k,
        "hit@3": sum(r["hit@3"] for r in rows) / n,
        "hit@k": sum(r["hit@k"] for r in rows) / n,
        "precision": sum(r["precision"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "tool_short_results": sum(r["tool"]["returned"] < TOOL_N_RESULTS for r in rows),
        "tool_no_relevant_in_budget": sum(r["tool"]["relevant_in_budget"] == 0 for r in rows),
        "tool_budget_violations": sum(r["tool"]["budget_violations"] for r in rows),
    }

    k = args.k
    print(f"\n{'case':26} {'hit@3':>5} {f'hit@{k}':>6} {f'P@{k}':>6} {'MRR':>5}   tool: returned / relevant")
    print("-" * 82)
    for r in rows:
        t = r["tool"]
        print(
            f"{r['id']:26} {'✓' if r['hit@3'] else '✗':>5} {'✓' if r['hit@k'] else '✗':>6} "
            f"{r['precision']:6.2f} {r['mrr']:5.2f}   {t['returned']:>8} / {t['relevant_in_budget']}"
        )
    print("-" * 82)
    print(f"hit@3 {pct(summary['hit@3'])}   hit@{k} {pct(summary['hit@k'])}   "
          f"precision@{k} {pct(summary['precision'])}   MRR {summary['mrr']:.2f}   ({n} cases)")
    print(f"semantic_search tool: {summary['tool_short_results']} case(s) returned fewer than "
          f"{TOOL_N_RESULTS} results after the price filter, {summary['tool_no_relevant_in_budget']} "
          f"had no relevant in-budget product, {summary['tool_budget_violations']} budget violation(s)")

    path = save_results("retrieval", {"summary": summary, "cases": rows})
    print(f"\nSaved {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
