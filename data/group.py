"""
Grouping step: works out which listings are variants of the same product
(colours, storage sizes, refurbished grades, etc.) using Claude Haiku.

Input:  data/jb_raw.csv
Output: data/jb_grouped.csv  (one row per listing, plus group_id and group_name)

Run this after scraping and before enrich.py. Check the printed summary:
anything grouped wrongly can be fixed by editing group_name in the CSV and
re-running enrich.py and index.py.
"""

import json
import os
import re

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

PROMPT = """You are cleaning up a product catalogue. Below is a numbered list of listings from the "{category}" category. Work out which listings are variants of the same product.

Variants of ONE product (same group):
- different colours or finishes
- different storage sizes (64GB vs 256GB)
- refurbished condition grades (As New, Excellent, Very Good)
- Wi-Fi vs Wi-Fi + Cellular
- case, watch or screen sizes of the same model (40mm vs 44mm, 11-inch vs 13-inch)
- pack sizes (2 Pack vs 4 Pack) and bundles of the same item (Standard Combo vs Creator Combo)

DIFFERENT products (separate groups):
- different generations or model years (iPad 9th Gen vs iPad 10th Gen, Gen 1 vs Gen 2)
- different models in a range (DualSense vs DualSense Edge, Apple Watch SE vs Series 11, Buds3 Pro vs Buds4 Pro)
- different product lines from the same brand

For each listing give a group name: the product name without colour, storage, size, condition, connectivity, pack or bundle details. For example "Apple iPad 9th Gen" or "PS5 DualSense Wireless Controller". Listings in the same group must have exactly the same group name, spelled identically.

Reply with only a JSON array, one object per listing, in the same order:
[{{"i": 0, "group": "..."}}, {{"i": 1, "group": "..."}}, ...]

Listings:
{listings}"""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def ask_haiku(category: str, names: list[str]) -> dict[int, str]:
    clean = [re.sub(r"\s+", " ", n) for n in names]  # some names contain non-breaking spaces
    listings = "\n".join(f"{i}. {n}" for i, n in enumerate(clean))
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=8000,
        temperature=0,
        messages=[{"role": "user", "content": PROMPT.format(category=category, listings=listings)}],
    )
    text = msg.content[0].text
    start, end = text.find("["), text.rfind("]")
    items = json.loads(text[start : end + 1])
    return {int(x["i"]): str(x["group"]).strip() for x in items if str(x.get("group", "")).strip()}


def group_category(category: str, names: list[str]) -> list[str]:
    groups = {}
    for attempt in range(2):
        try:
            groups = ask_haiku(category, names)
            break
        except Exception as e:  # bad JSON or API error: try once more
            print(f"  Attempt {attempt + 1} failed for {category}: {e}")
    missing = [i for i in range(len(names)) if i not in groups]
    if missing:
        print(f"  {len(missing)} listing(s) in {category} weren't grouped; keeping them as their own product")
    # Anything Haiku didn't return stays as its own product, under its full name.
    return [groups.get(i, names[i]) for i in range(len(names))]


def main():
    df = pd.read_csv("data/jb_raw.csv")
    print(f"Grouping {len(df)} listings with Claude Haiku...\n")

    df["group_name"] = ""
    for category, idx in df.groupby("category").groups.items():
        names = df.loc[idx, "name"].tolist()
        print(f"  {category}: {len(names)} listings")
        df.loc[idx, "group_name"] = group_category(category, names)

    df["group_id"] = df["group_name"].map(slugify)
    # Use one spelling per group_id, in case Haiku varied punctuation or case.
    df["group_name"] = df.groupby("group_id")["group_name"].transform("first")

    df.to_csv("data/jb_grouped.csv", index=False)

    sizes = df.groupby("group_id").size()
    print(f"\n{len(df)} listings -> {len(sizes)} products ({(sizes > 1).sum()} with variants)")
    print("Saved to data/jb_grouped.csv\n")
    print("Products with variants (check these look right):")
    for gid in sizes[sizes > 1].sort_values(ascending=False).index:
        g = df[df["group_id"] == gid]
        print(f"\n  {g['group_name'].iloc[0]}  (${g['price'].min():.0f}-{g['price'].max():.0f})")
        for name in g["name"]:
            print(f"    - {name}")


if __name__ == "__main__":
    main()
