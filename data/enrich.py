"""
Enrichment pipeline: writes one "who is this perfect for?" persona per
product (not per colour/size variant) using Claude Haiku.
Input:  data/jb_grouped.csv  (from group.py)
Output: data/jb_enriched.csv (one row per product group)
"""

import os
import re
import time

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

PROMPT = """You are a gifting expert. Given a product, write 2-3 sentences describing WHO it is perfect for as a gift.

Focus on:
- Personality types, hobbies, and lifestyles it suits
- Age groups or life stages it fits
- Problems it solves or experiences it enables
- What kind of person would be THRILLED to receive this

Do NOT describe product features or specs. Describe the PERSON it suits.
Make it specific to THIS product: think about what sets it apart from similar products in the same category, so the description would not fit them just as well. Avoid catch-all audiences like "anyone", "busy professionals" or "students and parents alike".
Be specific and vivid. Write in present tense.

Product: {name}
Category: {category}
Price: {price}
Available as: {variants}

Who is this perfect for (2-3 sentences only):"""

MAX_VARIANTS_IN_PROMPT = 6


def summarise_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["price"])
    return (
        df.groupby("group_id")
        .agg(
            group_name=("group_name", "first"),
            category=("category", "first"),
            store=("store", "first"),
            price_min=("price", "min"),
            price_max=("price", "max"),
            n_variants=("name", "count"),
            variant_names=("name", list),
        )
        .reset_index()
    )


def format_price(lo: float, hi: float) -> str:
    return f"${lo:.0f}" if lo == hi else f"${lo:.0f} to ${hi:.0f}"


def enrich(row) -> str:
    names = [re.sub(r"\s+", " ", n) for n in row["variant_names"]]
    variants = "; ".join(names[:MAX_VARIANTS_IN_PROMPT])
    if len(names) > MAX_VARIANTS_IN_PROMPT:
        variants += f"; and {len(names) - MAX_VARIANTS_IN_PROMPT} more"
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": PROMPT.format(
                    name=row["group_name"],
                    category=row["category"],
                    price=format_price(row["price_min"], row["price_max"]),
                    variants=variants,
                ),
            }],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        print(f"  Error enriching '{row['group_name']}': {e}")
        return f"A {row['category'].replace('-', ' ')} gift: {row['group_name']}."


def main():
    groups = summarise_groups(pd.read_csv("data/jb_grouped.csv"))
    print(f"Enriching {len(groups)} products with Claude Haiku...\n")

    personas = []
    for _, row in tqdm(groups.iterrows(), total=len(groups), desc="Enriching"):
        personas.append(enrich(row))
        time.sleep(0.1)  # stay under rate limits

    groups["persona"] = personas
    groups.drop(columns=["variant_names"]).to_csv("data/jb_enriched.csv", index=False)
    print("\nSaved to data/jb_enriched.csv")

    print("\nExample enrichments:")
    for _, row in groups.head(3).iterrows():
        print(f"\n  Product: {row['group_name']} ({format_price(row['price_min'], row['price_max'])})")
        print(f"  Persona: {row['persona']}")


if __name__ == "__main__":
    main()
