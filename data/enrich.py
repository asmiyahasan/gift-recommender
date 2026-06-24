"""
Enrichment pipeline: rewrites raw product descriptions into
"who is this perfect for?" persona descriptions using Claude Haiku.
Input:  data/jb_raw.csv
Output: data/jb_enriched.csv
"""

import os
import time
import pandas as pd
import anthropic
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

PROMPT = """You are a gifting expert. Given a product name, category, price and description, write 2-3 sentences describing WHO this product is perfect for as a gift.

Focus on:
- Personality types, hobbies, and lifestyles it suits
- Age groups or life stages it fits  
- Problems it solves or experiences it enables
- What kind of person would be THRILLED to receive this

Do NOT describe product features or specs. Describe the PERSON it suits.
Be specific and vivid. Write in present tense.

Product: {name}
Category: {category}
Price: ${price}
Description: {description}

Who is this perfect for (2-3 sentences only):"""


def enrich(row) -> str:
    desc = str(row.get("description", "")).strip()
    if not desc:
        desc = f"A {row['category']} product."

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": PROMPT.format(
                    name=row["name"],
                    category=row["category"],
                    price=row.get("price", "unknown"),
                    description=desc[:500],
                )
            }]
        )
        return msg.content[0].text.strip()
    except Exception as e:
        print(f"  Error enriching '{row['name']}': {e}")
        return desc


def main():
    df = pd.read_csv("data/jb_raw.csv")
    print(f"Enriching {len(df)} products with Claude Haiku...")
    print("(This costs ~$1-3 for a few thousand products)\n")

    personas = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Enriching"):
        persona = enrich(row)
        personas.append(persona)
        time.sleep(0.1)  # stay under rate limits

    df["persona"] = personas
    df.to_csv("data/jb_enriched.csv", index=False)
    print(f"\nSaved to data/jb_enriched.csv")

    # Show a few examples
    print("\nExample enrichments:")
    for _, row in df.head(3).iterrows():
        print(f"\n  Product: {row['name']} (${row['price']})")
        print(f"  Persona: {row['persona']}")


if __name__ == "__main__":
    main()
