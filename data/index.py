"""
Loads the catalogue into:
  - SQLite (data/products.db)
      products: one row per listing (variant), with its group_id
      groups:   one row per product, with price range and persona
  - ChromaDB (chroma_db/): one entry per product, embedded locally with
    all-MiniLM-L6-v2, for semantic search
Input:  data/jb_grouped.csv + data/jb_enriched.csv
"""

import os
import sqlite3

import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()


def load_sqlite(variants: pd.DataFrame, groups: pd.DataFrame, db_path: str):
    print("Loading into SQLite...")
    conn = sqlite3.connect(db_path)
    variants[["name", "price", "url", "category", "store", "group_id", "group_name"]].to_sql(
        "products", conn, if_exists="replace", index=False
    )
    groups.to_sql("groups", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON products(price)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON products(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_group ON products(group_id)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_groups_id ON groups(group_id)")
    conn.commit()
    conn.close()
    print(f"  Saved {len(variants)} listings in {len(groups)} products to {db_path}")


def load_chroma(groups: pd.DataFrame, chroma_path: str):
    print("Loading into ChromaDB...")
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma = chromadb.PersistentClient(path=chroma_path)
    try:
        chroma.delete_collection("products")
    except Exception:
        pass
    collection = chroma.create_collection("products", embedding_function=ef)

    valid = groups.dropna(subset=["persona", "price_min", "price_max"])
    batch_size = 100
    for i in range(0, len(valid), batch_size):
        batch = valid.iloc[i : i + batch_size]
        collection.upsert(
            ids=batch["group_id"].tolist(),
            documents=batch["persona"].tolist(),
            metadatas=[
                {
                    "group_id": r.group_id,
                    "name": r.group_name,
                    "category": r.category,
                    "store": r.store,
                    "price_min": float(r.price_min),
                    "price_max": float(r.price_max),
                    "n_variants": int(r.n_variants),
                }
                for r in batch.itertuples()
            ],
        )
        print(f"  Indexed {min(i + batch_size, len(valid))}/{len(valid)} products...")
    print(f"  Done — {len(valid)} products in ChromaDB")


def main():
    variants = pd.read_csv("data/jb_grouped.csv").dropna(subset=["price"])
    groups = pd.read_csv("data/jb_enriched.csv")
    print(f"Indexing {len(groups)} products ({len(variants)} listings)...\n")

    os.makedirs("data", exist_ok=True)
    load_sqlite(variants, groups, "data/products.db")
    print()
    load_chroma(groups, "chroma_db")
    print("\nDone! Run: streamlit run app.py")


if __name__ == "__main__":
    main()
