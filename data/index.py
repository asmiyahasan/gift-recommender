"""
Loads enriched product data into:
  - SQLite  (structured filtering by price, category, store)
  - ChromaDB (semantic search using Voyage AI embeddings via Anthropic)
Input:  data/jb_enriched.csv
Output: data/products.db + chroma_db/
"""

import os
import sqlite3
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()


def load_sqlite(df: pd.DataFrame, db_path: str):
    print("Loading into SQLite...")
    conn = sqlite3.connect(db_path)
    df.to_sql("products", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON products(price)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON products(category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_store ON products(store)")
    conn.commit()
    conn.close()
    print(f"  Saved {len(df)} rows to {db_path}")


def load_chroma(df: pd.DataFrame, chroma_path: str):
    print("Loading into ChromaDB...")

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    chroma = chromadb.PersistentClient(path=chroma_path)

    try:
        chroma.delete_collection("products")
    except Exception:
        pass

    collection = chroma.create_collection("products", embedding_function=ef)

    valid = df.dropna(subset=["persona"]).reset_index(drop=True)

    # Batch in groups of 100 to avoid rate limits
    batch_size = 100
    for i in range(0, len(valid), batch_size):
        batch = valid.iloc[i : i + batch_size]
        collection.upsert(
            ids=[str(idx) for idx in batch.index],
            documents=batch["persona"].tolist(),
            metadatas=batch[["name", "price", "category", "url", "store"]]
                .fillna("")
                .to_dict("records"),
        )
        print(f"  Indexed {min(i + batch_size, len(valid))}/{len(valid)} products...")

    print(f"  Done — {len(valid)} products in ChromaDB")


def main():
    df = pd.read_csv("data/jb_enriched.csv")
    print(f"Indexing {len(df)} products...\n")

    os.makedirs("data", exist_ok=True)
    load_sqlite(df, "data/products.db")
    print()
    load_chroma(df, "chroma_db")

    print("\nDone! Run: streamlit run app.py")


if __name__ == "__main__":
    main()