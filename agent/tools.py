"""
Agent tools for the gift recommender.
"""

import os
import json
import sqlite3
import threading
import pandas as pd
import chromadb
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# --- Shared resources (loaded once at import) ---
_conn = None
_collection = None
# The agent can run several tool calls in parallel threads. Without a lock, two
# threads can both see _collection as None and open chroma_db at the same time,
# which makes ChromaDB raise KeyError: 'chroma_db'.
_init_lock = threading.Lock()


def get_db():
    global _conn
    if _conn is None:
        with _init_lock:
            if _conn is None:
                _conn = sqlite3.connect("data/products.db", check_same_thread=False)
    return _conn


def get_collection():
    global _collection
    if _collection is None:
        with _init_lock:
            if _collection is None:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                chroma = chromadb.PersistentClient(path="chroma_db")
                ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                _collection = chroma.get_collection("products", embedding_function=ef)
    return _collection


# --- Tools ---

@tool
def semantic_search(
    query: str,
    min_price: float = 0,
    max_price: float = 9999,
    n_results: int = 8,
) -> str:
    """
    Search the product catalogue semantically using a description of the gift recipient.
    
    The query should describe the PERSON (their personality, hobbies, lifestyle, needs),
    NOT the product. The search matches against 'who this product is perfect for'.
    
    Args:
        query: Description of the person receiving the gift
        min_price: Minimum price in AUD (default 0)
        max_price: Maximum price in AUD (default 9999)
        n_results: Number of results to return (default 8)
    
    Returns:
        JSON list of matching products with name, price, category, url, and why_it_fits
    """
    collection = get_collection()

    # Filter by price inside the vector search, so we get the n_results closest
    # in-budget products. (Fetching a fixed number and filtering afterwards often
    # left only one or two results for tight budgets.)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"$and": [{"price": {"$gte": min_price}}, {"price": {"$lte": max_price}}]},
        include=["metadatas", "distances", "documents"],
    )

    products = []
    for i, meta in enumerate(results["metadatas"][0]):
        price = meta.get("price")
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        if min_price <= price <= max_price:
            products.append({
                "name": meta.get("name", ""),
                "price": price,
                "category": meta.get("category", ""),
                "store": meta.get("store", ""),
                "url": meta.get("url", ""),
                "why_it_fits": results["documents"][0][i],
                "relevance": round(1 - results["distances"][0][i], 3),
            })

    products.sort(key=lambda x: x["relevance"], reverse=True)
    return json.dumps(products[:n_results], indent=2)


@tool
def filter_products(
    max_price: float = 9999,
    min_price: float = 0,
    exclude_categories: list = [],
    include_categories: list = [],
    exclude_keywords: list = [],
) -> str:
    """
    Filter products from the database using structured constraints.
    
    Use this when the user says things like:
    - "not another pair of headphones" (exclude_categories)
    - "specifically looking for smart home stuff" (include_categories)  
    - "nothing from Sony" (exclude_keywords)
    - "under $50" (max_price)
    
    Args:
        max_price: Maximum price in AUD
        min_price: Minimum price in AUD
        exclude_categories: List of category slugs to exclude
        include_categories: List of category slugs to include (empty = all)
        exclude_keywords: Words to exclude from product names
    
    Returns:
        JSON list of matching products
    """
    conn = get_db()
    query = f"SELECT name, price, category, store, url FROM products WHERE price BETWEEN {min_price} AND {max_price}"

    if include_categories:
        cats = ", ".join([f"'{c}'" for c in include_categories])
        query += f" AND category IN ({cats})"

    if exclude_categories:
        cats = ", ".join([f"'{c}'" for c in exclude_categories])
        query += f" AND category NOT IN ({cats})"

    for kw in exclude_keywords:
        query += f" AND LOWER(name) NOT LIKE LOWER('%{kw}%')"

    query += " ORDER BY RANDOM() LIMIT 20"

    try:
        df = pd.read_sql_query(query, conn)
        return df.to_json(orient="records")
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_categories() -> str:
    """
    Returns all available product categories and store names in the database.
    Use this to understand what's available before filtering.
    """
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT category, store, COUNT(*) as count FROM products GROUP BY category, store ORDER BY count DESC",
        conn,
    )
    return df.to_json(orient="records")