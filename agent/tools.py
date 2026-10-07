"""
Agent tools for the gift recommender.

The catalogue is organised as products (groups) and their variants: one
product such as "PS5 DualSense Wireless Controller" can have several
listings that differ only by colour, storage, size, condition or pack size.
Search works at the product level; each result lists its in-budget variants
with their exact names, prices and URLs.
"""

import json
import sqlite3
import threading

import chromadb
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

MAX_VARIANTS_SHOWN = 6  # per product, to keep tool output short

# --- Shared resources (loaded once, on first use) ---
_conn = None
_collection = None
# The agent can run several tool calls in parallel threads. Without a lock, two
# threads can both see _collection as None and open chroma_db at the same time,
# which makes ChromaDB raise KeyError: 'chroma_db'.
_init_lock = threading.Lock()
_db_lock = threading.Lock()  # one shared SQLite connection, so serialise queries


def get_db():
    global _conn
    if _conn is None:
        with _init_lock:
            if _conn is None:
                _conn = sqlite3.connect("data/products.db", check_same_thread=False)
                _conn.row_factory = sqlite3.Row
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


def _query(sql, params=()):
    with _db_lock:
        return [dict(r) for r in get_db().execute(sql, params).fetchall()]


def _variants(group_ids, min_price, max_price, exclude_keywords=()):
    """In-budget variants for each group, cheapest first: {group_id: [variant, ...]}."""
    if not group_ids:
        return {}
    sql = (
        "SELECT group_id, name, price, url FROM products "
        f"WHERE group_id IN ({', '.join('?' * len(group_ids))}) AND price BETWEEN ? AND ?"
    )
    params = [*group_ids, min_price, max_price]
    for kw in exclude_keywords:
        sql += " AND LOWER(name) NOT LIKE ?"
        params.append(f"%{kw.lower()}%")
    sql += " ORDER BY price"
    out = {}
    for r in _query(sql, params):
        out.setdefault(r.pop("group_id"), []).append(r)
    return out


def _product_entry(name, category, variants, **extra):
    return {
        "name": name,
        "category": category,
        "price_from": variants[0]["price"],
        "price_to": variants[-1]["price"],
        "variants": variants[:MAX_VARIANTS_SHOWN],
        **({"more_variants": len(variants) - MAX_VARIANTS_SHOWN} if len(variants) > MAX_VARIANTS_SHOWN else {}),
        **extra,
    }


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

    Each result is one product. Its "variants" list the in-budget colours / sizes /
    storage options with their exact name, price and URL. Recommend a specific variant.

    Args:
        query: Description of the person receiving the gift
        min_price: Minimum price in AUD (default 0)
        max_price: Maximum price in AUD (default 9999)
        n_results: Number of products to return (default 8)

    Returns:
        JSON list of products with name, category, price range, why_it_fits,
        relevance, and in-budget variants
    """
    collection = get_collection()

    # Filter by price inside the vector search: a product qualifies if any of its
    # variants is in budget (its price range overlaps the budget).
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"$and": [{"price_min": {"$lte": max_price}}, {"price_max": {"$gte": min_price}}]},
        include=["metadatas", "distances", "documents"],
    )

    metas = results["metadatas"][0]
    variants = _variants([m["group_id"] for m in metas], min_price, max_price)

    products = []
    for i, meta in enumerate(metas):
        v = variants.get(meta["group_id"])
        if not v:
            continue
        products.append(_product_entry(
            meta["name"], meta["category"], v,
            why_it_fits=results["documents"][0][i],
            relevance=round(1 - results["distances"][0][i], 3),
        ))
    return json.dumps(products, indent=2)


@tool
def filter_products(
    max_price: float = 9999,
    min_price: float = 0,
    exclude_categories: list[str] = [],
    include_categories: list[str] = [],
    exclude_keywords: list[str] = [],
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
        JSON list of up to 20 random matching products, each with its matching variants
    """
    sql = "SELECT DISTINCT group_id FROM products WHERE price BETWEEN ? AND ?"
    params = [min_price, max_price]
    if include_categories:
        sql += f" AND category IN ({', '.join('?' * len(include_categories))})"
        params += include_categories
    if exclude_categories:
        sql += f" AND category NOT IN ({', '.join('?' * len(exclude_categories))})"
        params += exclude_categories
    for kw in exclude_keywords:
        sql += " AND LOWER(name) NOT LIKE ?"
        params.append(f"%{kw.lower()}%")
    sql += " ORDER BY RANDOM() LIMIT 20"

    try:
        group_ids = [r["group_id"] for r in _query(sql, params)]
        variants = _variants(group_ids, min_price, max_price, exclude_keywords)
        groups = {
            g["group_id"]: g
            for g in _query(
                f"SELECT group_id, group_name, category FROM groups "
                f"WHERE group_id IN ({', '.join('?' * len(group_ids))})",
                group_ids,
            )
        } if group_ids else {}
        products = [
            _product_entry(groups[gid]["group_name"], groups[gid]["category"], variants[gid])
            for gid in group_ids
            if gid in groups and variants.get(gid)
        ]
        return json.dumps(products, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_categories() -> str:
    """
    Returns all available product categories and store names in the database,
    with how many products (not counting colour/size variants) each has.
    Use this to understand what's available before filtering.
    """
    rows = _query(
        "SELECT category, store, COUNT(*) AS products, SUM(n_variants) AS listings "
        "FROM groups GROUP BY category, store ORDER BY products DESC"
    )
    return json.dumps(rows)
