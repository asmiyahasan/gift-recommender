# gift-recommender

A Streamlit chat app that recommends real gifts based on a description of the person you're buying for. An agent (LangGraph + Claude) searches a scraped product catalogue semantically — matching on personality and lifestyle, not keywords — and returns 3 picks with reasoning.

## How it works

1. **Scrape** — `scrapers/jbhifi.py` pulls product listings from JB Hi-Fi into `data/jb_raw.csv`.
2. **Enrich** — `data/enrich.py` uses Claude Haiku to rewrite each product as a "who this is perfect for" persona, saved to `data/jb_enriched.csv`.
3. **Index** — `data/index.py` loads the enriched data into SQLite (`data/products.db`, for filtering) and ChromaDB (`chroma_db/`, for semantic search).
4. **Chat** — `app.py` runs the Streamlit UI; the agent in `agent/` calls `semantic_search`, `filter_products`, and `get_categories` tools to build recommendations.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit langgraph langchain-anthropic chromadb sentence-transformers pandas playwright anthropic voyageai tqdm python-dotenv
playwright install chromium
```

Add your key to `.env`:

```
ANTHROPIC_API_KEY=sk-...
```

## Run the pipeline (once, to build the catalogue)

```bash
python scrapers/jbhifi.py
python data/enrich.py
python data/index.py
```

## Run the app

```bash
streamlit run app.py
```
