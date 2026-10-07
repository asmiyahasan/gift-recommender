# gift-recommender

A Streamlit chat app that recommends real gifts based on a description of the person you're buying for. An agent (LangGraph + Claude) searches a scraped product catalogue semantically — matching on personality and lifestyle, not keywords — and returns 3 picks with reasoning.

The catalogue is ~230 products scraped from JB Hi-Fi (Australia) across 7 categories: audio, gaming accessories, smart home, cameras, tablets, wearables, and drones/robotics. Prices are in AUD.

## How it works

1. **Scrape** — `scrapers/jbhifi.py` pulls product listings from JB Hi-Fi into `data/jb_raw.csv`.
2. **Enrich** — `data/enrich.py` uses Claude Haiku to rewrite each product as a "who this is perfect for" persona, saved to `data/jb_enriched.csv`.
3. **Index** — `data/index.py` loads the enriched data into SQLite (`data/products.db`, for filtering) and ChromaDB (`chroma_db/`, for semantic search) using the local `all-MiniLM-L6-v2` embedding model.
4. **Chat** — `app.py` runs the Streamlit UI; the agent in `agent/` calls `semantic_search`, `filter_products`, and `get_categories` tools to build recommendations.

## Setup

Tested with Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your key to `.env`:

```
ANTHROPIC_API_KEY=sk-...
```

Run all commands from the project root — the scripts use relative paths.

## Run the app

```bash
streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB).

## Evaluation

Two eval scripts live in `evals/`, both driven by the 22 test personas in `evals/cases.json`. Each case has a prompt, a budget, optional exclusions, and name patterns for the products that would be a good fit.

**Retrieval** — tests semantic search on its own. No API calls, runs in seconds.

```bash
python evals/eval_retrieval.py            # add --verbose to see each ranked list
```

Reports hit@3, hit@10, precision@10 and MRR, plus how many results the `semantic_search` tool returns once the budget filter is applied.

**Agent** — runs the full agent on each persona and checks its answer against the database: exactly 3 picks in the expected format, every product really exists, stated prices match, everything is within budget (one pick up to 10% over is allowed as a "stretch"), exclusions are respected, and the agent actually searched. Calls the Claude API, roughly 1–3 cents per case.

```bash
python evals/eval_agent.py --limit 3      # quick check
python evals/eval_agent.py                # all cases
```

Results, including every agent reply, are saved to `evals/results/`. The `out_of_catalogue_cooking` case has no right answer in the catalogue and should be read by hand to see whether the agent says so honestly.

## Rebuilding the catalogue (optional)

A pre-built catalogue is included in the repo (`data/`, `chroma_db/`), so you can skip this. Only run it to refresh the products. Note that the enrich step calls the Claude API for every product.

```bash
playwright install chromium
python scrapers/jbhifi.py
python data/enrich.py
python data/index.py
```

## Disclaimer

The scraper is for personal and educational use. Product data belongs to JB Hi-Fi; check their terms before scraping or redistributing it.
