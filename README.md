# Agentic RAG Gift Recommender

Describe the person you're buying for and a Claude agent finds real products from a scraped JB Hi-Fi catalogue, matching on personality and lifestyle rather than keywords, and recommends up to 3 with reasoning, prices and links.

> *"My partner is training for her first marathon and tracks every run obsessively. Under $500."*
> → Garmin Forerunner 55 ($269), Shokz OpenRun Pro 2 ($239), Garmin Vivoactive 6 ($479), each with a short note on why it suits her.

Built with LangGraph, Claude (Sonnet for the agent, Haiku for data prep), ChromaDB, SQLite and Streamlit, plus an eval suite for both retrieval and agent behaviour.

## How it works

**Agentic RAG with hybrid retrieval.** Instead of a single fixed retrieval step, a LangGraph ReAct agent decides when and how to search. It rewrites the request into a description of the person, sets the budget, and adds exclusions or a colour preference. It often searches several times before answering.

- **Persona embeddings.** Product listings aren't embedded directly. Claude Haiku writes a "who is this perfect for" persona for each product, so a description of a *person* is matched against descriptions of *people*.
- **Hybrid search.** ChromaDB vector search, with the budget filter applied inside the vector query, is combined with SQLite filtering for hard constraints like "nothing from Apple" or "no cameras".
- **Variant grouping.** Haiku groups listings that differ only by colour, storage, size, refurbished grade, connectivity or pack size into one product: 232 listings → 161 products. Search returns one link per product, so the agent can't recommend two colours of the same thing. If someone asks for a specific colour or size, the agent passes it as a preference and the matching listing is linked.
- **Grounded answers.** Every product, price and link in an answer comes from the database, and the evals check this.

### Pipeline

1. **Scrape:** `scrapers/jbhifi.py` pulls listings from 7 JB Hi-Fi categories into `data/jb_raw.csv`.
2. **Group:** `data/group.py` (Haiku) groups variants of the same product into `data/jb_grouped.csv`. Different models and generations, such as iPad 9th vs 10th Gen or DualSense vs DualSense Edge, stay separate.
3. **Enrich:** `data/enrich.py` (Haiku) writes one persona per product into `data/jb_enriched.csv`.
4. **Index:** `data/index.py` builds `data/products.db` (a `products` table with one row per listing, and a `groups` table with one row per product) and `chroma_db/` (one entry per product, embedded locally with `all-MiniLM-L6-v2`).
5. **Chat:** `app.py` runs the Streamlit UI. The agent in `agent/` has three tools: `semantic_search`, `filter_products` and `get_categories`. The sidebar shows which tools it called and with what arguments.

The catalogue covers audio, gaming accessories, smart home, cameras, tablets, wearables and e-scooters. Prices are in AUD.

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

Run all commands from the project root, because the scripts use relative paths.

## Run the app

```bash
streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). A pre-built catalogue is included, so there's no need to scrape first.

## Evaluation

Two eval scripts in `evals/` share 23 test personas (`evals/cases.json`). Each case has a prompt, a budget, optional exclusions or a colour preference, and patterns for the products that would be a good fit.

### Retrieval: is search finding the right products?

```bash
python evals/eval_retrieval.py            # add --verbose to see each ranked list
```

Queries ChromaDB directly with each persona, with no LLM and no API calls. Reports hit@3, hit@10, precision@10 and MRR, plus how many in-budget results the `semantic_search` tool returns.

### Agent: does the full agent follow the rules?

```bash
python evals/eval_agent.py --limit 3      # quick check
python evals/eval_agent.py                # all cases, ~$1
```

Runs the real agent on each persona, parses its answer and checks every pick against the database. Hard checks:

| Check | Passes when |
|---|---|
| format | 1–3 numbered picks in the expected format |
| enough | at least as many picks as there are good fits in the catalogue (up to 3) |
| links | every pick has its own product URL |
| distinct | no two picks are variants of the same product |
| real | every product exists in the catalogue (nothing made up) |
| prices | the stated price matches the catalogue |
| budget | within budget; one pick up to 10% over is allowed as a "stretch" |
| exclusions | nothing the person ruled out ("no Apple", "no headsets") |
| searched | the agent actually searched rather than guessing |
| preference | a requested colour or option is the one linked |

Warnings flag softer problems: `stretch_label` (an in-budget pick called a "stretch") and `padding` (weak picks added when fewer than 3 good fits exist). Results, including every reply, are saved to `evals/results/`. Some cases are marked for checking by hand, such as a request the catalogue can't serve (cooking) and a vague Secret Santa request.

### Results

Latest full run: **23/23 agent cases pass.** Across those runs, the agent has never invented a product or misstated a price.

Problems the evals found and that were then fixed:

- **Budget filter.** Filtering after the vector search left tight budgets with only 1–2 results. Moving the price filter into the ChromaDB query fixed it.
- **Concurrent tool calls.** A race condition crashed the first query of a session when the agent ran two searches in parallel. Fixed with a lock around loading the index.
- **Duplicate variants.** The agent recommended two colours of the same controller. Prompt rules didn't stop it; returning one link per product did.
- **Over-generic personas.** 36 near-identical refurbished iPads with catch-all personas showed up in unrelated searches. Variant grouping and a more specific enrichment prompt fixed it.
- **"Stretch" mislabelling.** The agent called in-budget picks "stretch" options. A clearer rule fixed it.

Known limitations:

- **Requests to leave something out** ("no cameras", "hates bulky watches") pull those products *towards* the top of raw vector search. The agent compensates by writing its own queries and using `filter_products`, so agent results are fine even though retrieval scores for those cases are low.
- **Padding.** When only 1–2 good fits exist, the agent sometimes fills the list with loosely related products. The `padding` warning tracks this.
- **No popularity data.** "Safe" picks for vague requests are judged by broad appeal, not by sales or ratings.

## Rebuilding the catalogue (optional)

Run this to refresh products and prices, or after changing the grouping or enrichment. The group and enrich steps call Claude Haiku and cost a few cents for the whole catalogue.

```bash
playwright install chromium
python scrapers/jbhifi.py     # optional: re-scrape for fresh products and prices
python data/group.py          # prints every product with variants: check the groups look right
python data/enrich.py
python data/index.py
```

If `group.py` merges or splits something wrongly, edit `group_name` and `group_id` for those rows in `data/jb_grouped.csv`, then re-run `enrich.py` and `index.py`.

## Project structure

```
agent/       LangGraph agent and its tools
app.py       Streamlit chat UI
data/        group / enrich / index scripts, catalogue CSVs, products.db
chroma_db/   vector index
evals/       test cases and the retrieval + agent eval scripts
scrapers/    JB Hi-Fi scraper (Playwright)
```

## Disclaimer

The scraper is for personal and educational use. Product data belongs to JB Hi-Fi; check their terms before scraping or redistributing it.
