"""
JB Hi-Fi scraper using stable data-testid selectors.
Outputs: data/jb_raw.csv
"""

import asyncio
import os
import pandas as pd
from playwright.async_api import async_playwright

CATEGORIES = [
    "headphones-speakers-audio",
    "gaming-accessories",
    "smart-home",
    "cameras",
    "tablets",
    "wearables",
    "drones-robotics",
]

BASE_URL = "https://www.jbhifi.com.au"
EXTRACT_JS = """
() => {
    const cards = document.querySelectorAll('.ProductCard');
    const results = [];
    const seen = new Set();

    for (const card of cards) {
        const titleEl = card.querySelector('[data-testid="product-card-title"]');
        const linkEl  = card.querySelector('[data-testid="product-card-content-link"]');
        const priceEl = card.querySelector('[data-testid="ticket-price"]');

        if (!titleEl || !linkEl) continue;

        const name  = titleEl.innerText.trim();
        const href  = linkEl.getAttribute('href') || '';
        const price = priceEl ? parseFloat(priceEl.innerText.replace(/[^0-9.]/g, '')) : null;
        const url   = href.startsWith('http') ? href : 'https://www.jbhifi.com.au' + href;

        if (!name || !href || seen.has(url)) continue;
        seen.add(url);

        results.push({ name, price, url });
    }
    return results;
}
"""


async def scroll_and_extract(page, category: str) -> list[dict]:
    url = f"{BASE_URL}/collections/{category}"
    print(f"\n  Scraping: {url}")

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)
    except Exception as e:
        print(f"  Failed: {e}")
        return []

    # Scroll incrementally until no new products appear
    prev_count = 0
    stale = 0
    for _ in range(60):
        await page.evaluate("window.scrollBy(0, 600)")
        await page.wait_for_timeout(400)
        count = await page.evaluate(
            "document.querySelectorAll('.ProductCard').length"
        )
        if count == prev_count:
            stale += 1
            if stale >= 5:
                break
        else:
            stale = 0
        prev_count = count

    print(f"  {count} cards in DOM — extracting...")
    products = await page.evaluate(EXTRACT_JS)

    # Tag with category and store
    for p in products:
        p["category"] = category
        p["store"] = "jbhifi"

    print(f"  → {len(products)} unique products extracted")
    if products:
        print(f"     Sample: {products[0]['name']} | ${products[0]['price']}")
    return products


async def main():
    all_products = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        for category in CATEGORIES:
            products = await scroll_and_extract(page, category)
            all_products.extend(products)
            await asyncio.sleep(3)

        await browser.close()

    df = pd.DataFrame(all_products).drop_duplicates(subset=["url"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/jb_raw.csv", index=False)
    print(f"\n✅ Saved {len(df)} products to data/jb_raw.csv")
    print(df[["name", "price", "category"]].head(10).to_string())


if __name__ == "__main__":
    asyncio.run(main())