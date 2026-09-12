import asyncio
from cognitive_kitchen.ingestion.category_scraper import CategoryMenuScraper


async def run_test():
    test_category_url = "https://quotes.toscrape.com/"
    link_selector = "span a"

    scraper = CategoryMenuScraper(headless=True)
    results = await scraper.scrape_category_pipeline(
        category_url=test_category_url,
        link_selector=link_selector,
        max_items=2,
    )

    assert len(results) > 0
    assert "url" in results[0]
    assert "title" in results[0]
    assert "raw_text" in results[0]

    print("All assertions passed.")
    print(f"Scraped {len(results)} pages successfully.")
    for item in results:
        print(f"- Title: {item['title']} | URL: {item['url']}")


if __name__ == "__main__":
    asyncio.run(run_test())