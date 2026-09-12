import re
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Browser, Page

JUNK_SLUGS = {
    "login", "signin", "sign-up", "signup", "register", "password",
    "auth", "redirect", "account", "profile", "user", "settings",
    "privacy", "terms", "policy", "cookie", "about", "contact",
    "faq", "help", "download", "app", "feedback", "search", "category"
}

RECIPE_URL_INDICATORS = ("recipe", "recipes", "dish", "dishes", "menu", "meal")


class CategoryMenuScraper:
    def __init__(self, headless: bool = True):
        self.headless = headless

    def _score_link(self, href: str, text: str, el: Any) -> int:
        score = 0
        href_lower = href.lower()
        text_clean = text.strip()

        # 1. Immediate rejection filters
        if any(junk in href_lower for junk in JUNK_SLUGS):
            return -100
        if href.startswith(("#", "javascript:", "mailto:", "tel:")):
            return -100

        # 2. URL semantic bonuses
        for kw in RECIPE_URL_INDICATORS:
            if f"/{kw}/" in href_lower or f"-{kw}-" in href_lower:
                score += 40

        # Numeric ID slug bonus (e.g. /recipes/1234567-chicken-tikka)
        if re.search(r"/\d{4,}", href_lower):
            score += 20

        # 3. Structural element signals
        if el.find(["h1", "h2", "h3", "h4", "h5", "h6"]):
            score += 25
        if el.find("img"):
            score += 15

        # 4. Human-readable recipe title heuristics
        words = text_clean.split()
        if 2 <= len(words) <= 12 and len(text_clean) >= 10:
            score += 20
        elif len(words) == 1:
            score -= 10

        return score

    def discover_recipe_links(self, page: Page, base_url: str) -> List[str]:
        page.goto(base_url, wait_until="domcontentloaded", timeout=30000)
        page.evaluate("window.scrollBy(0, 1000)")
        page.wait_for_timeout(1000)

        soup = BeautifulSoup(page.content(), "html.parser")
        base_domain = urlparse(base_url).netloc

        scored_links = []
        seen_urls = set()

        for a in soup.find_all("a", href=True):
            raw_href = a["href"].strip()
            full_url = urljoin(base_url, raw_href)
            clean_url = full_url.split("?")[0].rstrip("/")

            if clean_url in seen_urls:
                continue

            # Keep requests within the same domain
            if urlparse(clean_url).netloc != base_domain:
                continue

            score = self._score_link(raw_href, a.get_text(), a)
            if score > 0:
                scored_links.append((score, clean_url))
                seen_urls.add(clean_url)

        # Order by highest probability of being an actual recipe page
        scored_links.sort(key=lambda x: x[0], reverse=True)
        return [url for _, url in scored_links]

    def scrape_detail_page(self, page: Page, detail_url: str) -> Dict[str, Any]:
        page.goto(detail_url, wait_until="domcontentloaded", timeout=30000)
        soup = BeautifulSoup(page.content(), "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "aside", "form"]):
            tag.decompose()

        title_el = soup.find(["h1", "h2"])
        title = title_el.get_text(strip=True) if title_el else "Untitled"

        # Filter out auth/system page titles that bypassed initial scoring
        title_lower = title.lower()
        if any(j in title_lower for j in ["login", "sign in", "sign up", "download app", "untitled"]):
            return {"url": detail_url, "title": "INVALID_PAGE", "menu_items": [], "raw_text": ""}

        menu_items = []
        for li in soup.find_all("li"):
            txt = li.get_text(strip=True)
            if 3 < len(txt) < 160:
                menu_items.append(txt)

        raw_text = "\n".join([line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()])

        return {
            "url": detail_url,
            "title": title,
            "menu_items": menu_items,
            "raw_text": raw_text[:3000],
        }

    def scrape_category_pipeline(
        self,
        category_url: str,
        link_selector: Optional[str] = None,
        max_items: int = 5,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        with sync_playwright() as p:
            browser: Browser = p.chromium.launch(headless=self.headless)
            page: Page = browser.new_page()

            try:
                # If the caller provided a selector, use it; otherwise, auto-discover
                if link_selector and link_selector.strip() and link_selector.strip() != "a":
                    page.goto(category_url, wait_until="domcontentloaded", timeout=30000)
                    soup = BeautifulSoup(page.content(), "html.parser")
                    candidate_urls = [
                        urljoin(category_url, el["href"])
                        for el in soup.select(link_selector)
                        if el.get("href")
                    ]
                else:
                    candidate_urls = self.discover_recipe_links(page, category_url)

                for url in candidate_urls:
                    if len(results) >= max_items:
                        break
                    data = self.scrape_detail_page(page, url)
                    if data["title"] != "INVALID_PAGE":
                        results.append(data)
            finally:
                browser.close()

        return results