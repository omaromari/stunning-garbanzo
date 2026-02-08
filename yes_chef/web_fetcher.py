"""
Web Fetcher for Recipe Sites
Fetches and parses recipes from whitelisted cooking websites
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import time

# Whitelisted recipe sites
WHITELISTED_SITES = [
    "https://minimalistbaker.com/",
    "https://smittenkitchen.com/",
    "https://www.gimmesomeoven.com/"
]

class RecipeFetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.timeout = 10

    def search_recipes(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Search for recipes across whitelisted sites
        Returns a list of recipe summaries
        """
        all_recipes = []

        for site in WHITELISTED_SITES:
            try:
                recipes = self._search_site(site, query)
                all_recipes.extend(recipes[:max_results])
            except Exception as e:
                print(f"Error searching {site}: {e}")
                continue

        return all_recipes[:max_results]

    def _search_site(self, base_url: str, query: str) -> List[Dict[str, str]]:
        """Search a specific site for recipes"""
        recipes = []

        # Try site-specific search URLs
        if "minimalistbaker.com" in base_url:
            search_url = f"{base_url}?s={query.replace(' ', '+')}"
            recipes = self._parse_minimalist_baker(search_url)
        elif "smittenkitchen.com" in base_url:
            search_url = f"{base_url}?s={query.replace(' ', '+')}"
            recipes = self._parse_smitten_kitchen(search_url)
        elif "gimmesomeoven.com" in base_url:
            search_url = f"{base_url}?s={query.replace(' ', '+')}"
            recipes = self._parse_gimme_some_oven(search_url)

        return recipes

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a webpage"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def _parse_minimalist_baker(self, search_url: str) -> List[Dict[str, str]]:
        """Parse Minimalist Baker search results"""
        soup = self._fetch_page(search_url)
        if not soup:
            return []

        recipes = []
        # Look for recipe cards/articles
        articles = soup.find_all('article', class_='post', limit=3)

        for article in articles:
            title_elem = article.find(['h2', 'h3'])
            link_elem = article.find('a')
            excerpt_elem = article.find(['p', 'div'], class_=['excerpt', 'entry-summary'])

            if title_elem and link_elem:
                recipes.append({
                    'title': title_elem.get_text(strip=True),
                    'url': link_elem.get('href', ''),
                    'source': 'Minimalist Baker',
                    'snippet': excerpt_elem.get_text(strip=True)[:200] if excerpt_elem else ''
                })

        return recipes

    def _parse_smitten_kitchen(self, search_url: str) -> List[Dict[str, str]]:
        """Parse Smitten Kitchen search results"""
        soup = self._fetch_page(search_url)
        if not soup:
            return []

        recipes = []
        # Look for recipe entries
        entries = soup.find_all(['article', 'div'], class_=['entry', 'post'], limit=3)

        for entry in entries:
            title_elem = entry.find(['h2', 'h3', 'a'])
            link_elem = entry.find('a')
            excerpt_elem = entry.find(['p', 'div'], class_=['excerpt', 'entry-summary', 'entry-content'])

            if title_elem and link_elem:
                recipes.append({
                    'title': title_elem.get_text(strip=True),
                    'url': link_elem.get('href', ''),
                    'source': 'Smitten Kitchen',
                    'snippet': excerpt_elem.get_text(strip=True)[:200] if excerpt_elem else ''
                })

        return recipes

    def _parse_gimme_some_oven(self, search_url: str) -> List[Dict[str, str]]:
        """Parse Gimme Some Oven search results"""
        soup = self._fetch_page(search_url)
        if not soup:
            return []

        recipes = []
        # Look for recipe articles
        articles = soup.find_all('article', limit=3)

        for article in articles:
            title_elem = article.find(['h2', 'h3'])
            link_elem = article.find('a')
            excerpt_elem = article.find(['p', 'div'], class_=['excerpt', 'entry-summary'])

            if title_elem and link_elem:
                recipes.append({
                    'title': title_elem.get_text(strip=True),
                    'url': link_elem.get('href', ''),
                    'source': 'Gimme Some Oven',
                    'snippet': excerpt_elem.get_text(strip=True)[:200] if excerpt_elem else ''
                })

        return recipes

    def fetch_recipe_details(self, url: str) -> Optional[Dict[str, str]]:
        """Fetch full recipe details from a URL"""
        soup = self._fetch_page(url)
        if not soup:
            return None

        # Try to extract recipe content (works for recipe schema markup)
        recipe_data = {
            'url': url,
            'title': '',
            'ingredients': [],
            'instructions': '',
            'full_text': ''
        }

        # Try to find recipe title
        title = soup.find(['h1', 'h2'], class_=['recipe-title', 'entry-title'])
        if title:
            recipe_data['title'] = title.get_text(strip=True)

        # Try to find ingredients
        ingredients_section = soup.find(['div', 'ul'], class_=['ingredients', 'recipe-ingredients'])
        if ingredients_section:
            ingredients = ingredients_section.find_all('li')
            recipe_data['ingredients'] = [ing.get_text(strip=True) for ing in ingredients]

        # Try to find instructions
        instructions_section = soup.find(['div', 'ol'], class_=['instructions', 'recipe-instructions'])
        if instructions_section:
            recipe_data['instructions'] = instructions_section.get_text(strip=True)

        # Get general text content as fallback
        content = soup.find(['div', 'article'], class_=['content', 'entry-content', 'post-content'])
        if content:
            recipe_data['full_text'] = content.get_text(strip=True)[:2000]  # Limit to 2000 chars

        return recipe_data


def format_recipes_for_context(recipes: List[Dict[str, str]]) -> str:
    """Format recipe search results for Claude's context"""
    if not recipes:
        return "No recipes found from whitelisted sources."

    formatted = "Here are relevant recipes from trusted cooking sites:\n\n"

    for i, recipe in enumerate(recipes, 1):
        formatted += f"{i}. **{recipe['title']}** ({recipe['source']})\n"
        formatted += f"   URL: {recipe['url']}\n"
        if recipe.get('snippet'):
            formatted += f"   {recipe['snippet']}\n"
        formatted += "\n"

    return formatted
