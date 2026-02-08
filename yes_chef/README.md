# 🍳 Yes Chef! - Grounded Recipe Assistant

An AI-powered cooking assistant that provides recipe recommendations **grounded exclusively** in trusted cooking websites.

## 🌐 Trusted Recipe Sources

All recipe recommendations come from these whitelisted sites:
- [Minimalist Baker](https://minimalistbaker.com/)
- [Smitten Kitchen](https://smittenkitchen.com/)
- [Gimme Some Oven](https://www.gimmesomeoven.com/)

## Features

- **Grounded Recipe Search**: Real-time fetching from trusted cooking websites
- **Ingredient-Based Recommendations**: Tell it what you have, get relevant recipes
- **Recipe Adaptation**: Adapt recipes for dietary restrictions or missing ingredients
- **Recipe Scaling**: Adjust serving sizes
- **Shopping Lists**: Generate consolidated shopping lists
- **Cooking Timelines**: Plan multi-dish preparation

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set your Anthropic API key**:

   Create a `.env` file:
   ```bash
   ANTHROPIC_API_KEY=your-api-key-here
   ```

   Or set environment variable:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   ```

3. **Run the app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## How It Works

1. **User Query**: You ask about recipes or ingredients
2. **Web Search**: The app searches the whitelisted cooking sites in real-time
3. **Context Building**: Found recipes are formatted and added to the prompt
4. **Grounded Response**: Claude responds based ONLY on the fetched recipe information
5. **Source Citations**: All recommendations include the source website and URL

## Example Queries

- "I have chicken, tomatoes, and pasta"
- "Show me vegetarian dinner recipes"
- "I need a gluten-free dessert"
- "Quick weeknight meals with ground beef"

## Architecture

- **streamlit_app.py**: Main Streamlit interface
- **web_fetcher.py**: Fetches and parses recipes from whitelisted sites
- **recipe_chef_agent.py**: Core agent logic and helper functions

## Notes

- Responses may be slower than general AI chat due to real-time web fetching
- Quality depends on the search results from the whitelisted sites
- The bot will only recommend recipes it can actually find on these sites
