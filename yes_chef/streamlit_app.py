"""
Recipe Remix Chef - Streamlit Interface
A web-based cooking assistant powered by Claude
"""

import streamlit as st
from recipe_chef_agent import RecipeRemixChef, client
from web_fetcher import RecipeFetcher, format_recipes_for_context

# Page configuration
st.set_page_config(
    page_title="Recipe Remix Chef",
    page_icon="🍳",
    layout="centered"
)

# Initialize the chef
if 'chef' not in st.session_state:
    st.session_state.chef = RecipeRemixChef()

# Initialize the recipe fetcher
if 'fetcher' not in st.session_state:
    st.session_state.fetcher = RecipeFetcher()

# Initialize conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Header
st.title("🍳 Yes Chef!")
st.markdown("---")
st.markdown("Tell me what ingredients you have, and I'll help you cook something delicious!")

# Display conversation history
for message in st.session_state.conversation:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant", avatar="👨‍🍳"):
            st.write(message["content"])

# Chat input
user_input = st.chat_input("What ingredients do you have?")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Get response from Claude with grounded recipe context
    with st.chat_message("assistant", avatar="👨‍🍳"):
        # Search for relevant recipes from whitelisted sites
        with st.spinner("Searching recipe sites..."):
            recipes = st.session_state.fetcher.search_recipes(user_input, max_results=3)
            recipe_context = format_recipes_for_context(recipes)

        # System prompt for the chef with grounding instruction
        system_prompt = f"""You are a creative chef assistant that ONLY uses recipes from these trusted cooking websites:
- Minimalist Baker (https://minimalistbaker.com/)
- Smitten Kitchen (https://smittenkitchen.com/)
- Gimme Some Oven (https://www.gimmesomeoven.com/)

IMPORTANT: Base your responses ONLY on the recipe information provided below. If no relevant recipes are found, let the user know and suggest they try different ingredients or search terms.

{recipe_context}

Help users:
- Find recipes based on their ingredients
- Adapt recipes for dietary needs or missing ingredients
- Scale recipes for different servings
- Plan cooking timelines
- Generate shopping lists

Always cite which website a recipe comes from and provide the URL when recommending recipes.
Be friendly, practical, and creative with substitutions."""

        # Add user message to conversation
        st.session_state.conversation.append({
            "role": "user",
            "content": user_input
        })

        # Get Claude's response
        with st.spinner("Chef Claude is thinking..."):
            response = client.messages.create(
                model=st.session_state.chef.model,
                max_tokens=2000,
                system=system_prompt,
                messages=st.session_state.conversation
            )

            assistant_message = response.content[0].text
            st.write(assistant_message)

    # Add assistant message to conversation
    st.session_state.conversation.append({
        "role": "assistant",
        "content": assistant_message
    })

# Sidebar with controls
with st.sidebar:
    st.header("Controls")

    if st.button("Clear Conversation"):
        st.session_state.conversation = []
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.markdown("This is an interactive cooking assistant powered by Claude AI that helps you:")
    st.markdown("- Create recipes from available ingredients")
    st.markdown("- Adapt recipes for dietary restrictions")
    st.markdown("- Scale recipes for different servings")
    st.markdown("- Plan cooking timelines")
    st.markdown("- Generate shopping lists")

    st.markdown("---")
    st.markdown("### 🌐 Trusted Recipe Sources")
    st.markdown("All recipes are sourced from:")
    st.markdown("- [Minimalist Baker](https://minimalistbaker.com/)")
    st.markdown("- [Smitten Kitchen](https://smittenkitchen.com/)")
    st.markdown("- [Gimme Some Oven](https://www.gimmesomeoven.com/)")
