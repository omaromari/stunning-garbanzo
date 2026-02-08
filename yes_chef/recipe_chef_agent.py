"""
Recipe Remix Chef - Claude Agent SDK Starter Project
A cooking assistant that adapts recipes based on available ingredients and dietary restrictions.
"""

import anthropic
import os
from typing import List, Dict, Optional

# Initialize the Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

class RecipeRemixChef:
    def __init__(self):
        self.model = "claude-sonnet-4-5-20250929"
        self.conversation_history = []
        
    def search_recipes(self, ingredients: List[str], dietary_restrictions: List[str] = None) -> str:
        """Search for recipes based on ingredients and restrictions"""
        # TODO: Implement actual recipe search (API integration or web scraping) using Claude SDK
        return f"Found recipes for: {', '.join(ingredients)}"
    
    def adapt_recipe(self, recipe: str, available_ingredients: List[str], 
                     dietary_restrictions: List[str] = None) -> Dict:
        """Adapt a recipe based on what's available"""
        prompt = f"""
        Adapt this recipe based on the available ingredients and restrictions:
        
        Recipe: {recipe}
        Available Ingredients: {', '.join(available_ingredients)}
        Dietary Restrictions: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}
        
        Provide:
        1. Adapted recipe with substitutions explained
        2. Missing ingredients needed
        3. Step-by-step instructions
        4. Estimated prep/cook time
        """
        
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "adapted_recipe": response.content[0].text,
            "conversation_id": response.id
        }
    
    def scale_recipe(self, recipe: str, original_servings: int, desired_servings: int) -> str:
        """Scale recipe quantities for different serving sizes"""
        prompt = f"""
        Scale this recipe from {original_servings} servings to {desired_servings} servings.
        Adjust all ingredient quantities proportionally and round to practical measurements.
        
        Recipe: {recipe}
        """
        
        response = client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_shopping_list(self, recipes: List[str], pantry_items: List[str] = None) -> Dict:
        """Generate optimized shopping list from multiple recipes"""
        pantry_items = pantry_items or []
        
        prompt = f"""
        Generate a consolidated shopping list for these recipes, excluding pantry items:
        
        Recipes:
        {chr(10).join(f"- {recipe}" for recipe in recipes)}
        
        Already have: {', '.join(pantry_items)}
        
        Group items by category (produce, dairy, meat, pantry, etc.) and combine quantities.
        """
        
        response = client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "shopping_list": response.content[0].text,
            "estimated_cost": "TBD"  # Could integrate with price APIs
        }
    
    def create_cooking_timeline(self, recipes: List[Dict]) -> str:
        """Create a coordinated timeline for cooking multiple dishes"""
        prompt = f"""
        Create a cooking timeline to prepare these dishes so they're all ready at the same time:
        
        {chr(10).join(f"- {r['name']}: Prep {r['prep_time']}min, Cook {r['cook_time']}min" 
                      for r in recipes)}
        
        Provide a minute-by-minute schedule working backwards from serving time.
        """
        
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def interactive_cooking_session(self):
        """Interactive chat session for cooking guidance"""
        print("🍳 Recipe Remix Chef - Interactive Mode")
        print("=" * 50)
        print("Tell me what ingredients you have, and I'll help you cook something delicious!")
        print("Type 'quit' to exit.\n")
        
        conversation = []
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Happy cooking! 👨‍🍳")
                break
            
            conversation.append({
                "role": "user",
                "content": user_input
            })
            
            # Add system context for the agent
            system_prompt = """You are a creative chef assistant. Help users:
            - Find recipes based on their ingredients
            - Adapt recipes for dietary needs or missing ingredients
            - Scale recipes for different servings
            - Plan cooking timelines
            - Generate shopping lists
            
            Be friendly, practical, and creative with substitutions."""
            
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system_prompt,
                messages=conversation
            )
            
            assistant_message = response.content[0].text
            conversation.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            print(f"\nChef Claude: {assistant_message}\n")


def main():
    """Demo the Recipe Remix Chef capabilities"""
    chef = RecipeRemixChef()
    
    print("🍳 Recipe Remix Chef - Demo Mode\n")
    
    # Example 1: Adapt a recipe
    print("=" * 50)
    print("Example 1: Adapting a recipe")
    print("=" * 50)
    
    sample_recipe = "Chicken Parmesan (serves 4)"
    available = ["chicken breast", "tomato sauce", "mozzarella", "breadcrumbs"]
    restrictions = ["gluten-free"]
    
    result = chef.adapt_recipe(sample_recipe, available, restrictions)
    print(result["adapted_recipe"])
    print("\n")
    
    # Example 2: Generate shopping list
    print("=" * 50)
    print("Example 2: Shopping List")
    print("=" * 50)
    
    recipes = ["Spaghetti Carbonara", "Caesar Salad"]
    pantry = ["salt", "pepper", "olive oil", "garlic"]
    
    shopping = chef.generate_shopping_list(recipes, pantry)
    print(shopping["shopping_list"])
    print("\n")
    
    # Example 3: Interactive mode
    print("=" * 50)
    print("Entering Interactive Mode...")
    print("=" * 50)
    chef.interactive_cooking_session()


if __name__ == "__main__":
    # Make sure to set your API key:
    # export ANTHROPIC_API_KEY='your-api-key-here'
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY environment variable")
        print("   export ANTHROPIC_API_KEY='your-api-key-here'")
    else:
        main()
