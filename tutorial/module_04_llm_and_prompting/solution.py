# =============================================================
# Module 04: LLM Inference & Prompt Construction — Solution
# =============================================================
# NOTE: requires GPT4All model at MODEL_PATH.
# See ../README.md for download instructions.
# Try the exercise first before reading this file.
# Run with:
#   python solution.py        (Windows)
#   python3 solution.py       (macOS / Linux)
# =============================================================

# Set this to your actual .gguf model path before running
MODEL_PATH = "path/to/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

import os
from gpt4all import GPT4All


def load_model():
    return GPT4All(MODEL_PATH)


def get_response(model, prompt: str) -> str:
    tokens = []
    # top_k=1 → greedy decoding (always pick the most likely next token)
    # streaming=True → yield one token at a time so output appears immediately
    for token in model.generate(prompt=prompt, top_k=1, streaming=True):
        tokens.append(token)
        print(token, end="", flush=True)
    print()
    return "".join(tokens)


def get_context(model, query: str, documents: list, top_k: int = 5) -> str:
    context = []
    for doc in documents:
        if query.lower() in doc.lower():
            context.append(doc)
    context = context[:top_k]
    return "\n".join(context)


def get_answer_with_context(model, query: str, documents: list, top_k: int = 5) -> str:
    context = get_context(model, query, documents, top_k)
    prompt = f"""
    You are a helpful assistant tasked with answering questions based on the provided context.
    The context is a collection of documents that may contain the answer to the question.

    ## Instructions: ##
    - Read the context carefully.
    - Provide a concise and accurate answer to the question.
    - If the answer is not found in the context, state that you do not know the answer.
    - Offer the user to ask another question if they wish.

    ## Context: ##
    {context}

    ## Question: ##
    {query}
    """
    return get_response(model, prompt)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at:\n  {MODEL_PATH}")
        print("\nSet MODEL_PATH at the top of this file.")
        print("See ../README.md for download instructions.")
        raise SystemExit(1)

    print("Loading model...")
    model = load_model()
    print("Model loaded.\n")

    sample_documents = [
        "Lipitor (atorvastatin) is used to lower cholesterol. "
        "Common side effects include muscle pain and weakness.",
        "Metformin is the first-line medication for type 2 diabetes. "
        "It works by decreasing glucose production in the liver.",
        "Patients taking Lipitor should avoid grapefruit juice.",
        "Metformin may cause nausea, vomiting, and diarrhea at the start of treatment.",
    ]

    query = "What are the side effects of Lipitor?"
    print(f"Query: {query}\n")
    print("Answer:")
    get_answer_with_context(model, query, sample_documents)
