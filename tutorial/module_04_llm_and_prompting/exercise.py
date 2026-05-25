# =============================================================
# Module 04: LLM Inference & Prompt Construction — Exercise
# =============================================================
# Instructions:
#   1. Set MODEL_PATH below to the full path of your .gguf file.
#      See ../README.md for download instructions.
#   2. Replace every _____ with the correct code.
#   3. Run with:
#       python exercise.py        (Windows)
#       python3 exercise.py       (macOS / Linux)
#   Hints and answers are in README.md
# =============================================================

# NOTE: requires GPT4All model at the path below.
# Download instructions: see ../README.md — "Downloading the LLM"
# Examples:
#   Windows:  r"C:\Users\you\AppData\Local\nomic.ai\GPT4All\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
#   macOS:    "/Users/you/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
#   Linux:    "/home/you/.local/share/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf"
MODEL_PATH = "path/to/Meta-Llama-3-8B-Instruct.Q4_0.gguf"

from gpt4all import GPT4All


# =============================================================
# Exercise 1: Load the GPT4All model
# =============================================================

def load_model():
    # --- Exercise 1: Instantiate GPT4All with the model file path ---
    # FILL IN: Pass MODEL_PATH to GPT4All to load the local LLM
    model = _____(MODEL_PATH)
    return model


# =============================================================
# Exercise 2: Generate a streaming response
# =============================================================

def get_response(model, prompt: str) -> str:
    """Generate a response and print tokens as they arrive."""
    tokens = []
    # --- Exercise 2: Call generate() with streaming enabled ---
    # FILL IN: Pass two keyword arguments:
    #   - one to set the sampling parameter to 1 (greedy decoding)
    #   - one to enable token-by-token streaming output
    for token in model.generate(prompt=prompt, _____, _____):
        tokens.append(token)
        print(token, end="", flush=True)
    print()
    return "".join(tokens)


# =============================================================
# Exercises 3, 4, 5: Extract context from retrieved documents
# =============================================================

def get_context(model, query: str, documents: list, top_k: int = 5) -> str:
    """Filter retrieved documents to those relevant to the query."""
    context = []
    for doc in documents:
        # --- Exercise 3: Keep only documents that contain the query ---
        # FILL IN: Case-insensitive substring check: does `doc` contain `query`?
        if _____.lower() in _____.lower():
            context.append(doc)

    # --- Exercise 4: Limit to top_k documents ---
    # FILL IN: Slice context to at most top_k items
    context = context[:_____]

    # --- Exercise 5: Join documents into one string ---
    # FILL IN: Use the string method that joins a list with a separator between items
    return _____.join(context)


# =============================================================
# Exercise 6: Build the RAG prompt and get an answer
# =============================================================

def get_answer_with_context(model, query: str, documents: list, top_k: int = 5) -> str:
    """Build a context-aware prompt and generate an answer."""
    context = get_context(model, query, documents, top_k)

    # --- Exercise 6: Fill in the Instructions section of the RAG prompt ---
    # FILL IN: Write 4 bullet-point instructions telling the model how to behave:
    #   1. Read context carefully
    #   2. Give a concise and accurate answer
    #   3. If answer not found, say so explicitly
    #   4. Offer to answer more questions
    prompt = f"""
    You are a helpful assistant tasked with answering questions based on the provided context.
    The context is a collection of documents that may contain the answer to the question.

    ## Instructions: ##
    _____

    ## Context: ##
    {context}

    ## Question: ##
    {query}
    """

    return get_response(model, prompt)


# =============================================================
# Main — load model, run a test query
# =============================================================
if __name__ == "__main__":
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at:\n  {MODEL_PATH}")
        print("\nSet MODEL_PATH at the top of this file.")
        print("See ../README.md for download instructions.")
        raise SystemExit(1)

    print("Loading model (this may take a moment)...")
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
    answer = get_answer_with_context(model, query, sample_documents)
    print(f"\nFull answer (stored): {answer[:200]}...")
