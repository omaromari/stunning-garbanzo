from gpt4all import GPT4All
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st

# model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", allow_download=True)
# model_path = "C:\\Users\\thero\\.cache\\gpt4all\\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
def load_model():
    # Load the GPT4All model
    model_path = r"D:\thero\gpt4all\Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    model = GPT4All(model_path)
    return model

def get_response(model, prompt: str) -> str:
    tokens = []
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for token in model.generate(prompt=prompt, top_k=1, streaming=True):
            tokens.append(token)
            message_placeholder.markdown(f"**Bot:** {''.join(tokens)}", unsafe_allow_html=True)
    response = ''.join(tokens)
    return response

# def get_response(model, prompt: str) -> str:
#     # Get the response from the model
#     # Generate a streaming response and print it as it arrives
#     tokens = []
#     for token in model.generate(prompt=prompt, top_k=1, streaming=True):
#         tokens.append(token)
#         # print(token, end='', flush=True)
#         ## write to streamlit with ** Bot: ** prefix
#         # st.write(f"** Bot: ** {''.join(tokens)}", unsafe_allow_html=True)
#         st.write(f"** Bot: ** {token}", unsafe_allow_html=True)
#     # Join the tokens to form the final response
#     response = ''.join(tokens)
#     st.write(f"** Bot: ** {response}", unsafe_allow_html=True)

    return response

def get_context(model, query: str, documents: List[str], top_k: int = 5) -> str:
    # Get the context for the query from the documents
    context = []
    for doc in documents:
        if query.lower() in doc.lower():
            context.append(doc)
    context = context[:top_k]
    return "\n".join(context)

def get_answer(model, query: str) -> str:
    # Get the answer from the model based on the query
    prompt = f"Answer the following question: {query}"
    answer = get_response(model, prompt)
    return answer

def get_answer_with_context(model, query: str, documents: List[str], top_k: int = 5) -> str:
    # Get the answer from the model based on the query and context
    context = get_context(model, query, documents, top_k)
    ## Simple prompt with context
    # prompt = f"Answer the following question based on the context: {query}\n\nContext:\n{context}"
    ## More complex prompt with context
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
    answer = get_response(model, prompt)
    return answer

def main():
    model = load_model()
    query = "What is the capital of France?"
    documents = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Italy is Rome."
    ]
    answer = get_answer_with_context(model, query, documents)
    print(answer)

if __name__ == "__main__":
    main()