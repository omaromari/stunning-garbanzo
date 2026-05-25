# This Documentation enconmpasses the structure of the directory, how to launch the chatbot, and shoqcase ideas for future development.

### Directory Structure
- app.py: This the main Python file used to reference the necessary utils.py, libraries and resources needed to launch the chatbot
- utils.py: This file contains the necessary functions such as documentation-chunking.
- rag_agent.py: This file initializes the selected LLM (only 1 for now)
- tools.py: This file contains other necessary agent-specific function such as context-retrieval.
- docs/: Subdirectory containing knowledge-articles in PDF format.

### How to launch
- Open a terminal window, type `streamlit run app.py` and hit `Enter` or `Return`.

### Future development
- Update prompt with relevant context to minimize hallucinations and improve response quality.
- Upload documents to Vector Database such Vertex AI or Azure OpenAI where chunking can be done systematically and reliably.
    - If available, can experiment with different embedding strategies (e.g. "Layout embedding" for image-heavy documents)     
- Use a more established LLM such as any of the Gemini-Flash or GPT-mini models for faster responses and larger context windows for better response quality.
- If going with the "GPT" route, can look into managed-infrastructures such as Open AI's Responses API or AWS Strands for automating tool-handling, and cleaning up the code-base.
- Packaging the directory in a DockerFile and deploying on AWS could be useful for showcasing to stakeholders via a shareable URL.
- Improve the infrastructure by incorporating the vector-searching as part of the tools available to the LLM.
- Move the "Chunking+Embedding" portion to a separate operation that's done whenever a new document is uploaded, or via a CRON job if there's a fixed periodic data-upload schedule.