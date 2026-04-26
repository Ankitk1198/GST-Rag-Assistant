# GST Bare-Law Assistant

A grounded RAG chatbot for answering GST bare-law questions from GST Acts and Rules.

## Main files

- notebooks/streamlit_app.py: Streamlit chatbot UI
- notebooks/gst_rag_backend.py: Retrieval and generation backend
- chunks/: processed retrieval records
- indexes/: dense embeddings and retrieval index files

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install and run Ollama separately, then pull the generation model:

```bash
ollama pull gemma2:9b
ollama serve
```

## Run the app

From the project root:

```bash
cd notebooks
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in terminal.

## Notes

This assistant is grounded in the GST bare-law corpus. It does not provide legal advice, portal workflow guidance, case-law interpretation, or latest notification tracking unless such text exists in the corpus.
