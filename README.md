# DSA Planning Chatbot

The DSA Planning Chatbot was created as a project for DSA4213 Natural Language Processing for Data Science to help prospective students explore National University of Singapore (NUS) Data Science & Analytics (DSA) degree requirements. It combines a LangGraph-powered assistant, a Flask REST API, and a Streamlit interface so the complete planning workflow is available from this repository.

## Highlights
- **Conversational planner** – LangGraph coordinates tool calls, retrieval, and response generation to guide students through multi-turn planning conversations.
- **Curated programme knowledge** – A FAISS vector store indexes CHS/DSA requirement documents so the assistant can ground answers with authoritative references.
- **Evaluation toolkit** – Built-in grading flows and tracking templates simplify comparing configuration changes and monitoring response quality.

## Architecture overview
- **Backend (`app/backend.py`)** – Exposes chat, grading, and evaluation endpoints via Flask. It wires the LangGraph graph from `chat_graph.py`, retrieval helpers, and grading utilities.
- **User interface (`app/ui.py`)** – Streamlit front end for chatting with the assistant, toggling developer traces, and launching batch grading jobs.
- **Retrieval assets** – `app/build_vectors.py` refreshes the FAISS store found in `app/all_requirements_vectors/`, built from the structured and markdown sources in `app/data/`.
- **Support modules** – LangChain tools, NUSMods client wrappers, and grading helpers live alongside the core code for easy reuse.

## Getting started
1. *(Optional)* create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the application dependencies.
   ```bash
   pip install -r app/requirements.txt
   ```
3. Start the Flask API (defaults to `http://localhost:5000`).
   ```bash
   python -m app.backend
   ```
4. In a separate terminal, launch the Streamlit UI (defaults to `http://localhost:8501`).
   ```bash
   streamlit run app/ui.py
   ```

Set the `CHATBOT_API_ROOT` environment variable before starting Streamlit if the UI should target a different backend address. Chat requests are configured with generous timeouts so the system remains responsive even when using larger local Ollama models such as `qwen3:14b`.

## Repository map
```
.
├── README.md — Project overview, setup instructions, and repository map.
├── evaluation_results.xlsx — Testing log of 21 evaluation questions captured from the grading workflow.
└── app/
    ├── __init__.py — Marks the directory as a package for `python -m app.*` execution.
    ├── README.md — Additional quick-start notes for the application package.
    ├── backend.py — Flask application exposing chat, grading, and evaluation REST endpoints.
    ├── build_vectors.py — Regenerates the FAISS vector store from the source requirement documents.
    ├── chat_graph.py — LangGraph orchestration, chat state handling, and response serialisation.
    ├── grading.py — Wraps LangGraph grading flows to score answers against reference rubrics.
    ├── nusmods_client.py — Cached NUSMods API client with helper utilities.
    ├── retrieval.py — Loads the FAISS store and exposes retrieval/formatting helpers.
    ├── tools.py — LangChain tool definitions that surface NUSMods functionality to the agent.
    ├── ui.py — Streamlit interface for interacting with the chatbot and grading workflows.
    ├── requirements.txt — Python dependencies shared by the backend and UI components.
    ├── all_requirements_vectors/
    │   ├── index.faiss — FAISS index for similarity search.
    │   └── index.pkl — Serialized metadata accompanying the FAISS index.
    └── data/
        ├── dsa_chs_requirements.json — Structured CHS requirement data used for embedding.
        └── dsa_requirements.md — Markdown representation of DSA requirements for retrieval.
```

## Working with models and data
- Install Ollama and pull the required models (`mxbai-embed-large` for embeddings and a chat model such as `qwen3:14b`) before starting the backend.
- Run `python -m app.build_vectors` whenever the requirement source files change so the FAISS artefacts stay in sync.
- Use the `/api/grade-response` endpoint or the Streamlit grading controls to benchmark different model configurations.
