# DSA Module Planning Chatbot

An interactive assistant that helps NUS Data Science & Analytics students plan
modules using live data from the NUSMods API. The project currently ships as the
`dsa_chatbot.ipynb` notebook, which demonstrates how the LangGraph agent loop,
LangChain tools, and NUSMods integration work together.

---

## Features

- **NUSMods integration** — fetch module overviews, prerequisites, and timetable
  details directly from the NUSMods v2 API.
- **Tool-enabled LLM** — orchestrate queries through an Ollama-hosted
  `qwen3:14b` model with structured tool calling.
- **Interactive traces** — stream LangGraph execution to understand how the
  assistant uses tools and composes final answers.
- **Notebook helpers** — `ask` and `reset_chat` functions make it easy to run
  ad-hoc experiments without building a separate UI.

---

## Repository Layout

```
.
├── dsa_chatbot.ipynb      # Main development notebook with the full pipeline
├── minimal_example.ipynb  # Early spike / scratchpad (kept for reference)
├── pipeline_overview.md   # Detailed documentation of the notebook pipeline
└── DSA4213 Project.pdf    # Project brief / original specification
```

---

## Prerequisites

1. **Python 3.10+** with the following packages installed:
   - `requests`
   - `langchain-core`
   - `langchain-ollama`
   - `langgraph`
2. **Ollama** running locally with the `qwen3:14b` model pulled:
   ```bash
   ollama pull qwen3:14b
   ollama run qwen3:14b --help  # sanity-check availability
   ```
3. **Jupyter** (or VS Code / JupyterLab) to open and execute the notebook.

---

## Getting Started

1. **Clone the repository** and install Python dependencies (poetry/venv/pip —
   your choice).
2. **Launch Ollama** if it is not already running in the background.
3. **Open `dsa_chatbot.ipynb`** in your notebook environment and execute the
   cells from top to bottom.
4. Use the sample prompts in the final section of the notebook to validate that
   module lookups, prerequisite reasoning, and guardrails behave as expected.

For a detailed walkthrough of the notebook code, refer to
[`pipeline_overview.md`](pipeline_overview.md).

---

## Development Tips

- Enable `developer_view=True` when calling `ask(...)` to inspect the exact
  messages and tool invocations sent to the LLM.
- If you modify the system prompt or add new tools, update `pipeline_overview.md`
  so future contributors understand the latest behaviour.
- The notebook caches NUSMods responses in memory; restart the kernel if you need
  to clear caches during testing.

---

## License

This project is provided for educational purposes as part of the NUS DSA module
planning assistant coursework. Please review institutional guidelines before
reusing it in other contexts.
