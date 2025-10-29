# DSA Planning Chatbot Demo

This folder contains a minimal demo of the NUS Data Science & Analytics planning chatbot.
The code is organised for a lightweight Flask backend paired with a Streamlit user
interface so the experience matches the original notebook while being easy to run for a demo.

## Getting started

1. **Create a virtual environment (optional but recommended).**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. **Install the app requirements.**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start the Flask backend.**
   ```bash
   python -m app.backend
   ```
4. **Launch the Streamlit UI in a separate terminal.**
   ```bash
   streamlit run app/ui.py
   ```

The Streamlit page defaults to `http://localhost:8501` and expects the Flask API to be
available at `http://localhost:5000`. You can point the UI to a different backend by
setting the `CHATBOT_API_ROOT` environment variable before running Streamlit.

Chat requests wait up to five minutes for the backend to respond, which is helpful when
running larger local models that take longer to generate an answer.

Use the **Developer view** toggle in the sidebar to stream the full trace of LangGraph
events, including every intermediate message emitted during tool calls. The toggle can be
disabled at any time for a clean end-user experience. A **Reset conversation** button is
also available to clear the backend session, wipe the UI, and start a fresh chat.

## Notes

- The LangChain/LangGraph stack still depends on a local Ollama model (``qwen3:14b``) as
  configured in the notebook. Make sure the model is installed and running locally before
  starting the backend.
- This repository root may contain experimental notebooks or other assets that will be
  discarded later; the `/app` directory is self-contained so it can be copied elsewhere
  when the demo is ready.
