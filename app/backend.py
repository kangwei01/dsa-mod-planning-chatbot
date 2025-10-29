"""Flask backend that exposes the chatbot as a simple HTTP API."""

from __future__ import annotations

from flask import Flask, jsonify, request

from .chat_graph import ChatService

# Single shared instance keeps the conversation in memory for demo purposes.
chat_service = ChatService()

app = Flask(__name__)


@app.post("/api/reset")
def reset_chat():
    """Reset the conversation back to an empty state."""

    chat_service.reset()
    return jsonify({"status": "ok"})


@app.post("/api/chat")
def chat():
    """Submit a new user prompt to the chatbot."""

    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    developer_view = bool(payload.get("developer_view"))
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    response = chat_service.ask(prompt, developer_view=developer_view)
    body = {
        "answer": response.answer,
        "history": response.history,
    }
    if response.developer_view is not None:
        body["developer_view"] = response.developer_view
    return jsonify(body)


if __name__ == "__main__":
    # Running through ``python backend.py`` starts a development server suitable for demos.
    app.run(host="0.0.0.0", port=5000, debug=True)
