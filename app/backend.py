"""Flask backend that exposes the chatbot as a simple HTTP API."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from flask import Flask, jsonify, request

from .chat_graph import (
    ChatService,
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
)

# Single shared instance keeps the conversation in memory for demo purposes.
chat_service = ChatService()

app = Flask(__name__)


def _normalise_system_prompt(candidate: str | None) -> str:
    if candidate is None:
        return DEFAULT_SYSTEM_PROMPT_TEMPLATE
    stripped = candidate.strip()
    return stripped or DEFAULT_SYSTEM_PROMPT_TEMPLATE


def _normalise_bool(value: bool | None, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


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
    system_prompt_template = payload.get("system_prompt_template")
    reasoning_enabled = payload.get("enable_reasoning")
    retriever_enabled = payload.get("enable_retriever")
    chat_service.configure(
        system_prompt_template=system_prompt_template or DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        reasoning_enabled=reasoning_enabled,
        retriever_enabled=retriever_enabled,
    )
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


@app.post("/api/evaluate")
def evaluate():
    """Run an evaluation sweep over a batch of prompts with isolated state."""

    payload = request.get_json(silent=True) or {}
    prompts = payload.get("prompts") or []
    if not isinstance(prompts, list) or not any((prompt or "").strip() for prompt in prompts):
        return jsonify({"error": "prompts must be a non-empty list of strings"}), 400

    system_prompt_template = _normalise_system_prompt(
        payload.get("system_prompt_template")
    )
    reasoning_enabled = _normalise_bool(payload.get("enable_reasoning"), False)
    retriever_enabled = _normalise_bool(payload.get("enable_retriever"), True)

    evaluation_service = ChatService(
        max_history=chat_service.max_history,
        system_prompt_template=system_prompt_template,
        reasoning_enabled=reasoning_enabled,
        retriever_enabled=retriever_enabled,
    )

    cleaned_prompts = [prompt for prompt in prompts if (prompt or "").strip()]
    responses = evaluation_service.evaluate(cleaned_prompts)
    body = {
        "configuration": {
            "system_prompt_template": system_prompt_template,
            "reasoning_enabled": reasoning_enabled,
            "retriever_enabled": retriever_enabled,
        },
        "results": [
            {
                "prompt": prompt,
                "answer": response.answer,
                "history": response.history,
            }
            for prompt, response in zip(cleaned_prompts, responses)
        ],
    }

    return jsonify(body)


if __name__ == "__main__":
    # Running through ``python backend.py`` starts a development server suitable for demos.
    app.run(host="0.0.0.0", port=5000, debug=True)
