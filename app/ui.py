"""Streamlit user interface for the chatbot demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from app.chat_graph import DEFAULT_SYSTEM_PROMPT_TEMPLATE
else:
    from .chat_graph import DEFAULT_SYSTEM_PROMPT_TEMPLATE

API_ROOT = os.getenv("CHATBOT_API_ROOT", "http://localhost:5000/api")
CHAT_ENDPOINT = f"{API_ROOT}/chat"
RESET_ENDPOINT = f"{API_ROOT}/reset"

st.set_page_config(page_title="DSA Planning Chatbot", page_icon="🧭")
st.title("DSA Planning Chatbot")
st.caption("Quick demo UI backed by a Flask API and LangGraph agent.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "developer_payload" not in st.session_state:
    st.session_state["developer_payload"] = []
if "ablation_prompt_template" not in st.session_state:
    st.session_state["ablation_prompt_template"] = DEFAULT_SYSTEM_PROMPT_TEMPLATE
if "ablation_reasoning_enabled" not in st.session_state:
    st.session_state["ablation_reasoning_enabled"] = False
if "ablation_retriever_enabled" not in st.session_state:
    st.session_state["ablation_retriever_enabled"] = True
if "ground_truth_text" not in st.session_state:
    st.session_state["ground_truth_text"] = ""

st.markdown(
    """
    <style>
    .assistant-loading {
        display: inline-flex;
        gap: 0.35rem;
        align-items: center;
        padding: 0.1rem 0;
    }
    .assistant-loading .dot {
        width: 0.35rem;
        height: 0.35rem;
        background-color: currentColor;
        border-radius: 50%;
        opacity: 0.25;
        animation: assistant-loading-bounce 1.4s infinite ease-in-out;
    }
    .assistant-loading .dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    .assistant-loading .dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    @keyframes assistant-loading-bounce {
        0%, 80%, 100% {
            opacity: 0.25;
            transform: scale(0.75);
        }
        40% {
            opacity: 1;
            transform: scale(1);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ASSISTANT_LOADING_HTML = (
    "<div class=\"assistant-loading\">"
    "<span class=\"dot\"></span>"
    "<span class=\"dot\"></span>"
    "<span class=\"dot\"></span>"
    "</div>"
)

def _merge_history(existing: list[dict[str, str]], new: list[dict[str, str]]) -> list[dict[str, str]]:
    """Combine truncated API history with the full UI history."""

    if not existing:
        return list(new)

    max_overlap = min(len(existing), len(new))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if existing[-size:] == new[:size]:
            overlap = size
            break

    if overlap == 0 and new:
        # If no overlap is detected we assume the API returned a new session.
        return list(new)

    return existing + new[overlap:]


def _render_evaluation(evaluation: dict) -> None:
    """Display the grader scores inside the active chat container."""

    if not evaluation:
        return

    error_message = evaluation.get("error")
    if error_message:
        st.warning(f"Grader could not produce a score: {error_message}")
        return

    scores = evaluation.get("scores", {})
    accuracy = scores.get("accuracy")
    relevance = scores.get("relevance")
    coherence = scores.get("coherence")
    total = evaluation.get("total")

    def _format_score(value: float | None) -> str:
        if value is None:
            return "–"
        return f"{value:.1f}"

    lines = [
        "**Grader scores**",
        f"- Accuracy: {_format_score(accuracy)}",
        f"- Relevance: {_format_score(relevance)}",
        f"- Fluency & coherence: {_format_score(coherence)}",
    ]
    if total is not None:
        lines.extend(["", f"**Final score:** {total:.1f} / 3"])

    st.markdown("\n".join(lines))


ASSISTANT_AVATAR = "🤖"
AVATARS = {
    "assistant": ASSISTANT_AVATAR,
    "user": "🧑‍💻",
}

with st.sidebar:
    st.header("Session Controls")
    developer_view_enabled = st.toggle("Developer view", value=False, help="Show request and tool trace details.")
    if st.button("Reset conversation"):
        try:
            requests.post(RESET_ENDPOINT, timeout=10)
            st.session_state["messages"] = []
            st.session_state["developer_payload"] = []
            st.success("Conversation cleared.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Failed to reset chat: {exc}")

    st.divider()
    st.header("Ablation controls")
    st.caption("Adjust the assistant configuration for quick ablation studies.")
    st.text_area(
        "Assistant system prompt",
        key="ablation_prompt_template",
        height=220,
    )
    st.toggle(
        "Enable Qwen reasoning",
        key="ablation_reasoning_enabled",
        value=st.session_state["ablation_reasoning_enabled"],
        help="Toggle the model's built-in reasoning capabilities.",
    )
    st.toggle(
        "Enable retriever tool",
        key="ablation_retriever_enabled",
        value=st.session_state["ablation_retriever_enabled"],
        help="Allow the assistant to call the curriculum retriever node.",
    )
    st.divider()
    st.header("Grading")
    st.caption(
        "Provide a reference answer if you would like the assistant's next response to be graded."
    )
    st.text_area(
        "Ground truth reference (optional)",
        key="ground_truth_text",
        height=180,
        help="Leave blank to skip grading.",
    )

for item in st.session_state.get("messages", []):
    role = item.get("role", "assistant")
    content = item.get("content", "")
    avatar = AVATARS.get(role, "💬")
    chat_role = role if role in ("user", "assistant") else "assistant"
    with st.chat_message(chat_role, avatar=avatar):
        st.markdown(content)
        metadata = item.get("metadata") or {}
        evaluation_meta = metadata.get("evaluation") if isinstance(metadata, dict) else None
        if evaluation_meta:
            _render_evaluation(evaluation_meta)

user_prompt = st.chat_input(
    placeholder="e.g. What is the timetable for DSA4213 in semester 1?",
)

if user_prompt and user_prompt.strip():
    user_prompt = user_prompt.strip()
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        message_placeholder = st.empty()
        message_placeholder.markdown(ASSISTANT_LOADING_HTML, unsafe_allow_html=True)
        try:
            response = requests.post(
                CHAT_ENDPOINT,
                json={
                    "prompt": user_prompt,
                    "developer_view": developer_view_enabled,
                    "system_prompt_template": st.session_state["ablation_prompt_template"],
                    "enable_reasoning": st.session_state["ablation_reasoning_enabled"],
                    "enable_retriever": st.session_state["ablation_retriever_enabled"],
                    "ground_truth": st.session_state.get("ground_truth_text", ""),
                },
                timeout=(10, 600),
            )
            response.raise_for_status()
            data = response.json()
            api_history = data.get("history", [])
            st.session_state["messages"] = _merge_history(
                st.session_state.get("messages", []), api_history
            )
            developer_info = data.get("developer_view")
            if developer_info:
                st.session_state["developer_payload"].append(developer_info)

            assistant_reply = ""
            last_assistant_message: dict[str, object] | None = None
            for message in reversed(st.session_state["messages"]):
                if message.get("role") == "assistant":
                    assistant_reply = message.get("content", "")
                    last_assistant_message = message
                    break
            if not assistant_reply:
                assistant_reply = data.get("answer", "")
            if assistant_reply:
                message_placeholder.markdown(assistant_reply)
            else:
                message_placeholder.empty()
            if last_assistant_message:
                metadata = last_assistant_message.get("metadata") or {}
                evaluation_meta = metadata.get("evaluation") if isinstance(metadata, dict) else None
                if evaluation_meta:
                    _render_evaluation(evaluation_meta)
        except requests.RequestException as exc:
            error_message = f"Failed to contact the chatbot API: {exc}"
            message_placeholder.markdown(error_message)
            st.session_state["messages"].append({"role": "assistant", "content": error_message})


if developer_view_enabled and st.session_state.get("developer_payload"):
    st.divider()
    st.subheader("Developer details")
    for index, payload in enumerate(st.session_state["developer_payload"], start=1):
        st.markdown(f"**Interaction {index}**")
        with st.expander("Model input", expanded=False):
            st.json(payload.get("model_input", []))
        stream_events = payload.get("stream_events", [])
        if stream_events:
            st.markdown("Stream trace")
            for event_index, event in enumerate(stream_events, start=1):
                label = event.get("type", f"event {event_index}")
                with st.expander(f"Event {event_index}: {label}", expanded=False):
                    st.json(event)
        router_decision = payload.get("router_decision")
        retrieved_docs = payload.get("retrieved_docs")
        if router_decision or retrieved_docs:
            st.markdown("Retrieval summary")
            if router_decision:
                st.write(f"Router decision: **{router_decision}**")
            if retrieved_docs:
                with st.expander("Retrieved documents", expanded=False):
                    st.json(retrieved_docs)
        grader_info = payload.get("grader")
        if grader_info:
            st.markdown("Grader details")
            prompt_messages = grader_info.get("messages")
            if prompt_messages:
                with st.expander("Grader prompt", expanded=False):
                    st.json(prompt_messages)
            parsed_scores = grader_info.get("parsed_scores")
            if parsed_scores:
                st.json(parsed_scores)
            reasoning_traces = grader_info.get("reasoning_traces")
            if reasoning_traces:
                with st.expander("Reasoning traces", expanded=True):
                    if isinstance(reasoning_traces, (dict, list)):
                        st.json(reasoning_traces)
                    else:
                        st.code(str(reasoning_traces))
            additional_kwargs = grader_info.get("additional_kwargs")
            if additional_kwargs:
                with st.expander("Additional kwargs", expanded=False):
                    st.json(additional_kwargs)
            response_metadata = grader_info.get("response_metadata")
            if response_metadata:
                with st.expander("Response metadata", expanded=False):
                    st.json(response_metadata)
            raw_response = grader_info.get("raw_response")
            if raw_response:
                st.code(raw_response, language="json")
        with st.expander("Stored chat state", expanded=False):
            st.json(payload.get("stored_state", []))
        configuration = payload.get("configuration")
        if configuration:
            with st.expander("Ablation configuration", expanded=False):
                st.json(configuration)
