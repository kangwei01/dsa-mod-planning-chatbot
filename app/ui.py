"""Streamlit user interface for the chatbot demo."""

from __future__ import annotations

import os
import requests
import streamlit as st

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

for item in st.session_state.get("messages", []):
    role = item.get("role", "assistant")
    content = item.get("content", "")
    avatar = AVATARS.get(role, "💬")
    chat_role = role if role in ("user", "assistant") else "assistant"
    with st.chat_message(chat_role, avatar=avatar):
        st.markdown(content)

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
                json={"prompt": user_prompt, "developer_view": developer_view_enabled},
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
            for message in reversed(st.session_state["messages"]):
                if message.get("role") == "assistant":
                    assistant_reply = message.get("content", "")
                    break
            if not assistant_reply:
                assistant_reply = data.get("answer", "")
            if assistant_reply:
                message_placeholder.markdown(assistant_reply)
            else:
                message_placeholder.empty()
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
        with st.expander("Stored chat state", expanded=False):
            st.json(payload.get("stored_state", []))
