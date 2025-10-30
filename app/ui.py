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
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"prompt": user_prompt, "developer_view": developer_view_enabled},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        api_history = data.get("history", [])
        st.session_state["messages"] = _merge_history(st.session_state.get("messages", []), api_history)
        developer_info = data.get("developer_view")
        if developer_info:
            st.session_state["developer_payload"].append(developer_info)
        st.rerun()
    except requests.RequestException as exc:
        st.error(f"Failed to contact the chatbot API: {exc}")

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
        with st.expander("Stored chat state", expanded=False):
            st.json(payload.get("stored_state", []))
