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
if "prompt_input" not in st.session_state:
    st.session_state["prompt_input"] = ""
if "clear_prompt" not in st.session_state:
    st.session_state["clear_prompt"] = False

# Reset the prompt field before the text area renders so Streamlit does not raise
# an error about modifying widget state post-instantiation.
if st.session_state.get("clear_prompt"):
    st.session_state["prompt_input"] = ""
    st.session_state["clear_prompt"] = False

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
            st.session_state["clear_prompt"] = True
            st.success("Conversation cleared.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Failed to reset chat: {exc}")

user_prompt = st.text_area(
    "Ask the planner a question",
    placeholder="e.g. What is the timetable for DSA4213 in semester 1?",
    height=120,
    key="prompt_input",
)
submit = st.button("Send message")

if submit and user_prompt.strip():
    try:
        response = requests.post(
            CHAT_ENDPOINT,
            json={"prompt": user_prompt, "developer_view": developer_view_enabled},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        st.session_state["messages"] = data.get("history", [])
        developer_info = data.get("developer_view")
        if developer_info:
            st.session_state["developer_payload"].append(developer_info)
        st.session_state["clear_prompt"] = True
        st.rerun()
    except requests.RequestException as exc:
        st.error(f"Failed to contact the chatbot API: {exc}")

for item in st.session_state.get("messages", []):
    role = item.get("role", "assistant")
    content = item.get("content", "")
    avatar = AVATARS.get(role, "💬")
    chat_role = role if role in ("user", "assistant") else "assistant"
    with st.chat_message(chat_role, avatar=avatar):
        st.markdown(content)

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
