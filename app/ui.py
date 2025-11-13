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
EVALUATE_ENDPOINT = f"{API_ROOT}/evaluate"

EVALUATION_QUERIES = [
    "List all the Level 2000 core modules for the DSA major.",
    "What are the Level 3000 modules I need to take for DSA?",
    "Do I have to do a specialisation in DSA to graduate?",
    "How many MCs do I need to graduate from the DSA programme?",
    "Explain the difference between Option A and Option B in Level 4000 requirements.",
    "What modules are in the Operations Research specialisation?",
    "List the Statistical Methodology specialisation modules.",
    "What are the CHS Common Curriculum pillars?",
    "Give examples of Communities and Engagement courses I can take.",
    "Which CHS pillar does DSA1101 fulfil?",
    "When should I take Communities and Engagement modules?",
    "When can I take the Interdisciplinary CHS courses?",
    "Plan a 20-MC semester to cover my Level 2000 DSA core.",
    "What modules should I take next semester?",
    "Book a Grab ride to NUS for me.",
    "What are the prerequisites for XYZ9999?",
    "What are the prerequisites for DSA4213?",
    "What is the timetable for DSA4213 like in Sem 1?",
]

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
if "evaluation_runs" not in st.session_state:
    st.session_state["evaluation_runs"] = []


def _record_evaluation_run(label: str, configuration: dict) -> dict:
    """Create and store a new evaluation run entry in session state."""

    run_record = {
        "label": label,
        "payload": {
            "configuration": configuration,
            "results": [],
        },
    }
    st.session_state["evaluation_runs"].insert(0, run_record)
    return run_record


def _update_run_payload(run_record: dict) -> None:
    """Persist in-memory updates for the active evaluation run."""

    if st.session_state.get("evaluation_runs"):
        st.session_state["evaluation_runs"][0] = run_record


def _run_evaluation(label: str, *, system_prompt_template: str, enable_reasoning: bool, enable_retriever: bool) -> None:
    """Execute the benchmark prompts sequentially and stream their answers."""

    total_prompts = len(EVALUATION_QUERIES)
    status_placeholder = st.empty()
    results_container = st.container()
    configuration = {
        "system_prompt_template": system_prompt_template,
        "reasoning_enabled": enable_reasoning,
        "retriever_enabled": enable_retriever,
    }
    run_record = _record_evaluation_run(label, configuration)

    for index, prompt in enumerate(EVALUATION_QUERIES, start=1):
        status_placeholder.info(f"Running prompt {index} of {total_prompts}...")
        try:
            response = requests.post(
                EVALUATE_ENDPOINT,
                json={
                    "prompts": [prompt],
                    "system_prompt_template": system_prompt_template,
                    "enable_reasoning": enable_reasoning,
                    "enable_retriever": enable_retriever,
                },
                timeout=(20, 1200),
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            timeout_message = (
                f"Evaluation timed out after 600 seconds on prompt {index}: {prompt}"
            )
            status_placeholder.error(timeout_message)
            run_record.setdefault("payload", {}).setdefault("errors", []).append(
                {
                    "prompt_index": index,
                    "prompt": prompt,
                    "error": "timeout",
                    "detail": str(exc),
                }
            )
            _update_run_payload(run_record)
            with results_container:
                st.markdown(f"**Prompt {index}:** {prompt}")
                st.error("Request timed out after 600 seconds.")
            return
        except requests.RequestException as exc:
            error_message = f"Evaluation failed on prompt {index}: {exc}"
            status_placeholder.error(error_message)
            run_record.setdefault("payload", {}).setdefault("errors", []).append(
                {
                    "prompt_index": index,
                    "prompt": prompt,
                    "error": "request",
                    "detail": str(exc),
                }
            )
            _update_run_payload(run_record)
            with results_container:
                st.markdown(f"**Prompt {index}:** {prompt}")
                st.error("Evaluation request failed before completion.")
            return

        data = response.json()
        if data.get("configuration"):
            run_record["payload"]["configuration"] = data["configuration"]

        result_items = data.get("results", [])
        result_payload = {
            "prompt": prompt,
            "answer": "",
            "history": [],
        }
        if result_items:
            result_payload.update(result_items[0])
        run_record["payload"]["results"].append(result_payload)
        _update_run_payload(run_record)

        with results_container:
            st.markdown(f"**Prompt {index}:** {prompt}")
            answer = result_payload.get("answer", "")
            if answer.strip():
                st.markdown(answer)
            else:
                st.info("No answer returned for this prompt.")
            if index < total_prompts:
                st.markdown("---")

    status_placeholder.success("Evaluation completed.")

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
    st.header("Evaluate configurations")
    st.caption("Run the benchmark prompts against specific ablation settings.")
    if st.button("Run baseline evaluation", use_container_width=True):
        _run_evaluation(
            "Baseline configuration",
            system_prompt_template=DEFAULT_SYSTEM_PROMPT_TEMPLATE,
            enable_reasoning=False,
            enable_retriever=True,
        )
    if st.button("Run evaluation with current settings", use_container_width=True):
        _run_evaluation(
            "Current ablation settings",
            system_prompt_template=st.session_state["ablation_prompt_template"],
            enable_reasoning=st.session_state["ablation_reasoning_enabled"],
            enable_retriever=st.session_state["ablation_retriever_enabled"],
        )

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
                json={
                    "prompt": user_prompt,
                    "developer_view": developer_view_enabled,
                    "system_prompt_template": st.session_state["ablation_prompt_template"],
                    "enable_reasoning": st.session_state["ablation_reasoning_enabled"],
                    "enable_retriever": st.session_state["ablation_retriever_enabled"],
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
        configuration = payload.get("configuration")
        if configuration:
            with st.expander("Ablation configuration", expanded=False):
                st.json(configuration)

st.divider()
st.subheader("Evaluation prompts")
st.caption("Use these questions to compare ablation configurations.")
with st.expander("Show evaluation prompt set", expanded=False):
    for idx, query in enumerate(EVALUATION_QUERIES, start=1):
        st.markdown(f"{idx}. {query}")

if st.session_state["evaluation_runs"]:
    st.divider()
    st.subheader("Evaluation results")
    for run_index, run in enumerate(st.session_state["evaluation_runs"], start=1):
        payload = run.get("payload", {})
        configuration = payload.get("configuration", {})
        results = payload.get("results", [])
        summary_label = run.get("label") or f"Run {run_index}"
        with st.expander(summary_label, expanded=(run_index == 1)):
            config_lines = [
                f"**System prompt**: {configuration.get('system_prompt_template', '').strip() or '[blank]'}",
                f"**Reasoning enabled**: {configuration.get('reasoning_enabled', False)}",
                f"**Retriever enabled**: {configuration.get('retriever_enabled', True)}",
            ]
            st.markdown("\n".join(config_lines))
            errors = payload.get("errors", [])
            if errors:
                for error in errors:
                    prompt_idx = error.get("prompt_index")
                    prompt_text = error.get("prompt", "")
                    if error.get("error") == "timeout":
                        st.error(
                            f"Prompt {prompt_idx} timed out after 600 seconds: {prompt_text}"
                        )
                    else:
                        st.error(
                            f"Prompt {prompt_idx} failed before completion: {prompt_text}"
                        )
            if results:
                for idx, item in enumerate(results, start=1):
                    st.markdown(f"**Prompt {idx}:** {item.get('prompt', '')}")
                    answer_text = item.get("answer", "")
                    if answer_text.strip():
                        st.markdown(answer_text)
                    else:
                        st.info("No answer returned for this prompt.")
                    if idx < len(results):
                        st.markdown("---")
            else:
                st.info("No results returned for this run.")
