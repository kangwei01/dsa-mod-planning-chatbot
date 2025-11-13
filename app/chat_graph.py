"""Chat orchestration utilities that mirror the original notebook logic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .retrieval import combine_context, format_documents, get_retriever
from .tools import API_TOOLS

# System prompt reproduced from the notebook so behaviour remains familiar.
DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are an academic planning assistant for the NUS Data Science & Analytics major. "
    "Always review the full chat history so follow-up questions stay consistent. "
    "Use a private chain-of-thought to break complex requests into sub-questions, plan the tool-call sequence, and call multiple tools when needed before answering. "
    "If a student's question is ambiguous or missing critical details, ask for clarification before committing to a tool plan. "
    "If a student's question does not specify an Academic year, assume the current: 2025-2026."
    "Ground every module fact in the provided NUSMods API tools and cross-check conflicting data. "
    "If a question falls outside academic planning, politely steer the student back to relevant topics. "
    "If a module cannot be located, apologise and suggest verifying the code or academic year, and if the tools cannot answer, explain the limitation instead of guessing. "
    "If external information have been retrieved, make use of those information."
)


def _build_system_message(template: str) -> SystemMessage:
    """Create the ``SystemMessage`` from the configured template string."""

    return SystemMessage(content=template)


class ChatState(MessagesState):
    """Extended LangGraph state that tracks retrieval decisions."""

    retrieved_docs: Optional[List[Document]] = None
    router_decision: Optional[str] = None
    router_query: Optional[str] = None


def _trim(messages: Iterable[Any], max_history: int) -> List[Any]:
    """Limit the stored history to the most recent ``max_history`` turns."""

    max_messages = max_history * 2
    if max_messages <= 0:
        return []
    seq = list(messages)
    if len(seq) <= max_messages:
        return seq
    return seq[-max_messages:]


def _condense_history(messages: Iterable[Any], max_history: int) -> List[Any]:
    """Condense prior turns to just the human prompt and final assistant reply."""

    condensed: List[Any] = []
    pending_human: Optional[HumanMessage] = None
    for message in messages:
        if isinstance(message, HumanMessage):
            pending_human = HumanMessage(content=getattr(message, "content", str(message)))
        elif isinstance(message, AIMessage):
            ai_message = AIMessage(content=getattr(message, "content", str(message)))
            if pending_human is not None:
                condensed.extend([pending_human, ai_message])
                pending_human = None
            else:
                condensed.append(ai_message)
    if pending_human is not None:
        condensed.append(pending_human)
    return _trim(condensed, max_history)


def _msg_type(message: Any) -> str:
    """Pretty-print helper for LangChain message objects."""

    return getattr(message, "type", message.__class__.__name__).upper()


def _msg_text(message: Any) -> str:
    """Extract message body, handling ``ToolMessage`` payload differences."""

    if isinstance(message, ToolMessage):
        return str(message.content)
    return getattr(message, "content", str(message))


def _msg_metadata(message: Any) -> Dict[str, Any]:
    """Collect auxiliary metadata attached to LangChain messages."""

    metadata: Dict[str, Any] = {}
    additional = getattr(message, "additional_kwargs", None)
    if additional:
        metadata["additional_kwargs"] = additional
    response_meta = getattr(message, "response_metadata", None)
    if response_meta:
        metadata["response_metadata"] = response_meta
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        metadata["tool_calls"] = tool_calls
    return metadata


def _safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce metadata values into JSON-serialisable primitives."""

    safe: Dict[str, Any] = {}
    for key, value in metadata.items():
        try:
            json.dumps(value)
            safe[key] = value
        except TypeError:
            safe[key] = str(value)
    return safe


def _serialise_message(message: Any) -> Dict[str, Any]:
    """Represent LangChain messages as JSON-friendly dictionaries."""

    metadata = _safe_metadata(_msg_metadata(message))
    return {
        "type": _msg_type(message),
        "content": _msg_text(message),
        "metadata": metadata,
    }


def _role_for_message(message: Any) -> str:
    """Map LangChain message classes to chat bubble roles."""

    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return "tool"
    return _msg_type(message).lower()


@dataclass
class ChatResponse:
    """Structured response returned by :class:`ChatService`."""

    answer: str
    history: List[Dict[str, Any]]
    developer_view: Optional[Dict[str, Any]] = None


class ChatService:
    """Stateful service that mirrors the behaviour of the original notebook."""

    def __init__(
        self,
        max_history: int = 5,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        reasoning_enabled: bool = False,
        retriever_enabled: bool = True,
    ) -> None:
        self.max_history = max_history
        self.chat_state: Dict[str, Any] = {"messages": []}
        self.system_prompt_template = system_prompt_template
        self.reasoning_enabled = reasoning_enabled
        self.retriever_enabled = retriever_enabled
        self._system_message = _build_system_message(self.system_prompt_template)
        self.graph = self._build_graph()

    def clone(
        self,
        *,
        system_prompt_template: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
        retriever_enabled: Optional[bool] = None,
    ) -> "ChatService":
        """Return a new :class:`ChatService` instance with matching configuration."""

        return ChatService(
            max_history=self.max_history,
            system_prompt_template=(
                system_prompt_template if system_prompt_template is not None else self.system_prompt_template
            ),
            reasoning_enabled=(
                bool(reasoning_enabled)
                if reasoning_enabled is not None
                else self.reasoning_enabled
            ),
            retriever_enabled=(
                bool(retriever_enabled)
                if retriever_enabled is not None
                else self.retriever_enabled
            ),
        )

    def configure(
        self,
        *,
        system_prompt_template: Optional[str] = None,
        reasoning_enabled: Optional[bool] = None,
        retriever_enabled: Optional[bool] = None,
    ) -> None:
        """Update runtime configuration used for ablation experiments."""

        if system_prompt_template is not None:
            candidate = system_prompt_template.strip()
            new_template = (
                system_prompt_template if candidate else self.system_prompt_template
            )
        else:
            new_template = self.system_prompt_template
        new_reasoning = (
            self.reasoning_enabled if reasoning_enabled is None else bool(reasoning_enabled)
        )
        new_retriever = (
            self.retriever_enabled if retriever_enabled is None else bool(retriever_enabled)
        )

        if (
            new_template == self.system_prompt_template
            and new_reasoning == self.reasoning_enabled
            and new_retriever == self.retriever_enabled
        ):
            return

        self.system_prompt_template = new_template
        self.reasoning_enabled = new_reasoning
        self.retriever_enabled = new_retriever
        self._system_message = _build_system_message(self.system_prompt_template)
        self.graph = self._build_graph()

    def _build_graph(self):
        """Create the LangGraph state machine with the registered tools."""

        llm = ChatOllama(
            model="qwen3:14b",
            temperature=0.2,
            num_predict=-1,
            reasoning=self.reasoning_enabled,
            validate_model_on_init=True,
        )
        llm_with_tools = llm.bind_tools(API_TOOLS)

        retriever = get_retriever() if self.retriever_enabled else None
        system_message = self._system_message

        def retrieve_node(state: ChatState) -> Dict[str, Any]:
            """Fetch supporting context before the assistant reasons."""

            if retriever is None:
                return {"retrieved_docs": []}

            messages = state.get("messages", [])
            if not messages:
                return {"retrieved_docs": []}

            query = state.get("router_query") or ""
            if not query:
                query_msg = messages[-1]
                query = getattr(query_msg, "content", "")
            if not query:
                return {"retrieved_docs": []}

            docs = retriever.invoke(query)
            return {"retrieved_docs": docs}

        def assistant_node(state: ChatState) -> Dict[str, Any]:
            """Invoke the LLM with optional retrieved context."""

            history = list(state.get("messages", []))
            if (
                self.retriever_enabled
                and history
                and isinstance(history[-1], HumanMessage)
            ):
                context = combine_context(state.get("retrieved_docs"))
                if context:
                    last_user = history[-1]
                    history[-1] = HumanMessage(
                        content=f"Context:\n{context}\n\nUser: {last_user.content}",
                        additional_kwargs=getattr(last_user, "additional_kwargs", None) or {},
                        response_metadata=getattr(last_user, "response_metadata", None) or {},
                    )

            reply = llm_with_tools.invoke([system_message] + history)
            return {"messages": [reply]}

        def router_node(state: ChatState) -> Dict[str, Any]:
            """Decide whether an additional retrieval pass is required."""

            if not self.retriever_enabled:
                return {"router_decision": "assistant", "router_query": None}

            messages = state.get("messages", [])
            if not messages:
                return {"router_decision": "assistant", "router_query": None}

            query_msg = messages[-1]
            query = getattr(query_msg, "content", "")
            retrieved_texts = format_documents(state.get("retrieved_docs"))
            router_prompt = f"""
You are a routing agent helping a course-planning assistant.

The user asked: "{query}"

The assistant already has access to these retrieved documents:
{retrieved_texts if retrieved_texts else "[No documents retrieved yet]"}

DECIDE: Does the assistant need to retrieve more documents?

Choose "retrieve" if:
- Query asks about DSA major requirements, CHS curriculum, specializations, or graduation requirements
- AND the retrieved documents do NOT contain the specific information needed to answer

Choose "proceed" if:
- The retrieved documents already contain sufficient information to answer the query
- OR the query doesn't require academic requirement documents

If you choose "retrieve", ALSO craft a concise keyword-style search phrase (no more than 10 words) that best captures the user's intent for document retrieval.

Respond in JSON with two keys:
- "decision": either "retrieve" or "proceed"
- "query": the keyword search phrase when decision is "retrieve", otherwise an empty string
"""

            router_reply = llm.invoke([SystemMessage(content=router_prompt)]).content
            router_decision = "assistant"
            router_query: Optional[str] = None
            try:
                parsed = json.loads(router_reply)
                decision_value = str(parsed.get("decision", "")).strip().lower()
                if decision_value == "retrieve":
                    router_decision = "retrieve_requirements"
                router_query = str(parsed.get("query", "")).strip() or None
            except (json.JSONDecodeError, AttributeError):
                decision_value = router_reply.strip().lower()
                if "retrieve" in decision_value:
                    router_decision = "retrieve_requirements"

            return {"router_decision": router_decision, "router_query": router_query}

        builder = StateGraph(ChatState)
        builder.add_node("router", router_node)
        builder.add_node("retrieve_requirements", retrieve_node)
        builder.add_node("assistant", assistant_node)
        builder.add_node("tools", ToolNode(API_TOOLS))
        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            lambda state: state.get("router_decision"),
            {
                "retrieve_requirements": "retrieve_requirements",
                "assistant": "assistant",
            },
        )
        builder.add_edge("retrieve_requirements", "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        return builder.compile()

    def reset(self) -> None:
        """Clear any stored chat history."""

        self.chat_state = {"messages": []}

    def ask(self, prompt: str, developer_view: bool = False) -> ChatResponse:
        """Submit a user prompt and return the assistant's reply."""

        history = _trim(self.chat_state["messages"] + [HumanMessage(content=prompt)], self.max_history)
        developer_payload: Optional[Dict[str, Any]] = None
        if developer_view:
            developer_payload = {
                "model_input": [_serialise_message(msg) for msg in [self._system_message] + history],
                "configuration": {
                    "system_prompt_template": self.system_prompt_template,
                    "reasoning_enabled": self.reasoning_enabled,
                    "retriever_enabled": self.retriever_enabled,
                },
            }

        last_len = len(history)
        final_state: Optional[Dict[str, Any]] = None
        stream_events: List[Dict[str, Any]] = []
        for state in self.graph.stream({"messages": history}, stream_mode="values"):
            msgs = state.get("messages", [])
            new_msgs = msgs[last_len:]
            if developer_view and new_msgs:
                stream_events.extend(_serialise_message(msg) for msg in new_msgs)
            last_len = len(msgs)
            final_state = state

        if final_state is not None:
            condensed = _condense_history(final_state["messages"], self.max_history)
            self.chat_state = {"messages": condensed}
        else:
            # If no state was produced we preserve the previous history for safety.
            condensed = self.chat_state["messages"]

        if developer_view:
            assert developer_payload is not None
            developer_payload["stream_events"] = stream_events
            developer_payload["stored_state"] = [_serialise_message(msg) for msg in condensed]
            if final_state is not None:
                retrieved = final_state.get("retrieved_docs")
                if retrieved:
                    developer_payload["retrieved_docs"] = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                        }
                        for doc in retrieved
                    ]
                decision = final_state.get("router_decision")
                if decision:
                    developer_payload["router_decision"] = decision
                router_query = final_state.get("router_query")
                if router_query:
                    developer_payload["router_query"] = router_query

        answer = ""
        for msg in reversed(self.chat_state["messages"]):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break

        history_payload = []
        for msg in self.chat_state["messages"]:
            if not isinstance(msg, (HumanMessage, AIMessage)):
                continue

            content = _msg_text(msg)
            # Tool planning turns sometimes emit empty assistant messages that only
            # contain structured tool-call metadata. These render as blank bubbles
            # in the Streamlit chat. Skip them so the UI only shows meaningful
            # assistant responses while still preserving the underlying metadata
            # for developer view tracing.
            if isinstance(msg, AIMessage) and not content.strip():
                continue

            history_payload.append(
                {
                    "role": _role_for_message(msg),
                    "content": content,
                    "metadata": _safe_metadata(_msg_metadata(msg)),
                }
            )

        return ChatResponse(answer=answer, history=history_payload, developer_view=developer_payload)

    def evaluate(
        self,
        prompts: Iterable[str],
        *,
        developer_view: bool = False,
    ) -> List[ChatResponse]:
        """Run each prompt as an isolated chat turn and return their responses."""

        results: List[ChatResponse] = []
        for prompt in prompts:
            clean_prompt = (prompt or "").strip()
            if not clean_prompt:
                continue
            self.reset()
            results.append(self.ask(clean_prompt, developer_view=developer_view))
        return results


__all__ = [
    "ChatResponse",
    "ChatService",
    "DEFAULT_SYSTEM_PROMPT_TEMPLATE",
]
