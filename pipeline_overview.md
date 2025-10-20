# DSA Module Planning Assistant Pipeline

This document walks through the end-to-end flow implemented in `dsa_chatbot.ipynb`,
showing how a student's prompt is processed, which components are involved, and
where NUSMods data is fetched.

## 1. Environment and Dependencies

The notebook first imports Python standard library utilities (typing helpers,
logging) and third-party packages (Requests, LangChain, LangGraph, Ollama). This
ensures HTTP access to the NUSMods API and provides the agent framework that the
planning assistant relies on.

## 2. NUSMods API Client

`NusModsClient` encapsulates the REST calls to the public NUSMods v2 API. It
defaults to academic year **2025-2026** so the latest catalogue is used unless a
caller overrides it, and provides:

- `module` for module detail payloads.
- `module_list` to cache the module catalogue for search.
- `module_timetable` for semester-specific timetable data.
- `search_modules` for simple keyword-level filtering.

The client centralises caching, session reuse, and input normalisation so the rest
of the notebook can call concise helper methods without duplicating networking
logic.

## 3. LangChain Tool Definitions

Four LangChain tools wrap the client methods and expose LLM-friendly schemas:

1. **`nusmods_module_overview`** — Fetches module metadata for general inquiries.
2. **`nusmods_module_prerequisites`** — Focuses on prerequisite and fulfilment
   relationships for eligibility checking.
3. **`nusmods_module_timetable`** — Returns semester-level timetable structures
   (optionally filtered by semester) for scheduling questions.
4. **`nusmods_module_search`** — Performs cached keyword search across the module
   list, with an optional level constraint, for discovery-style prompts.

Each tool provides a detailed description, input contract, and output schema
(embedded in its docstring) so the LLM can pick the right tool and correctly
interpret the resulting payload.

## 4. LLM Initialisation and Tool Binding

An Ollama-hosted `qwen3:14b` model is instantiated via `ChatOllama` with
reasoning mode enabled and conservative decoding parameters for deterministic
answers. The LangChain `bind_tools` method associates the tools above with the
model, generating the function-call interface the agent uses. A refreshed system
prompt now:

- Emphasises multi-step planning, including breaking complex questions into
  sub-tasks and orchestrating multiple tool calls.
- Reminds the assistant to consult prior chat history before responding so
  follow-up questions stay consistent.
- Directs the assistant to ask the student for clarification when the request is
  ambiguous or missing critical details before committing to a tool plan.
- Reinforces guardrails to keep responses grounded in the tools and to redirect
  off-topic queries.

## 5. LangGraph Conversation Loop

A two-node LangGraph state machine orchestrates each turn:

- The **assistant node** invokes the tool-enabled LLM with the accumulated
  message history (system prompt + conversation to date).
- The **tool node** executes any tool calls emitted by the LLM and feeds the
  results back as `ToolMessage` objects.

Conditional edges (`tools_condition`) keep the loop running until the LLM
responds with a final answer instead of another tool call.

## 6. Notebook Chat Helpers

Helper functions manage conversational state within the notebook environment:

- `ask` adds a `HumanMessage`, streams the LangGraph trace (showing tool calls and
  responses), and prints the latest assistant reply. A `developer_view` toggle
  can be enabled per turn to surface the exact messages shown to the LLM, its
  tool-call metadata, and the stored chat history for easier debugging.
- `reset_chat` clears the shared chat history so repeated experiments start fresh.

These utilities demonstrate how the planning assistant can be exercised without
building a separate UI.

## 7. Example Interactions

The final notebook cells provide sample prompts to validate the workflow:

1. Module overview for `CS3244`.
2. Timetable lookup for `ST3131` in Semester 1.
3. Level-3000 module search focused on data-related keywords.

Running the cells showcases the streaming trace and confirms the tools are wired
correctly end-to-end.
