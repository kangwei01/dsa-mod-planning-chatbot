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
provides:

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

An Ollama-hosted Llama 3.1 model is instantiated via `ChatOllama`. The LangChain
`bind_tools` method associates the tools above with the model, generating the
function-call interface the agent uses. A strengthened system prompt now reminds
the assistant to review prior turns, plan multi-step queries by breaking them
into sub-tasks, and justify when external tools are unnecessary. This helps the
LLM tackle complex, follow-up heavy questions more reliably.

## 5. LangGraph Conversation Loop and Memory

A two-node LangGraph state machine orchestrates each turn:

- The **assistant node** invokes the tool-enabled LLM with the accumulated
  message history (system prompt + conversation to date).
- The **tool node** executes any tool calls emitted by the LLM and feeds the
  results back as `ToolMessage` objects.

Conditional edges (`tools_condition`) keep the loop running until the LLM
responds with a final answer instead of another tool call. The compiled graph is
wrapped in LangChain's `RunnableWithMessageHistory`, backed by
`InMemoryChatMessageHistory`, so every notebook session keeps a robust rolling
memory without manual truncation logic.

## 6. Notebook Chat Helpers and Developer View

Helper functions manage conversational state within the notebook environment:

- `ask` adds a `HumanMessage`, streams the LangGraph trace (showing tool calls and
  responses), and prints the latest assistant reply. Optional parameters allow
  callers to select a session, disable streaming, or enable a `developer_view`
  mode. When `developer_view=True`, the helper prints the conversation history,
  the exact payload sent to the LLM, and any exposed reasoning metadata for
  debugging.
- `reset_chat` clears the stored history for a given session so repeated
  experiments start fresh.
- `get_session_history` exposes the underlying LangChain history object when
  deeper inspection is required.

These utilities demonstrate how the planning assistant can be exercised without
building a separate UI while still offering rich observability during
development.

## 7. Example Interactions

The final notebook cells provide sample prompts to validate the workflow:

1. Module overview for `CS3244`.
2. Timetable lookup for `ST3131` in Semester 1.
3. Level-3000 module search focused on data-related keywords.

Running the cells showcases the streaming trace and confirms the tools are wired
correctly end-to-end.
