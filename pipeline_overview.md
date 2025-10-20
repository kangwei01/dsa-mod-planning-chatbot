# DSA Module Planning Assistant — Pipeline Overview

This document explains in detail the end-to-end flow implemented in `dsa_chatbot.ipynb`.

---

## 1. Notebook Structure at a Glance

The notebook is organised into the following high-level sections:

1. **Imports & Global Configuration** — standard library utilities, third-party
   dependencies, and logging configuration.
2. **`NusModsClient`** — a lightweight HTTP client for the NUSMods v2 REST API
   with local caching.
3. **LangChain Tool Definitions** — tool-wrapped functions that expose the
   client methods to the LLM.
4. **Model & Prompt Setup** — the Ollama-hosted `qwen3:14b` chat model,
   system prompt, and tool binding.
5. **LangGraph Orchestration** — a two-node graph that loops between the model
   and tool execution until an answer is produced.
6. **Notebook Helper Utilities** — convenience functions (`ask`, `reset_chat`,
   etc.) for running interactive traces inside the notebook.
7. **Example Runs** — sample prompts that demonstrate the pipeline in action.

Each section is described in detail below, following the same order the notebook
executes its cells.

---

## 2. Environment, Imports, and Logging

The first code cell configures the notebook runtime:

- **Standard library**: `logging` for diagnostics and `typing` helpers for
  readability.
- **Third-party packages**: `requests` for HTTP calls, LangChain core message
  and tool primitives, `ChatOllama` for interacting with a locally hosted Ollama
  model, and LangGraph constructs (`StateGraph`, `ToolNode`, `tools_condition`).
- **Logging**: a module-level logger named `nusmods` is initialised with
  `logging.basicConfig(level=logging.INFO)` so HTTP requests and cache behaviour
  can be inspected during debugging sessions.

If you are running this notebook from scratch, ensure the Python environment has
`requests`, `langchain-core`, `langchain-ollama`, and `langgraph` installed, and
that an Ollama server is running with the `qwen3:14b` model pulled.

---

## 3. NUSMods API Client (`NusModsClient`)

The `NusModsClient` class encapsulates all direct communication with the
NUSMods public API:

- **Construction**: initialises a persistent `requests.Session`, sets the
  default academic year to `2025-2026`, and prepares dictionaries to cache
  module metadata, module lists, and timetables.
- **Key methods**:
  - `module(module_code, acad_year=None)`: fetches the full module JSON payload
    (including description, workload, prerequisites) and caches the response per
    academic year.
  - `module_list(acad_year=None)`: downloads the entire module catalogue as a
    list, storing it in memory to support repeated searches.
  - `search_modules(query, acad_year=None, level=None, limit=10)`: performs a
    simple keyword search over module codes and titles, optionally filtering by
    the module level digit (`1`–`4`).
  - `module_timetable(module_code, acad_year=None, semester=None)`: retrieves the
    timetable blocks for each semester; supports optional semester filtering and
    caches per `(AY, module)` pair.
- **Internal helpers**: `_year_base` builds the base URL for a given academic
  year, and `_normalise_code` sanitises module codes and enforces uppercase
  strings.

By consolidating networking logic and memoisation here, the rest of the pipeline
interacts with NUSMods through simple, predictable method calls.

---

## 4. LangChain Tool Layer

Each `NusModsClient` method is exposed to the language model through LangChain
`@tool` decorators. Tools provide structured interfaces that the LLM can invoke
via function-calling. The notebook defines four tools:

1. **`nusmods_module_overview`** — wraps `client.module` to fetch high-level
   metadata while omitting the large `semesterData` block to keep responses
   compact.
2. **`nusmods_module_prerequisites`** — returns prerequisite and fulfilment
   information, helping the agent answer eligibility questions.
3. **`nusmods_module_timetable`** — wraps `client.module_timetable`, offering
   optional semester filtering for scheduling queries.
4. **`nusmods_module_search`** — interfaces with `client.search_modules` for
   discovery prompts. The tool allows specifying `level` and `limit` arguments.

Each tool includes a descriptive docstring explaining when to use it, the input
schema (arguments and types), and the shape of the returned data. These
descriptions are critical: LangChain passes them to the LLM so it can plan
appropriate tool calls.

---

## 5. Model Initialisation and System Prompt

With the tools defined, the notebook configures the conversational agent:

- **Model**: `ChatOllama` points to the locally running Ollama server and loads
  the `qwen3:14b` model in reasoning mode. Decoding parameters favour
  determinism (`temperature=0.2`, `top_p=0.9`, `num_predict=-1`).
- **Tool binding**: `model.bind_tools([...])` produces `tool_llm`, a wrapper that
  automatically exposes the tool call interface to the model.
- **System prompt**: a multi-paragraph string that instructs the assistant to:
  - break complex requests into smaller sub-tasks and plan tool usage,
  - reference previous conversation turns before responding,
  - request clarification when a prompt is ambiguous or lacks critical details,
  - ground answers in tool outputs and redirect unrelated queries.

This configuration ensures the LLM has both the capability (tools) and guidance
(prompt) to operate as a reliable planning assistant.

---

## 6. LangGraph Conversation Flow

LangGraph orchestrates the iterative exchange between the LLM and the tool
execution layer. The notebook builds a minimal graph with two nodes:

1. **`assistant` node** — invokes `tool_llm` with the current message history and
   yields either a final answer or one or more tool calls.
2. **`tools` node** — implemented via `ToolNode(tools)`; it receives tool call
   requests, executes the corresponding Python functions, and returns the results
   as `ToolMessage` objects.

Edges connect the nodes as follows:

- `START` → `assistant`
- `assistant` → `tools` (conditional)
- `assistant` → `END` (conditional)
- `tools` → `assistant`

The conditional routing is handled by `tools_condition`: if the assistant emits
tool calls, execution transitions to the `tools` node; otherwise, the graph
terminates and returns the assistant's final reply. This design keeps the agent
loop concise while providing full support for multi-tool reasoning.

A global `chat_state` dictionary stores the conversation history under the
`"messages"` key. Each turn through the graph updates this shared state so the
assistant can maintain context across user prompts.

---

## 7. Conversation Helpers inside the Notebook

To interact with the graph from notebook cells, several helper functions are
defined:

- **`_trim(messages, max_turns=5)`** — keeps only the latest five
  user/assistant pairs to prevent the context window from growing indefinitely.
- **`_condense_history(messages)`** — collapses intermediate tool messages so the
  stored chat history contains only alternating human and AI messages (with the
  most recent assistant response preserved).
- **`_msg_type`, `_msg_text`, `_msg_metadata`** — formatting utilities used when
  printing traces and developer diagnostics.
- **`reset_chat()`** — resets `chat_state` to an empty history; handy when
  starting fresh experiments.
- **`ask(prompt, show_trace=True, developer_view=False)`** — the primary entry
  point for sending a user message through the graph. It:
  1. Appends the human message to the current history (after trimming).
  2. Optionally prints a **Developer View**, which dumps the structured
     `messages` payload (role, content, and metadata for every turn), the
     formatted tool call arguments the LLM is about to receive, and the cached
     `chat_state` after the run. This verbose output is invaluable when you need
     to verify that planning instructions are present in the prompt, confirm the
     agent is emitting the expected tool schema, or spot hallucinated arguments
     before they are executed.
  3. Streams the LangGraph execution trace, printing every intermediate message
     (tool invocations and responses) in chronological order.
  4. Updates `chat_state` with the condensed history and prints the final AI
     response.

Together, these helpers make the notebook a useful playground for debugging
prompting issues, observing tool usage, and demonstrating the agent to
stakeholders without a dedicated UI.

---

## 8. Example Interaction Flow

The final cells showcase how to exercise the chatbot:

1. **Reset state** with `reset_chat()` to clear any previous conversation.
2. **Complex query** — `ask("What is the timetable for DSA4213 in sem 1 like? What are its prerequisites", developer_view=True)`:
   - Demonstrates multi-step planning where the assistant typically calls both
     the timetable and prerequisite tools.
   - `developer_view=True` prints the model inputs, tool call metadata, and the
     final stored chat state for inspection.
3. **Follow-up query** — `ask("What about for DSA3101?", developer_view=True)`
   shows how the assistant reuses recent history to answer a related question.
4. **Irrelevant prompt** — `ask("Whats the weather today?", developer_view=True)`
   confirms the guardrail behaviour described in the system prompt.

These examples double as smoke tests: running them after changes ensures the
pipeline still issues tool calls correctly, handles context trimming, and applies
its guardrails.

---

## 9. Extending the Pipeline

When extending the assistant (e.g. adding retrieval-augmented responses or new
NUS data sources), follow these guidelines:

- **Add new client capabilities** inside `NusModsClient` or create sibling
  helper classes if the data source is unrelated to NUSMods.
- **Wrap capabilities in LangChain tools** with clear descriptions and typed
  signatures so the LLM can discover them.
- **Update the system prompt** to instruct the assistant on when to use new
  tools.
- **Expand the LangGraph** if more complex control flow is required (e.g. custom
  routing or guard nodes).
- **Write notebook demonstrations** that exercise the new behaviour so future
  developers can see the full loop in action.

Understanding this flow will help you confidently modify `dsa_chatbot.ipynb`,
integrate additional planning features, and diagnose issues across the HTTP,
agent, and orchestration layers.
