# Architecture Reference

This document explains the architectural decisions behind the Learning
Accelerator multi-agent system.

---

## System overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph System                          │
│                                                             │
│  curriculum_planner → human_approval → explainer            │
│                              ↑              ↓              │
│                       (rejected)      quiz_generator        │
│                                            ↓               │
│                                      progress_coach         │
│                                       ↓        ↓           │
│                                (next topic)  (done→END)     │
└─────────────────────────────────────────────────────────────┘
         │ MCP (tools)              │ A2A (agents)
         ▼                          ▼
┌──────────────────┐    ┌─────────────────────┐
│  Filesystem MCP  │    │  Quiz Generator A2A  │  ┌──────────────────────┐
│  Memory MCP      │    │  (port 9001)         │  │ CrewAI Study Buddy   │
└──────────────────┘    └─────────────────────┘  │ A2A (port 9002)      │
                                                └──────────────────────┘
                        every LLM call
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              LiteLLM gateway (localhost:4000)                 │
│   model alias "learning-accelerator" → routing per litellm_config     │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      Ollama endpoint                          │
│   localhost:11434 or remote (ngrok/HTTPS) · qwen2.5:7b       │
└──────────────────────────────────────────────────────────────┘
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                 Langfuse (localhost:3000)                     │
│         Full traces · Token counts · Latency per agent       │
└──────────────────────────────────────────────────────────────┘
```

---

## Why LangGraph

LangGraph models the agent workflow as a directed graph where:
- **Nodes** are Python functions (agents or utilities)
- **Edges** define routing, static or conditional
- **State** is a typed `dict` shared across all nodes
- **Checkpoints** are saved after every node execution to PostgreSQL
  (SQLite fallback for development)

The alternative (a simple loop) loses all progress on crash and cannot
support human-in-the-loop approval without invasive changes. LangGraph
makes resilience and human oversight first-class primitives.

### Graph state

All agents read from and write to a single `AgentState` dict:

```python
class AgentState(TypedDict):
    messages:             Annotated[list[BaseMessage], add_messages]
    goal:                 str
    roadmap:              StudyRoadmap | None
    approved:             bool
    current_topic_index:  int
    quiz_results:         list[QuizResult]
    weak_areas:           list[str]
    session_id:           str
    study_materials_path: str
    error:                str | None
```

Nodes return **partial updates**, only the keys they changed.
LangGraph merges the update into the full state.

### Checkpointing

LangGraph's checkpoint interface is backend-agnostic, so the graph and every
agent are identical whichever store is active. The backend is chosen in
`src/graph/checkpointer.py`:

| Condition | Backend |
|---|---|
| `CHECKPOINT_DB_URL` (terminal) or `CHECKPOINT_DB_URL_UI` (web) is set | `PostgresSaver` |
| both are empty | `SqliteSaver` (legacy, single process) |

Postgres is the production path because a shared store lets several
instances of `main.py` or the Streamlit app run against the same sessions:
any instance can resume a thread another one started, and sessions survive
instance restarts.

```python
# Postgres: pool-backed saver (see graph/checkpointer.py)
pool = ConnectionPool(conninfo=dsn, min_size=0, max_size=4,
                      kwargs={"autocommit": True})
checkpointer = PostgresSaver(pool)
checkpointer.setup()
```

Two details matter:

1. **Connection pool, not a raw connection.** Streamlit runs sessions on
   threads and a psycopg3 connection is not thread-safe. The pool also
   reconnects transparently when the Postgres container restarts.
2. **`setup()` on its own autocommit connection, under an advisory lock.**
   LangGraph's migrations include `CREATE INDEX CONCURRENTLY`, which cannot
   run inside a transaction block. The session-level advisory lock
   serializes simultaneous first starts of multiple instances so they
   cannot race the table creation.

The legacy SQLite path is unchanged: a direct
`sqlite3.connect(db_path, check_same_thread=False)` passed to
`SqliteSaver(conn)`. The connection stays open for the process lifetime,
this is intentional. `check_same_thread=False` is required because
LangGraph runs node functions and checkpoint writes on different threads.

Because both backends serialize state the same way, values read back after
a checkpoint round-trip may be plain dicts (`msgpack`/JsonPlus). The
helpers in `graph/state.py` and both entry points already coerce them.

### Human-in-the-loop

`interrupt()` inside `human_approval_node` pauses the graph and
persists a checkpoint. `main.py` collects user input and resumes
with `graph.invoke(Command(resume=user_input), config=config)`.
The same mechanism works for web UIs, the only difference is how
input is collected.

```python
# streamlit_app.py: a second compiled graph for the web interface.
# Cached with @st.cache_resource because the script re-runs per interaction.
@st.cache_resource
def get_ui_graph():
    return build_graph(
        db_path="data/checkpoints_ui.db",        # SQLite fallback only
        interrupt_before=["quiz_generator"],
        dsn_env_var="CHECKPOINT_DB_URL_UI",      # Postgres when set
    )
```

---

## Web interface (Streamlit)

`streamlit_app.py` runs the identical graph with a different I/O harness.
The design goal: zero changes to any agent node or the quiz generator.

### How the UI takes over quiz I/O

`input()` blocks Streamlit, so the UI compiles a second graph instance with
`interrupt_before=["quiz_generator"]`. The graph now pauses at a known point
where the quiz has not run yet, but the Explainer has finished, so the UI
can:

1. Extract the explanation from message history
2. Generate questions directly with `generate_questions()` (the same
   function `quiz_generator_node` uses)
3. Collect and grade answers in the browser with `grade_answer()`
4. Inject the fabricated `QuizResult` into the checkpoint:

```python
ui_graph.update_state(config, {...quiz result...}, as_node="quiz_generator")
ui_graph.invoke(None, config=config)   # resumes at progress_coach
```

`update_state(..., as_node="quiz_generator")` records the result as if the
node had executed, so `invoke(None)` continues at `progress_coach` without
re-running the quiz node. The graph logic, A2A delegation, and coach loop
are literally identical to terminal mode.

### Separate checkpoint databases

The UI graph uses a separate store from the terminal graph so web and
terminal sessions cannot corrupt each other's checkpoint threads. In
Postgres mode that is two databases in the same instance:

| Interface | Env var | Database |
|---|---|---|
| Terminal (`main.py`) | `CHECKPOINT_DB_URL` | `learning_accelerator` |
| Web (`streamlit_app.py`) | `CHECKPOINT_DB_URL_UI` | `learning_accelerator_ui` |

With SQLite the equivalent split is `data/checkpoints.db` and
`data/checkpoints_ui.db`. Cross-interface resume is unavailable in both
modes because the UI graph compiles with `interrupt_before=["quiz_generator"]`
while the terminal graph does not; the thread semantics are not
interchangeable.

### UI state machine

The app is a five screen state machine (GOAL_INPUT → ROADMAP_APPROVAL →
EXPLAINING → QUIZZING → COMPLETE) driven by Streamlit session state and
`st.rerun()`. Transitions that require the graph call `invoke()` inside
spinners, or `stream()` when the Explainer's answer should render live.

### Token streaming for explanations

The two transitions that produce an explanation — roadmap approval and the
next topic after a quiz — use `graph.stream(..., stream_mode="messages")`
instead of `invoke()`. LangGraph's message stream emits an
`AIMessageChunk` per token, tagged with metadata that includes
`langgraph_node`; `ChatOpenAI` switches to its streaming path
automatically when that callback is attached, so no agent changes are
needed.

`src/graph/streaming.py` isolates the UI-relevant events:

1. Ignore chunks from nodes other than `explainer`, which filters out
   `progress_coach` output during topic transitions.
2. When a chunk carries `tool_call_chunks`, discard the accumulated text
   and discard the rest of that LLM run. This prevents tool-planning
   preamble from being rendered as the explanation.
3. Accumulate text from the final tool-free assistant run and yield it to
   the Streamlit placeholder.

The UI throttles repaints to roughly 20 FPS. When the stream ends, it calls
`ui_graph.get_state(config).values` to obtain the authoritative checkpoint
state, then reuses the existing `extract_explanation()` logic and quiz
generation. Streaming therefore changes only how the answer is displayed,
not what is stored, graded, or traced.

The terminal interface still uses `invoke()`. Streaming is a Streamlit-only
presentation feature; the graph, checkpoints, Langfuse callbacks, and the
`interrupt_before=["quiz_generator"]` contract are unchanged.

---

## Why MCP

Before MCP, every agent-tool integration was a custom adapter. MCP
standardises the interface so a tool server built once works with
any MCP-compatible client.

### MCP primitives used

| Primitive | Example | When used |
|---|---|---|
| **Tool** | `read_study_file(filename)` | Agent needs to do something |
| **Resource** | `notes://index` | Agent needs to read structured data |

### Process boundary

The MCP servers run as Python imports in development (simpler, testable)
but are designed to run as separate processes in production. The tool
functions are plain Python, switching from import to subprocess changes
one line.

### Security

`read_study_file()` resolves the requested path and verifies it falls
within `NOTES_BASE` before reading. Path traversal attempts return
an error string rather than raising, so the LLM can see what went wrong.

---

## Why A2A

MCP connects agents to tools. A2A connects agents to other agents.

The Quiz Generator is exposed as an A2A service so any agent from any
framework, LangGraph, CrewAI, Google ADK, a custom Python script.
can request quiz generation and grading without importing our code.

### A2A request flow

```
Progress Coach (LangGraph)
  1. GET /.well-known/agent-card.json  → discover capabilities
  2. POST / (JSON-RPC 2.0 message/send)
  3. Parse result from result.parts[0].text
```

### Circuit breaker pattern

Every A2A call first checks `is_quiz_service_available()`. If the
service is down, `try_a2a_quiz_delegation()` returns `None` and the
system falls back to local quiz generation. A2A is always optional,
never load-bearing.

### SDK version

Using `a2a-sdk==0.3.25` (spec v0.3.0). The v1.0 spec shipped March 2026.
The migration when the v1.0 SDK ships to PyPI is primarily enum casing
(`"submitted"` → `"TASK_STATE_SUBMITTED"`) and Agent Card URL restructuring.

---

## Why CrewAI for the Study Buddy

The Study Buddy is built with CrewAI to prove the architecture claim:
two agents from completely different frameworks can collaborate via A2A
without either knowing about the other.

CrewAI's role/goal/backstory abstraction maps naturally to a tutoring
agent. The `TopicAnalyserTool` gives the agent structured access to
topic context. The whole crew is rebuilt per-request to avoid state
leakage between sessions.

`crew.kickoff()` is synchronous, so it runs in `asyncio.to_thread()`
inside the A2A executor to avoid blocking the event loop.

---

## Observability

Langfuse integrates via LangChain's callback system. One handler
attached to `graph.invoke()` captures every agent node, LLM call,
and tool call automatically:

```python
config = {
    "configurable": {"thread_id": session_id},
    "callbacks": [langfuse_handler],
}
graph.invoke(state, config=config)
```

`flush_langfuse()` at process exit ensures all async traces are sent
before the process terminates.

---

## Testing strategy

| Tier | Command | Speed | Dependencies |
|---|---|---|---|
| Unit | `pytest tests/ -m "not eval"` | ~5s | None (gateway mocked) |
| Eval | `pytest tests/test_eval.py -m eval` | ~90s | LiteLLM gateway |

Unit tests mock all LLM calls and test parsing, routing, and tool
logic deterministically. Eval tests run the actual agents and use
LLM-as-judge (via DeepEval through the LiteLLM gateway) to score
output quality.

### Key test patterns

- **Agent node tests**: mock the LLM factory (`agents.*.build_llm`), test state transformation
- **Routing tests**: call routing functions directly with crafted state
- **MCP tests**: call tool functions directly (no subprocess needed)
- **A2A tests**: mock `httpx` for all network calls
- **Eval tests**: run real agents, score with `FaithfulnessMetric` / `GEval`

---

## Data flow: one complete session

```
1. User: "Learn Python closures"
   → initial_state() creates AgentState

2. curriculum_planner_node(state)
   → LLM call: goal → StudyRoadmap JSON
   → state["roadmap"] = StudyRoadmap(5 topics)

3. interrupt() in human_approval_node
   → graph pauses, checkpoint saved
   → main.py shows roadmap, collects "yes"
   → graph.invoke(Command(resume="yes"))
   → state["approved"] = True

4. explainer_node(state)  [repeats per topic]
   → MCP: list_study_files() → ["closures.md", ...]
   → MCP: search_notes("closures") → matches
   → MCP: read_study_file("closures.md") → content
   → LLM: produces grounded explanation
   → MCP: memory_set(session_id, "explained_topics", ...)
   → Streamlit: tokens arrive through stream_mode="messages"

5. quiz_generator_node(state)
   → LLM: generates 3 questions
   → input(): collects user answers
   → LLM × 3: grades each answer
   → state["quiz_results"].append(QuizResult)

6. progress_coach_node(state)
   → LLM: coaching message
   → MCP: memory_set(session_id, "progress_topic_0", ...)
   → if score < 0.5: A2A → CrewAI Study Buddy
   → state["current_topic_index"] += 1

7. route_after_coach(state)
   → index < len(topics): → "explainer" (loop)
   → index >= len(topics): → END
```

---

## Production considerations

See `docs/MODEL_SELECTION.md` for model recommendations by hardware.

For deploying beyond local development:
- ~~Replace SQLite checkpoints with a shared store~~ **Done** — checkpoints
  use PostgreSQL (`PostgresSaver`) when `CHECKPOINT_DB_URL*` are set, so
  multiple instances share sessions; SQLite remains a single-process
  fallback
- Replace `InMemoryTaskStore` with a persistent store (PostgreSQL)
- Replace `MemoryServer._store` dict with Redis
- ~~Add authentication to A2A services~~ **Done** — auth is implemented
  (see the next section)
- Run MCP servers as separate processes via stdio transport
- Use LiteLLM's built-in rate limiting and budget checks (enable them in
  `litellm_config.yaml`; the gateway already runs from
  `docker-compose.gateway.yml`)
- Add structured logging to replace print statements

## A2A authentication

Both A2A services speak the same security dialect, defined once in
`src/a2a_services/auth.py`:

1. **Declaration** — each Agent Card carries a standard OpenAPI-style
   `securitySchemes` entry (HTTP `bearer` scheme) plus `security: [bearer]`.
   The card stays publicly readable: A2A clients fetch it first to discover
   which credential to present.
2. **Enforcement** — a Starlette middleware (`BearerAuthMiddleware`) is
   attached to both services in their `create_*_server()` factories. While
   `A2A_BEARER_TOKEN` is unset the middleware is a pass-through, so local
   development behaves exactly as before. With a token set, `POST /`
   (JSON-RPC) requires `Authorization: Bearer <token>`; failures get a 401
   with a `WWW-Authenticate: Bearer` header. Tokens compare under
   `secrets.compare_digest` (not timing-attackable) and the well-known card
   path (plus any extra paths listed in `A2A_PUBLIC_PATHS`) stays public.
3. **Client side** — `a2a_client.py` reads the same `A2A_BEARER_TOKEN` and
   attaches the header to discovery, task submissions, and health checks.
   401/403 responses surface as the usual error dict instead of raising.

Because the credential travels in the plain HTTP `Authorization` header
(RFC 7235), any A2A-compatible caller — including third-party ones — works
from the Agent Card declaration without our client code.

For mTLS instead of (or in addition to) bearer tokens, the services support
it at the uvicorn layer (`uvicorn.run(..., ssl_certfile=..., ssl_keyfile=...)`
plus client-policy verification). That is deployment configuration, not
application code, and a documented follow-up.

## The LLM gateway layer

Every agent builds its chat model through one factory: `src/llm/client.py`.
The factory returns an OpenAI-compatible `ChatOpenAI` pointing at the
LiteLLM gateway (`LITELLM_BASE_URL`), using the alias registered in
`litellm_config.yaml` (`LITELLM_MODEL`).

Why a gateway around Ollama instead of calling Ollama directly:

1. **One config surface for inference**: model, provider, fallbacks live in
   YAML. Changing Ollama model or switching providers never touches agent
   code.
2. **Fallback routing**: uncommenting the example entries in
   `litellm_config.yaml` gives GPT-4 → Claude → local Ollama failover; the
   agents cannot see the difference.
3. **Cross-provider compatibility shim**: LiteLLM translates
   `response_format` (JSON mode), tool-call schemas, and unsupported
   parameters (`drop_params: true`) per provider.

CrewAI, in the Study Buddy service, uses its own LLM client class, which
is configured to the same gateway with the `openai/<alias>` model prefix —
same protocol, same routing. CrewAI can only resolve such aliases with the
litellm SDK installed; that light package stays in the venv (the gateway
server itself never runs from the venv — it is the
`docker-compose.gateway.yml` container).
