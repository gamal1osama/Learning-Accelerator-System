# Notes Helper Agent

A multi agent learning system that plans a curriculum, teaches from your own
notes, quizzes you, and adapts based on how well you actually understand the
material. All inference runs through a LiteLLM gateway that routes to your
Ollama endpoint (local or remote), with optional cloud fallbacks configured
on the gateway side. Nothing about you leaves your machine except the LLM
inference endpoints you configure.

Session flow is identical across both interfaces: terminal (main.py) and web
(Streamlit). Same graph, same agents, same services.

The system is orchestrated by LangGraph and demonstrates three distinct
agentic patterns working together:

1. **LangGraph orchestration**: five specialized agents in a stateful graph
   with SQLite checkpointing and human in the loop approval
2. **Agent to Agent (A2A) protocol**: the Progress Coach delegates work to two
   independently running services over JSON RPC
3. **Cross framework interop**: one A2A service is built with LangGraph, the
   other with CrewAI. Neither knows about the other's internals

## How it works

```
+============================================================+
|                          main.py                           |
|                 (entry point, interrupt loop)              |
+============================================================+
                             |
                   LangGraph (state graph)
                             |
   +========================+=================================+
   |                         |                                |
Curriculum                Explainer                    Quiz Generator
Planner                      |                                |
   |              MCP: filesystem + memory                    |
   |                         |                                |
   +====> Human Approval ====+                          Progress Coach
           (interrupt)                                        |
                                                    +==========+===========+
                                           A2A quiz delegating      A2A study buddy
                                                    |                      |
                                              Quiz Service              CrewAI
                                           (LangGraph :9001)       Study Buddy (:9002)

Every LLM call (all agents + both A2A services) goes through:
                             |
                   LiteLLM gateway (:4000)
                             |
                        Ollama endpoint
```

Workflow per topic:

1. The **Curriculum Planner** builds a study roadmap from your goal
2. You approve or reject the plan at a **human approval** interrupt
3. The **Explainer** reads your notes through MCP filesystem tools and memory
   tools, then explains the topic
4. The **Quiz Generator** creates conceptual questions and grades your
   answers with an LLM judge
5. The **Progress Coach** records your score, updates topic status, delegates
   to the Quiz A2A service when it is available, and requests supplementary
   help from the CrewAI Study Buddy when your score falls below the pass
   threshold
6. The loop repeats for the next topic until the roadmap is complete

## Key features

| Feature | What it gives you |
|---|---|
| Adaptive roadmap | Topics generated from your goal with prerequisites and time estimates |
| Human approval | Human approval checkpoint before studying starts |
| Personal notes | The Explainer teaches from your own markdown notes via MCP tools |
| Conceptual quizzes | Questions test understanding, not memorization |
| LLM as judge grading | Each answer graded with feedback, score, and identified weak areas |
| A2A delegation | Progress Coach calls external services transparently, with graceful fallback |
| Cross framework A2A | LangGraph can call CrewAI through a protocol contract |
| Observability | Langfuse traces for every LLM call, tool call, and agent step |
| Checkpointing | Any session can be stopped and resumed with its session ID |
| Two interfaces | Full session in the terminal (`make run`) or the browser (`make streamlit`), identical graph underneath |

## Requirements

* Python 3.11 or newer
* Docker. Required for the LiteLLM gateway; optional for the Langfuse observability stack
* An Ollama endpoint. Local (`http://localhost:11434`) or remote over HTTPS

## Quickstart

```bash
make setup
```

Then edit `.env` and point the stack at your Ollama endpoint:

```env
LITELLM_BASE_URL=http://localhost:4000
LITELLM_API_KEY=sk-notes-helper-gateway-key
LITELLM_MASTER_KEY=sk-notes-helper-gateway-key
LITELLM_MODEL=notes-helper
OLLAMA_BASE_URL=http://localhost:11434
```

Start the gateway, pull the model if needed, then pick your interface:

Terminal:

```bash
ollama pull qwen2.5:7b

make gateway      # LiteLLM gateway (docker) on :4000, routes to Ollama
make langfuse     # optional but recommended
make services     # start both A2A services in the background
make run
```

Or the web interface:

```bash
make gateway
make services
make streamlit   # then open http://localhost:8501
```

The program asks for a study goal, shows a proposed roadmap, waits for your
approval, then teaches, quizzes, and coaches you topic by topic. You can exit
at any point and resume later with `make resume SESSION=<session id>`.

## How to use it

From your point of view the session is a simple terminal conversation
(the Streamlit interface mirrors every step in the browser, see
"The Streamlit web interface" below):

1. **Start the app** with a goal in natural language:

   ```bash
   make run-goal GOAL='Learn Python closures and decorators from scratch'
   ```

2. **Review the roadmap.** The system shows a study plan with topics,
   prerequisites, and time estimates, then waits for you:

   ```
   Does this study plan look good?
     Type 'yes' to start studying
     Type 'no' to generate a different plan
   > yes
   ```

   Typing `no` sends the planner back to the whiteboard. It regenerates and
   asks again until you accept something.

3. **Read the explanation.** The Explainer teaches the topic grounded in your
   notes, then the quiz starts.

4. **Answer the quiz.** Three questions appear one at a time. Type your
   answers in your own words. Press Enter to submit each one. Each answer is
   graded instantly with a score, feedback, and weak areas.

5. **Read the coaching.** After every quiz the Progress Coach prints a
   summary, an encouragement line, and (when relevant) a supplementary note
   from the CrewAI Study Buddy.

6. **Keep going or stop.** The session continues automatically through the
   remaining topics and prints a final summary. You can Ctrl+C at any time;
   progress is checkpointed after every step.

7. **Resume later** with the session ID printed at startup:

   ```bash
   make resume SESSION=8a679ee7
   ```

Your notes live in the `NOTES_PATH` directory as markdown files. Add your
own notes there and the Explainer will teach from them.

## The Streamlit web interface

Prefer a browser over the terminal? Run:

```bash
make services
make streamlit
```

Then open http://localhost:8501. The interface walks through five screens:

| Screen | What you do |
|---|---|
| GOAL_INPUT | Type what you want to learn, click "Build Study Plan" |
| ROADMAP_APPROVAL | Review topics with prerequisites and time estimates; approve or ask for a new plan |
| EXPLAINING | Read the explanation grounded in your notes, see the coach tip from the last topic, click "Start Quiz" |
| QUIZZING | Answer one question at a time in a text box; each answer is graded instantly with feedback and score; continue to the next topic when the quiz finishes |
| COMPLETE | Session summary: per-topic scores, overall average, topics to revisit, your session ID |

What stays the same as terminal mode:

* The **same LangGraph graph** with the same agents, MCP tools, A2A
  delegation, and Langfuse tracing
* The **same quiz engine**: questions generated and answers graded by
  the identical functions the terminal flow uses
* Roadmap approval via human in the loop interrupt, and the
  Progress Coach loop from topic to topic

What is different:

* Quiz answers are typed in a text area instead of the terminal
* One **graceful unfinished-topic flow**: if you refresh or close the
  browser mid session, your progress stays in the checkpoint database
  (`data/checkpoints_ui.db`), a separate file from the terminal's
  `data/checkpoints.db`, so web and terminal sessions never clobber
  each other
* Cross resume is not available between interfaces: browser sessions are
  checkpointed in `checkpoints_ui.db` while `make resume
  SESSION=<id>` reads the terminal checkpoint `checkpoints.db`. Take
  the session ID from the COMPLETE screen of a terminal session to use
  `make resume`

Nothing needs to change between interfaces: you can study one session in
the browser and finish it in the terminal, and the A2A services
(`make services`) power both. Missing services are not an error, the
Progress Coach falls back to local generation invisibly.

## How the project works

Under the hood there are four cooperating layers.

**The graph.** Every agent is a node function that receives the current state
and returns updates. The edges and conditional edges define the control flow:

| Transition | Routing rule |
|---|---|
| `START` → `curriculum_planner` | always on a new session |
| `curriculum_planner` → `human_approval` | always; toolbar shows roadmap and pauses |
| `human_approval` → `explainer` | when the user approves |
| `human_approval` → `curriculum_planner` | when the user rejects; a new plan is generated |
| `explainer` → `quiz_generator` | always; quiz receives the explanation from message history |
| `quiz_generator` → `progress_coach` | always; coach receives the graded quiz |
| `progress_coach` → `explainer` | while topics remain in the roadmap |
| `progress_coach` → `END` | when every topic has a final status |

**The state.** A single `AgentState` object holds the goal, the roadmap, the
current topic index, quiz results, accumulated weak areas, chat messages, and
the session ID. Each node reads what it needs and returns only its own
updates, which keeps agents decoupled.

**Persistence.** Every state change is checkpointed to SQLite, keyed by the
session ID used as the LangGraph thread ID. This is what makes the human
approval interrupt and the resume command possible: stopping the process
loses nothing, and resuming replays the graph from the last checkpoint.

**MCP tools.** The agents call tools through the Model Context Protocol.
Two servers are registered:

| Server | Tools |
|---|---|
| filesystem | list files, read notes, search content by keyword |
| memory | get and set session scoped key value storage |

The Explainer uses these to list your notes, search for the relevant
sections, and read the full text before explaining. It also remembers which
topics it already covered for the session.

**A2A delegation.** The Progress Coach talks to two background services over
JSON RPC. For each service it checks the Agent Card first. When the service
answers, work is delegated through the protocol; when it is down, the coach
continues locally and the user never sees an error. This is a resilience
pattern as much as an interop pattern.

## End to end walkthrough

Here is a real session, start to finish, using the default goal. The trimmings
between blocks are printed output, slightly abbreviated.

The goal and session ID:

```
============================================================
Learning Accelerator
Session ID: 8a679ee7
Goal: Learn Python closures and decorators from scratch
============================================================
```

The planner builds a roadmap and you approve it:

```
[Curriculum Planner] Created roadmap: 4 topics, 4 weeks
  1. Python Basics, 60 min
  2. Functions in Python (needs: Python Basics), 60 min
  3. Closures (needs: Functions in Python), 90 min
  4. Decorators (needs: Closures), 90 min

Does this study plan look good?
  Type 'yes' to start studying
  Type 'no' to generate a different plan
> yes
```

The Explainer teaches each topic using your notes through MCP tools:

```
[Explainer] LLM call 1/8...
  -> tool_list_files({})
    <- ["closures.md", "decorators.md", "python_basics.md"]
[Explainer] LLM call 2/8...
  -> tool_search_notes({'query': ' decorators '})
    <- [search results in decorators.md]
[Explainer] LLM call 3/8...
  -> tool_read_file({'filename': 'decorators.md'})
    <- # Python Decorators
[Explainer] Complete after 4 LLM call(s)
```

The quiz asks you three questions and grades them on the fly:

```
Question 1 [hard]: How should the following code be fixed ...
Your answer: Grading...
Score: 30%. The answer does not address ...
Areas to review: The concept of decorators metadata preservation ...
```

The Progress Coach then grades, delegates, and, when your score is under
50 percent, calls the CrewAI Study Buddy through A2A:

```
[Progress Coach] Topic: 'Decorators'
[Progress Coach] Score: 43%
[Progress Coach] Delegating quiz to A2A service: http://localhost:9001
[Progress Coach] A2A quiz complete: score=43%
[Progress Coach] Requesting study assistance from CrewAI Study Buddy...
Coach: It looks like you're making good progress with decorators, but there
are a few areas that need more attention...
You're doing great, and with a bit more practice, you'll master decorators
in no time!

📚 Study Buddy says:
Start by understanding the basics of decorators. Then focus on memory usage
measurement, preserving the original function's return value, and decorator
metadata preservation...
```

When the roadmap finishes, a summary prints:

```
============================================================
Session Summary
============================================================
Goal: Learn Python closures and decorators from scratch
Topics covered: 4/4
Average score: 58%

  ✓ Python Basics: 60%, review: ...
  ✓ Functions in Python: 63%, review: ...
  ✓ Closures: 65%, review: ...
  ✗ Decorators: 43%, review: ...
============================================================
```

Every step of that session also lands in Langfuse as a trace, so you can
inspect each LLM call, prompt, tool call, and delegation after the fact.

## The three terminal setup

You do not need this manually. `make services` handles terminals 2 and 3. It
is shown here because it explains what runs:

```bash
# Terminal 1
python main.py "Learn Python closures"

# Terminal 2
python src/a2a_services/quiz_service.py

# Terminal 3
python src/crewai_agent/study_buddy.py
```

## A2A services

Both services speak the A2A protocol and publish an Agent Card for
discovery:

```bash
curl http://localhost:9001/.well-known/agent-card.json   # Quiz Generator
curl http://localhost:9002/.well-known/agent-card.json   # CrewAI Study Buddy
```

**Quiz Generator (port 9001, LangGraph)**
Generates conceptual quiz questions and grades answers, returning scores and
weak areas. The Progress Coach delegates to it automatically, and falls back
to the local quiz generator when the service is not reachable.

**CrewAI Study Buddy (port 9002, CrewAI)**
When you score below the pass threshold, the Progress Coach requests a fresh
analogy, a targeted example, and a memory tip from this service. It is
dispatched through the exact same A2A contract even though it is built with a
different framework.

Feature flags in `.env` control delegation:

```env
USE_A2A_QUIZ=true      # set false to always use the local quiz generator
USE_STUDY_BUDDY=true   # set false to skip the CrewAI supplementary help
```

## Configuration

All configuration lives in `.env` (copy from `.env.example`):

| Variable | Purpose |
|---|---|
| `LITELLM_BASE_URL` | Gateway endpoint the agents call (default `http://localhost:4000`) |
| `LITELLM_API_KEY` | Key the app sends as Bearer. Must match the gateway master key |
| `LITELLM_MASTER_KEY` | Gateway auth key (read by the docker container via `env_file`) |
| `LITELLM_MODEL` | Model alias registered in `litellm_config.yaml` (default `notes-helper`) |
| `OLLAMA_BASE_URL` | Where the **gateway** reaches Ollama (used by `litellm_config.yaml`) |
| `CHECKPOINT_DB` | SQLite checkpoint database path |
| `NOTES_PATH` | Directory with your markdown notes |
| `LANGFUSE_*` | Langfuse keys. Leave empty to disable tracing |
| `QUIZ_SERVICE_URL` / `STUDY_BUDDY_URL` | A2A endpoints |
| `USE_A2A_QUIZ` / `USE_STUDY_BUDDY` | A2A feature toggles |

The model name itself lives in `litellm_config.yaml` and must match a model
you pulled (`ollama pull qwen2.5:7b` by default). Editing that YAML also lets
you add cloud fallbacks: change the alias list, uncomment the GPT-4/Claude
entries, restart the gateway, and every agent routes through them
transparently. Your notes live in the `NOTES_PATH` directory as markdown
files. Add your own notes there and the Explainer will teach from them.

## Make targets

| Target | Description |
|---|---|
| `make setup` | Create venv, install dependencies, copy `.env` |
| `make gateway` | Start the LiteLLM gateway container on :4000 (Docker) |
| `make gateway-stop` | Stop the LiteLLM gateway |
| `make gateway-logs` | Tail the gateway container logs |
| `make run` | Run the main application (terminal) |
| `make streamlit` | Run the web interface at http://localhost:8501 |
| `make run-goal GOAL='Learn async Python'` | Run with a custom goal |
| `make resume SESSION=abc12345` | Resume a saved session |
| `make services` | Start both A2A services in the background |
| `make stop` | Stop background A2A services |
| `make langfuse` | Start the Langfuse docker stack |
| `make test` | Fast unit tests (no LLM gateway needed) |
| `make eval` | LLM as judge evaluation suite (~90 seconds, gateway required) |
| `make clean` | Remove checkpoints, pycache, and pytest cache |
| `make help` | Full target list |

## Testing

```bash
make test     # fast unit tests, no gateway or Ollama required
make eval     # evaluation suite, requires the LiteLLM gateway
```

The test suite covers the LLM gateway factory, A2A client behavior (mocked
HTTP), Agent Card structure, CrewAI crew construction, checkpoint
persistence, MCP servers, graph state transitions, and the full agent stack.
Evaluation tests measure generation quality with an LLM judge.

## Observability

Run `make langfuse`, create a project at `http://localhost:3000`, and put the
keys in `.env`. Every session produces a full trace: curriculum planning,
explanations with tool calls, quiz generation, grading, A2A delegations, and
the Progress Coach decision. The system runs fine without Langfuse. It just
traces nothing.

## Project structure

```
src/
  agents/              LangGraph node agents (planner, explainer, quiz, coach)
  crewai_agent/        CrewAI Study Buddy, served through A2A
  a2a_services/        A2A client helpers and the Quiz service
  graph/               State definitions, workflow graph, checkpointing
  llm/                 Shared LLM client factory (LiteLLM gateway)
  mcp_servers/         Filesystem and memory MCP tool servers
  observability/       Langfuse configuration
litellm_config.yaml    Gateway model routing and fallback configuration
docker-compose.gateway.yml  LiteLLM gateway container definition
study_materials/       Your notes (markdown). Samples included
tests/                 Unit and evaluation test suites
main.py                Entry point and interrupt resume loop (terminal)
streamlit_app.py       Web interface (same graph, UI state machine)
Makefile               One command targets for everything above
```

## Troubleshooting

**`make services` fails with Terminated**
Older pattern matching in the recipe self killed the shell. This repo uses
bracketed patterns; make sure your checkout is current.

**Curriculum planning is slow**
Each LLM call travels agent → gateway → Ollama. With a remote endpoint such
as ngrok, expect several seconds per call. A local GPU makes this much
faster.

**`make run` warns the gateway is not reachable**
Start it with `make gateway`. Every agent talks to the LiteLLM gateway on
`:4000`; without it, all LLM calls fail fast with a connection error.

**Gateway returns "model not found" / 404**
`litellm_config.yaml` registers the alias your agents use (default
`notes-helper`) and maps it to an Ollama model. Make sure the Ollama model
in the config matches something you pulled.

**Requests fail against ngrok / remote Ollama**
The gateway logs which upstream endpoint it used: check
`make gateway-logs` (docker logs). Make sure `OLLAMA_BASE_URL` in `.env`
points at a live endpoint, then `make gateway-stop && make gateway` — the
container reads `.env` fresh through its `env_file` on every start.

**A2A calls show "service not available"**
That is the graceful fallback working. Start the services with
`make services`, or set the feature flags to `false` to skip A2A entirely.

**Study Buddy responds but looks generic**
The model lives in `litellm_config.yaml` via the gateway. Better models, for
example `qwen2.5:14b` or `qwen2.5-coder:32b`, produce noticeably richer
tutoring.

## License

Released under the MIT License. See the LICENSE file. Copyright (c) 2026
Gamal Osama.
