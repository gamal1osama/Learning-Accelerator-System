# Notes Helper Agent

A multi agent learning system that plans a curriculum, teaches from your own
notes, quizzes you, and adapts based on how well you actually understand the
material. Everything runs locally through Ollama. Nothing about you leaves
your machine except the LLM inference endpoint you choose.

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
|                          main.py                            |
|                 (entry point, interrupt loop)               |
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

## Requirements

* Python 3.11 or newer
* Docker (optional, for the Langfuse observability stack)
* An Ollama endpoint. Local (`http://localhost:11434`) or remote over HTTPS

## Quickstart

```bash
make setup
```

Then edit `.env` and set your Ollama endpoint. Example for a local Ollama:

```env
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

Pull the model and start:

```bash
ollama pull qwen2.5:7b

make langfuse    # optional but recommended
make services    # start both A2A services in the background
make run
```

The program asks for a study goal, shows a proposed roadmap, waits for your
approval, then teaches, quizzes, and coaches you topic by topic. You can exit
at any point and resume later with `make resume SESSION=<session id>`.

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
| `OLLAMA_MODEL` | Model name, e.g. `qwen2.5:7b` |
| `OLLAMA_BASE_URL` | Ollama endpoint, local or ngrok or LAN |
| `CHECKPOINT_DB` | SQLite checkpoint database path |
| `NOTES_PATH` | Directory with your markdown notes |
| `LANGFUSE_*` | Langfuse keys. Leave empty to disable tracing |
| `QUIZ_SERVICE_URL` / `STUDY_BUDDY_URL` | A2A endpoints |
| `USE_A2A_QUIZ` / `USE_STUDY_BUDDY` | A2A feature toggles |

Put your own notes in the `NOTES_PATH` directory (sample notes on Python
closures, decorators, and basics are included). The Explainer searches these
files to ground its explanations in your material.

## Make targets

| Target | Description |
|---|---|
| `make setup` | Create venv, install dependencies, copy `.env` |
| `make run` | Run the main application |
| `make run-goal GOAL='Learn async Python'` | Run with a custom goal |
| `make resume SESSION=abc12345` | Resume a saved session |
| `make services` | Start both A2A services in the background |
| `make stop` | Stop background A2A services |
| `make langfuse` | Start the Langfuse docker stack |
| `make test` | Fast unit tests (LLM backed tests excluded) |
| `make eval` | LLM as judge evaluation suite (~90 seconds) |
| `make clean` | Remove checkpoints, pycache, and pytest cache |
| `make help` | Full target list |

## Testing

```bash
make test     # fast unit tests, no Ollama required
make eval     # evaluation suite, requires Ollama
```

The test suite covers A2A client behavior (mocked HTTP), Agent Card
structure, CrewAI crew construction, checkpoint persistence, MCP servers,
graph state transitions, and the full agent stack. Evaluation tests measure
generation quality with an LLM judge.

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
  mcp_servers/         Filesystem and memory MCP tool servers
  observability/       Langfuse configuration
study_materials/       Your notes (markdown). Samples included
tests/                 Unit and evaluation test suites
main.py                Entry point and interrupt resume loop
Makefile               One command targets for everything above
```

## Troubleshooting

**`make services` fails with Terminated**
Older pattern matching in the recipe self killed the shell. This repo uses
bracketed patterns; make sure your checkout is current.

**Curriculum planning is slow**
Each LLM call goes to Ollama. On a remote endpoint such as ngrok, expect
several seconds per call. A local GPU makes this much faster.

**A2A calls show "service not available"**
That is the graceful fallback working. Start the services with
`make services`, or set the feature flags to `false` to skip A2A entirely.

**Study Buddy responds but looks generic**
The CrewAI agent uses the same Ollama model. Better models, for example
`qwen2.5:14b` or `qwen2.5-coder:32b`, produce noticeably richer tutoring.

## License

MIT
