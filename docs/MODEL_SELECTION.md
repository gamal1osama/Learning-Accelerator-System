# Model Selection Guide

All agents in this system call the LiteLLM gateway, which runs as a docker
container (`docker-compose.gateway.yml`, started by `make gateway`) and
routes to Ollama
(the configured provider). This guide helps you choose the right model for
your hardware.

---

## Why model choice matters for agents

Agents call tools by generating structured JSON. A model that
hallucinates tool names or produces malformed JSON will fail
silently, the tool call won't execute, the agent will loop,
and you'll hit the `max_iterations` limit without understanding why.

**Minimum viable tier for reliable tool calling: 7B parameters.**
Sub-7B models work for simple chat but produce too many JSON errors
for production agentic use.

---

## Recommendations by VRAM

| VRAM | Model | Pull command | Best for |
|---|---|---|---|
| 8 GB | `qwen2.5:7b` | `ollama pull qwen2.5:7b` | General purpose, good tool calling |
| 8 GB | `qwen3:8b` | `ollama pull qwen3:8b` | Better reasoning, same VRAM |
| 24 GB | `qwen2.5-coder:32b` | `ollama pull qwen2.5-coder:32b` | Best tool calling at this tier |
| 24 GB | `qwen3:32b` | `ollama pull qwen3:32b` | Best overall at this tier |
| CPU only | `qwen2.5:7b` (Q4_K_M) | `ollama pull qwen2.5:7b` | Works, 5-10× slower |

**Check your VRAM on Mac:** Apple menu → About This Mac → chip info.
Apple Silicon unified memory is shared between CPU and GPU.
16 GB unified ≈ 8 GB available for the model.

---

## Agent-specific settings

These settings are used throughout the codebase (via `src/llm/client.py`):

```python
# Structured output (Curriculum Planner, Quiz Generator grader)
build_llm(temperature=0.1, json_mode=True)

# Tool-calling (Explainer)
build_llm(temperature=0.3)

# Creative/coaching (Progress Coach, Quiz Generator questions)
build_llm(temperature=0.4, json_mode=True)
```

Low temperature = more consistent JSON. Higher temperature = more
varied explanations and coaching messages. Never use temperature > 0.5
for any agent that produces structured output.

## Judge consistency

LLM as judge grading is normalized in `_normalize_grade()` in
`agents/quiz_generator.py`: scores are clamped to 0.0 to 1.0 and the
binary "correct" verdict overrides a contradictory generous numeric
score (small models sometimes return `correct: false` with an inflated
score). The grading prompt anchors away from example numbers and has a
dedicated 0.0 rubric bucket for non-answers such as "I don't know".

If you switch to a stronger judge model, grading gets sharper, but the
normalization applies regardless, so both interfaces stay consistent.

---

## Switching models

Edit `litellm_config.yaml` — the model name is a gateway routing concern,
not an app concern. Change the `model:` line to a model you have pulled and
restart the gateway:

```yaml
  - model_name: learning-accelerator
    litellm_params:
      model: ollama_chat/qwen2.5-coder:32b
      api_base: os.environ/OLLAMA_BASE_URL
```

```bash
ollama pull qwen2.5-coder:32b
make gateway-stop && make gateway    # restart the docker container to pick up the new config
```

No agent code changes needed. If you enable a cloud fallback chain
(GPT-4 primary, Claude then Ollama as fallbacks), do the same on that
entry and set `LITELLM_MODEL` to the cloud alias in `.env`.

## Fallback routing

The commented section at the bottom of `litellm_config.yaml` shows the
pattern: one alias per leg of the chain plus a `fallbacks` mapping. Rules
of thumb:

- Every model in a tool-calling chain must support tool calling (the
  Explainer uses `.bind_tools()`); a chain member without tool support
  breaks the Explainer whenever traffic lands on it.
- Keep the local Ollama leg as the last fallback: it is free and always
  reachable.
- Fallback latency stacks: each fallback attempt adds a full retry round
  trip. Two fallback legs is the practical maximum for a UI driven by
  per-call user waits.

---

## Eval test scores by model

Thresholds in `tests/test_eval.py` are calibrated for 7B models (0.6).
Larger models typically score higher:

| Model | Faithfulness | Relevancy | Question Quality |
|---|---|---|---|
| `qwen2.5:7b` | 0.70-0.82 | 0.75-0.90 | 0.65-0.75 |
| `qwen2.5-coder:32b` | 0.82-0.92 | 0.85-0.95 | 0.78-0.88 |

If eval tests consistently fail with a 7B model, try the 32B tier
before lowering thresholds.
