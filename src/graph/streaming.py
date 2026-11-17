"""Helpers for filtering LangGraph message streams.

The Streamlit interface consumes ``stream_mode="messages"`` events and
renders only the Explainer's final answer. Tool-calling turns are
discarded so their preamble or tool-call arguments never reach the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal

from langchain_core.messages import AIMessage, BaseMessage

EXPLAINER_NODE = "explainer"

TOOL_STATUS = {
    "tool_list_files": "Listing your study notes...",
    "tool_search_notes": "Searching your notes...",
    "tool_read_file": "Reading your notes...",
    "tool_memory_get": "Checking what we've covered...",
    "tool_memory_set": "Saving what we've covered...",
}

DEFAULT_TOOL_STATUS = "Working with your notes..."


@dataclass
class ExplanationUpdate:
    """One UI-relevant event extracted from the graph message stream."""

    kind: Literal["text", "reset", "status"]
    text: str = ""
    full_text: str = ""
    status: str = ""


def _content_text(message: BaseMessage) -> str:
    """Return plain text from string or content-block message content."""
    content: Any = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _tool_status(tool_chunks: list[dict] | None) -> str | None:
    for tool_chunk in tool_chunks or []:
        name = tool_chunk.get("name") if isinstance(tool_chunk, dict) else None
        if name:
            return TOOL_STATUS.get(name, DEFAULT_TOOL_STATUS)
    return None


class ExplainerStreamFilter:
    """Isolate the Explainer's final answer from tool-calling turns."""

    def __init__(self) -> None:
        self.full_text = ""
        self._run_id: str | None = None
        self._discard_run = False

    def feed(
        self,
        message: BaseMessage,
        metadata: dict[str, Any],
    ) -> ExplanationUpdate | None:
        if metadata.get("langgraph_node") != EXPLAINER_NODE:
            return None

        run_id = getattr(message, "id", None)
        if run_id != self._run_id:
            self._run_id = run_id
            self._discard_run = False

        tool_chunks = getattr(message, "tool_call_chunks", None)
        if tool_chunks:
            was_discarding = self._discard_run
            self._discard_run = True
            self.full_text = ""
            status = _tool_status(tool_chunks)
            if not was_discarding:
                return ExplanationUpdate(
                    kind="reset",
                    status=status or DEFAULT_TOOL_STATUS,
                )
            if status:
                return ExplanationUpdate(kind="status", status=status)
            return None

        if self._discard_run:
            return None

        text = _content_text(message)
        if not text:
            return None

        self.full_text += text
        return ExplanationUpdate(
            kind="text",
            text=text,
            full_text=self.full_text,
        )


def iter_explanation_updates(
    graph: Any,
    graph_input: Any,
    config: dict[str, Any],
) -> Iterator[ExplanationUpdate]:
    """Run the graph and yield only UI-relevant explanation events."""
    stream_filter = ExplainerStreamFilter()
    for message, metadata in graph.stream(
        graph_input,
        config=config,
        stream_mode="messages",
    ):
        update = stream_filter.feed(message, metadata)
        if update is not None:
            yield update


def extract_explanation(messages: list) -> str:
    """Get the Explainer's final response, recovering substantive text if meta-message was generated."""
    candidates = []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            # Skip messages explicitly tagged from progress_coach
            if getattr(msg, "name", None) == "progress_coach":
                continue
            text = str(msg.content).strip()
            # If no tool calls, this is a standard final message candidate
            if not getattr(msg, "tool_calls", None):
                is_meta = len(text) < 250 and any(
                    phrase in text.lower()
                    for phrase in (
                        "stored the explanation",
                        "session memory",
                        "stored in memory",
                        "saved the explanation",
                    )
                )
                if not is_meta:
                    return text
                candidates.append(text)
            else:
                # If an AIMessage had tool calls but also substantive explanation text, keep as candidate
                if len(text) >= 200:
                    candidates.append(text)

    # Return longest candidate if available
    if candidates:
        return max(candidates, key=len)

    # Fallback: check if explanation text was placed in a tool call argument
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc.get("name") == "tool_memory_set" and isinstance(tc.get("args"), dict):
                    val = str(tc["args"].get("value", "")).strip()
                    if len(val) >= 150:
                        return val

    return ""


def extract_coaching(messages: list, explanation: str = "") -> str:
    """Get the latest coaching message from progress_coach, never returning an explanation."""
    norm_exp = explanation.strip()

    # 1. Prefer message explicitly tagged as progress_coach
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "name", None) == "progress_coach":
            content = str(msg.content).strip()
            if content and content != norm_exp:
                return content

    # 2. Defensive fallback: find the most recent non-tool AIMessage whose content
    # is clearly not an explanation (not identical to the explanation and short).
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            text = str(msg.content).strip()
            if norm_exp and text == norm_exp:
                continue
            # Coaching messages are short summaries (under 400 chars, no markdown headers/code)
            if len(text) < 400 and not any(h in text for h in ["###", "##", "```", "---"]):
                return text

    return ""
