"""Unit tests for LangGraph message stream filtering."""

from langchain_core.messages import AIMessageChunk

from graph.streaming import (
    DEFAULT_TOOL_STATUS,
    ExplainerStreamFilter,
    iter_explanation_updates,
)


def meta(node: str = "explainer") -> dict:
    return {"langgraph_node": node}


def text_chunk(text: str, run_id: str = "run-1") -> AIMessageChunk:
    return AIMessageChunk(content=text, id=run_id)


def tool_chunk(
    name: str | None = "tool_read_file",
    run_id: str = "run-tool",
) -> AIMessageChunk:
    return AIMessageChunk(
        content="",
        id=run_id,
        tool_call_chunks=[
            {
                "name": name,
                "args": '{"filename": "closures.md"}',
                "id": "call-1",
                "index": 0,
            }
        ],
    )


class TestExplainerStreamFilter:
    def test_accumulates_explainer_text(self):
        stream_filter = ExplainerStreamFilter()

        first = stream_filter.feed(text_chunk("Hello"), meta())
        second = stream_filter.feed(text_chunk(" world"), meta())

        assert first is not None
        assert first.kind == "text"
        assert first.text == "Hello"
        assert first.full_text == "Hello"
        assert second is not None
        assert second.full_text == "Hello world"

    def test_ignores_other_nodes(self):
        stream_filter = ExplainerStreamFilter()

        update = stream_filter.feed(text_chunk("Coaching"), meta("progress_coach"))

        assert update is None
        assert stream_filter.full_text == ""

    def test_tool_call_resets_buffer(self):
        stream_filter = ExplainerStreamFilter()

        stream_filter.feed(text_chunk("Let me check your notes."), meta())
        reset = stream_filter.feed(tool_chunk(), meta())
        final = stream_filter.feed(
            text_chunk("The real explanation.", run_id="run-final"),
            meta(),
        )

        assert reset is not None
        assert reset.kind == "reset"
        assert reset.full_text == ""
        assert reset.status == "Reading your notes..."
        assert final is not None
        assert final.kind == "text"
        assert final.full_text == "The real explanation."

    def test_discards_remaining_chunks_in_tool_run(self):
        stream_filter = ExplainerStreamFilter()

        stream_filter.feed(tool_chunk(), meta())
        ignored = stream_filter.feed(
            AIMessageChunk(content="tool arguments", id="run-tool"),
            meta(),
        )

        assert ignored is None
        assert stream_filter.full_text == ""

    def test_new_run_after_tool_run_streams_again(self):
        stream_filter = ExplainerStreamFilter()

        stream_filter.feed(tool_chunk(), meta())
        update = stream_filter.feed(
            text_chunk("Final answer.", run_id="run-final"),
            meta(),
        )

        assert update is not None
        assert update.full_text == "Final answer."

    def test_tool_chunk_without_name_uses_default_status(self):
        stream_filter = ExplainerStreamFilter()

        update = stream_filter.feed(tool_chunk(name=None), meta())

        assert update is not None
        assert update.kind == "reset"
        assert update.status == DEFAULT_TOOL_STATUS

    def test_later_named_tool_chunk_updates_status(self):
        stream_filter = ExplainerStreamFilter()

        stream_filter.feed(tool_chunk(name=None), meta())
        update = stream_filter.feed(
            AIMessageChunk(
                content="",
                id="run-tool",
                tool_call_chunks=[
                    {
                        "name": "tool_search_notes",
                        "args": '{"query": "closures"}',
                        "id": "call-1",
                        "index": 0,
                    }
                ],
            ),
            meta(),
        )

        assert update is not None
        assert update.kind == "status"
        assert update.status == "Searching your notes..."

    def test_supports_content_blocks(self):
        stream_filter = ExplainerStreamFilter()

        update = stream_filter.feed(
            AIMessageChunk(
                content=[{"type": "text", "text": "Block text"}],
                id="run-blocks",
            ),
            meta(),
        )

        assert update is not None
        assert update.full_text == "Block text"

    def test_ignores_empty_content(self):
        stream_filter = ExplainerStreamFilter()

        update = stream_filter.feed(text_chunk(""), meta())

        assert update is None
        assert stream_filter.full_text == ""


class FakeGraph:
    def __init__(self, events: list[tuple]):
        self.events = events
        self.calls: list[dict] = []

    def stream(self, graph_input, config=None, stream_mode=None):
        self.calls.append(
            {
                "graph_input": graph_input,
                "config": config,
                "stream_mode": stream_mode,
            }
        )
        yield from self.events


class TestIterExplanationUpdates:
    def test_yields_only_final_explanation(self):
        events = [
            (text_chunk("Checking notes.", run_id="run-tool"), meta()),
            (tool_chunk(), meta()),
            (text_chunk("Coaching text.", run_id="run-coach"), meta("progress_coach")),
            (text_chunk("Final explanation.", run_id="run-final"), meta()),
        ]
        graph = FakeGraph(events)

        updates = list(
            iter_explanation_updates(
                graph,
                None,
                {"configurable": {"thread_id": "test"}},
            )
        )

        assert [u.kind for u in updates] == ["text", "reset", "text"]
        assert updates[-1].full_text == "Final explanation."
        assert graph.calls == [
            {
                "graph_input": None,
                "config": {"configurable": {"thread_id": "test"}},
                "stream_mode": "messages",
            }
        ]
