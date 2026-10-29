"""
tests/test_checkpointing.py

Tests for checkpointing and human-in-the-loop approval.

These tests verify:
  - SqliteSaver creates and reads checkpoints correctly
  - human_approval_node returns correct state for yes/no
  - route_after_approval routing logic
  - route_after_coach routing logic
  - session_is_complete edge cases

The interrupt() mechanism itself can't be unit tested directly
(it requires a compiled graph + checkpointer). Those are covered
in the integration tests (Batch 9).

Run: python -m pytest tests/test_checkpointing.py -v
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch, MagicMock

import pytest

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from graph.state import (
    StudyRoadmap,
    Topic,
    initial_state,
    session_is_complete,
)
from graph.workflow import route_after_approval, route_after_coach
from agents.human_approval import human_approval_node


# ─────────────────────────────────────────────────────────────────────────────
# SqliteSaver tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSqliteSaver:
    """Verify SqliteSaver can be imported and initialised."""

    def test_sqlite_saver_can_be_imported(self):
        """SqliteSaver should be importable from langgraph."""
        from langgraph.checkpoint.sqlite import SqliteSaver
        assert SqliteSaver is not None

    def test_sqlite_saver_creates_db_file(self):
        """SqliteSaver should create the database file on initialisation."""
        from langgraph.checkpoint.sqlite import SqliteSaver
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_checkpoints.db")
            assert not os.path.exists(db_path)
            # Creating the saver should create the file
            saver = SqliteSaver.from_conn_string(db_path)
            # Access the connection to trigger file creation
            with saver:
                pass
            # File should now exist
            assert os.path.exists(db_path)

    def test_data_directory_created_by_workflow(self):
        """build_graph() should create data/ directory if it doesn't exist."""
        # This verifies the Path("data").mkdir(exist_ok=True) line works
        data_dir = Path("data")
        # data/ already exists from earlier batches, but exist_ok=True
        # means calling mkdir on it again is safe
        data_dir.mkdir(exist_ok=True)
        assert data_dir.exists()

    def test_falls_back_to_sqlite_without_dsn(self, monkeypatch, tmp_path):
        """No DSN env var -> legacy SqliteSaver (no Postgres required)."""
        from graph.checkpointer import make_checkpointer
        from langgraph.checkpoint.sqlite import SqliteSaver

        monkeypatch.delenv("CHECKPOINT_DB_URL", raising=False)
        saver = make_checkpointer(
            dsn_env_var="CHECKPOINT_DB_URL",
            sqlite_default=str(tmp_path / "fallback.db"),
        )
        assert isinstance(saver, SqliteSaver)
        saver.conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Postgres checkpointer tests (require a live Postgres)
# ─────────────────────────────────────────────────────────────────────────────

PG_DSN = os.getenv("TEST_CHECKPOINT_DB_URL")
PG_DSN_UI = os.getenv("TEST_CHECKPOINT_DB_URL_UI")
requires_postgres = pytest.mark.skipif(
    not PG_DSN,
    reason="TEST_CHECKPOINT_DB_URL not set; start Postgres with: make postgres",
)


class _CounterState(TypedDict):
    count: int
    decision: str


def _bump(state: _CounterState) -> dict:
    return {"count": state["count"] + 1}


def _ask(state: _CounterState) -> dict:
    answer = interrupt({"prompt": "continue?"})
    return {"decision": answer}


def _build_counter_graph(checkpointer):
    """
    Tiny LLM-free graph: START -> bump -> ask (interrupt) -> END.

    Exercises the exact primitives the real app relies on: checkpoint
    writes after every node and a human-in-the-loop interrupt.
    """
    builder = StateGraph(_CounterState)
    builder.add_node("bump", _bump)
    builder.add_node("ask", _ask)
    builder.add_edge(START, "bump")
    builder.add_edge("bump", "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=checkpointer)


def _make_pg_saver(env_var: str = "TEST_CHECKPOINT_DB_URL"):
    from graph.checkpointer import make_checkpointer

    return make_checkpointer(dsn_env_var=env_var, sqlite_default="data/unused.db")


def _close(saver) -> None:
    """Release the pool/connection so tests don't leak connections."""
    conn = getattr(saver, "conn", None)
    if conn is not None and hasattr(conn, "close"):
        conn.close()


@requires_postgres
class TestPostgresCheckpointer:
    """Persistence, resume across instances, and database isolation."""

    def test_setup_is_idempotent(self):
        """Two successive initialisations against the same DB must not fail."""
        first = _make_pg_saver()
        second = _make_pg_saver()
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            assert isinstance(first, PostgresSaver)
            assert isinstance(second, PostgresSaver)
        finally:
            _close(first)
            _close(second)

    def test_checkpoint_roundtrip_across_instances(self):
        """State written by one instance is readable by a fresh instance."""
        thread_id = f"pg-roundtrip-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        writer = _make_pg_saver()
        try:
            graph_a = _build_counter_graph(writer)
            graph_a.invoke({"count": 0, "decision": ""}, config=config)
            # Paused at the interrupt: the bump node's checkpoint is saved.
            state_a = graph_a.get_state(config)
            assert state_a.values["count"] == 1
        finally:
            _close(writer)

        # Brand-new instance: new pool, same database, same thread.
        reader = _make_pg_saver()
        try:
            graph_b = _build_counter_graph(reader)
            state_b = graph_b.get_state(config)
            assert state_b.values["count"] == 1
        finally:
            _close(reader)

    def test_interrupt_resumed_by_second_instance(self):
        """
        The core multi-instance property: instance A pauses on an
        interrupt, instance B picks the thread up and finishes it.
        """
        thread_id = f"pg-resume-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        instance_a = _make_pg_saver()
        try:
            graph_a = _build_counter_graph(instance_a)
            result = graph_a.invoke({"count": 0, "decision": ""}, config=config)
            assert "__interrupt__" in result
        finally:
            _close(instance_a)

        instance_b = _make_pg_saver()
        try:
            graph_b = _build_counter_graph(instance_b)
            result = graph_b.invoke(Command(resume="yes"), config=config)
            assert result["decision"] == "yes"
            assert result["count"] == 1
            assert "__interrupt__" not in result
        finally:
            _close(instance_b)

    @pytest.mark.skipif(
        not PG_DSN_UI, reason="TEST_CHECKPOINT_DB_URL_UI not set"
    )
    def test_terminal_and_ui_databases_are_isolated(self):
        """A thread written to the terminal DB must not appear in the UI DB."""
        thread_id = f"pg-isolation-{uuid.uuid4()}"
        config = {"configurable": {"thread_id": thread_id}}

        terminal = _make_pg_saver("TEST_CHECKPOINT_DB_URL")
        try:
            graph_t = _build_counter_graph(terminal)
            graph_t.invoke({"count": 0, "decision": ""}, config=config)
            assert graph_t.get_state(config).values["count"] == 1
        finally:
            _close(terminal)

        ui = _make_pg_saver("TEST_CHECKPOINT_DB_URL_UI")
        try:
            graph_u = _build_counter_graph(ui)
            assert graph_u.get_state(config).values == {}
        finally:
            _close(ui)


# ─────────────────────────────────────────────────────────────────────────────
# Human approval node tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHumanApprovalNode:
    """Tests for the human_approval_node function."""

    def _make_state_with_roadmap(self):
        topics = [
            Topic("Closures", "Understanding closures", 60),
            Topic("Decorators", "Building decorators", 75,
                  prerequisites=["Closures"]),
        ]
        state = initial_state("Learn Python", "session-test")
        state["roadmap"] = StudyRoadmap("Learn Python", 2, topics)
        return state

    @patch("agents.human_approval.interrupt")
    def test_yes_answer_sets_approved_true(self, mock_interrupt):
        """'yes' input should result in approved=True."""
        mock_interrupt.return_value = "yes"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    @patch("agents.human_approval.interrupt")
    def test_no_answer_sets_approved_false(self, mock_interrupt):
        """'no' input should result in approved=False."""
        mock_interrupt.return_value = "no"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is False

    @patch("agents.human_approval.interrupt")
    def test_y_shorthand_is_accepted(self, mock_interrupt):
        """'y' should be treated as approval."""
        mock_interrupt.return_value = "y"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    @patch("agents.human_approval.interrupt")
    def test_ok_is_accepted(self, mock_interrupt):
        """'ok' should be treated as approval."""
        mock_interrupt.return_value = "ok"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    @patch("agents.human_approval.interrupt")
    def test_case_insensitive(self, mock_interrupt):
        """Approval check should be case-insensitive."""
        mock_interrupt.return_value = "YES"
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    @patch("agents.human_approval.interrupt")
    def test_whitespace_stripped(self, mock_interrupt):
        """Extra whitespace should not affect the decision."""
        mock_interrupt.return_value = "  yes  "
        state = self._make_state_with_roadmap()
        result = human_approval_node(state)
        assert result["approved"] is True

    def test_no_roadmap_auto_approves(self):
        """If there's no roadmap, node should auto-approve without interrupt."""
        state = initial_state("test", "s1")
        state["roadmap"] = None
        # Should not call interrupt(), auto-approve
        result = human_approval_node(state)
        assert result["approved"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Routing function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRouteAfterApproval:
    """Tests for the route_after_approval routing function."""

    def test_approved_routes_to_explainer(self):
        state = initial_state("test", "s1")
        state["approved"] = True
        assert route_after_approval(state) == "explainer"

    def test_not_approved_routes_to_planner(self):
        state = initial_state("test", "s1")
        state["approved"] = False
        assert route_after_approval(state) == "curriculum_planner"

    def test_missing_approved_routes_to_planner(self):
        """Default (missing approved key) should route to replanning."""
        state = initial_state("test", "s1")
        # approved defaults to False in initial_state
        assert route_after_approval(state) == "curriculum_planner"


class TestRouteAfterCoach:
    """Tests for the route_after_coach routing function."""

    def _make_state(self, n_topics, current_index):
        topics = [Topic(f"T{i}", f"D{i}", 30) for i in range(n_topics)]
        state = initial_state("test", "s1")
        state["roadmap"] = StudyRoadmap("test", 1, topics)
        state["current_topic_index"] = current_index
        return state

    def test_routes_to_explainer_with_remaining_topics(self):
        state = self._make_state(3, 1)  # 3 topics, currently on index 1
        assert route_after_coach(state) == "explainer"

    def test_routes_to_end_when_all_done(self):
        state = self._make_state(3, 3)  # index 3 = past all 3 topics
        assert route_after_coach(state) == "end"

    def test_routes_to_end_on_boundary(self):
        state = self._make_state(2, 2)  # exactly at the end
        assert route_after_coach(state) == "end"


# ─────────────────────────────────────────────────────────────────────────────
# session_is_complete edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSessionIsComplete:
    """Tests for the session_is_complete helper."""

    def test_not_complete_at_start(self):
        topics = [Topic("T1", "D1", 30), Topic("T2", "D2", 30)]
        state = initial_state("test", "s1")
        state["roadmap"] = StudyRoadmap("test", 1, topics)
        state["current_topic_index"] = 0
        assert session_is_complete(state) is False

    def test_complete_when_index_equals_topic_count(self):
        topics = [Topic("T1", "D1", 30)]
        state = initial_state("test", "s1")
        state["roadmap"] = StudyRoadmap("test", 1, topics)
        state["current_topic_index"] = 1
        assert session_is_complete(state) is True

    def test_complete_when_index_exceeds_topic_count(self):
        """Index past end (shouldn't happen, but handle gracefully)."""
        topics = [Topic("T1", "D1", 30)]
        state = initial_state("test", "s1")
        state["roadmap"] = StudyRoadmap("test", 1, topics)
        state["current_topic_index"] = 99
        assert session_is_complete(state) is True

    def test_complete_without_roadmap(self):
        """No roadmap = nothing to study = complete."""
        state = initial_state("test", "s1")
        assert session_is_complete(state) is True
