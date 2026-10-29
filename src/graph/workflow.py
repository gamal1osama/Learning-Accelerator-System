# src/graph/workflow.py

from langgraph.graph import END, START, StateGraph 

from agents.curriculum_planner import curriculum_planner_node
from agents.explainer import explainer_node
from agents.human_approval import human_approval_node
from agents.progress_coach import progress_coach_node
from agents.quiz_generator import quiz_generator_node
from graph.checkpointer import make_checkpointer
from graph.state import AgentState, session_is_complete


def route_after_approval(state: dict) -> str:
    if state.get("approved", False):
        return "explainer"
    return "curriculum_planner"


def route_after_coach(state: dict) -> str:
    if session_is_complete(state):
        return "end"
    return "explainer"


def build_graph(
    db_path: str = "data/checkpoints.db",
    interrupt_before: list | None = None,
    dsn_env_var: str = "CHECKPOINT_DB_URL",
):
    builder = StateGraph(AgentState)

    # Register all five nodes
    builder.add_node("curriculum_planner", curriculum_planner_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("explainer", explainer_node)
    builder.add_node("quiz_generator", quiz_generator_node)
    builder.add_node("progress_coach", progress_coach_node)

    # Static edges
    builder.add_edge(START, "curriculum_planner")
    builder.add_edge("curriculum_planner", "human_approval")
    builder.add_edge("explainer", "quiz_generator")
    builder.add_edge("quiz_generator", "progress_coach")

    # Conditional edges
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"explainer": "explainer", "curriculum_planner": "curriculum_planner"},
    )
    builder.add_conditional_edges(
        "progress_coach",
        route_after_coach,
        {"explainer": "explainer", "end": END},
    )

    # Backend-agnostic checkpointer: PostgresSaver when dsn_env_var is set,
    # SqliteSaver otherwise. See graph/checkpointer.py.
    checkpointer = make_checkpointer(
        dsn_env_var=dsn_env_var,
        sqlite_default=db_path,
    )

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or [],
    )
