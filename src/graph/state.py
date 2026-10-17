# src/graph/state.py

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class Topic:
    """A single topic within the study roadmap."""
    title: str
    description: str
    estimated_minutes: int
    prerequisites: list[str] = field(default_factory=list)
    # pending → in_progress → completed | needs_review
    status: str = "pending"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Topic":
        """Reconstruct from a plain dict (e.g., after JSON round-trip)."""
        return cls(
            title=data["title"],
            description=data["description"],
            estimated_minutes=data["estimated_minutes"],
            prerequisites=data.get("prerequisites", []),
            status=data.get("status", "pending"),
        )


@dataclass
class StudyRoadmap:
    """The full study plan produced by the Curriculum Planner."""
    goal: str
    total_weeks: int
    topics: list[Topic]
    weekly_hours: int = 5

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "goal": self.goal,
            "total_weeks": self.total_weeks,
            "weekly_hours": self.weekly_hours,
            "topics": [t.to_dict() for t in self.topics],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StudyRoadmap":
        """Reconstruct from a plain dict."""
        return cls(
            goal=data["goal"],
            total_weeks=data["total_weeks"],
            weekly_hours=data.get("weekly_hours", 5),
            topics=[Topic.from_dict(t) for t in data.get("topics", [])],
        )

    def completed_count(self) -> int:
        """How many topics have been completed."""
        return sum(1 for t in self.topics if t.status == "completed")

    def is_complete(self) -> bool:
        """True when all topics are completed or needs_review."""
        return all(t.status in ("completed", "needs_review") for t in self.topics)


@dataclass
class QuizQuestion:
    """
    One question within a quiz, with the user's answer and grading.

    The Quiz Generator creates these.
    The Progress Coach reads them to identify weak areas.
    """
    question: str
    expected_answer: str

    # Filled in after the user answers
    user_answer: str = ""
    correct: bool = False
    feedback: str = ""          # One sentence of specific feedback from grader
    score: float = 0.0          # 0.0 – 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QuizResult:
    """
    The complete result of one quiz session on a single topic.

    Stored in AgentState.quiz_results, one entry per topic per session.
    The Progress Coach reads this to decide what to do next.
    """
    topic: str
    questions: list[QuizQuestion]
    score: float            # Average score across all questions (0.0 – 1.0)
    weak_areas: list[str]   # Concepts the student got wrong or missed
    timestamp: str = ""     # ISO format UTC timestamp

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "score": self.score,
            "weak_areas": self.weak_areas,
            "timestamp": self.timestamp,
            "questions": [q.to_dict() for q in self.questions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuizResult":
        """
        Reconstruct from a plain dict.

        Called when LangGraph deserializes quiz_results from a SQLite
        checkpoint as raw dicts (msgpack round-trip). This happens when
        resuming a crashed or interrupted session.
        """
        return cls(
            topic=data.get("topic", ""),
            questions=[],           # Questions not needed for coaching logic
            score=float(data.get("score", 0.0)),
            weak_areas=data.get("weak_areas", []),
            timestamp=data.get("timestamp", ""),
        )

    def passed(self) -> bool:
        """A score of 0.5 or above is considered a pass."""
        return self.score >= 0.5

    def strong_pass(self) -> bool:
        """A score of 0.75 or above, ready to move to next topic."""
        return self.score >= 0.75



class AgentState(TypedDict):
    """
    The shared state for the Learning Accelerator graph.

    Partial updates: when a node returns {"approved": True}, LangGraph
    merges that into the existing state. It does NOT replace the whole dict.
    Nodes only return the keys they changed.

    The one exception is `messages`: it uses the add_messages reducer,
    which appends to the list instead of replacing it.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    goal: str
    roadmap: StudyRoadmap | None
    approved: bool
    current_topic_index: int
    quiz_results: list[QuizResult]
    weak_areas: list[str]
    study_materials_path: str
    error: str | None






def initial_state(
    goal: str,
    session_id: str,
    study_materials_path: str = "study_materials/sample_notes",
) -> dict:
    """Create the initial state for a new study session."""
    return {
        "messages": [],
        "session_id": session_id,
        "goal": goal,
        "roadmap": None,
        "approved": False,
        "current_topic_index": 0,
        "quiz_results": [],
        "weak_areas": [],
        "study_materials_path": study_materials_path,
        "error": None,
    }


def get_current_topic(state: dict) -> Topic | None:
    """Get the topic currently being studied, or None if done."""
    roadmap = state.get("roadmap")
    if roadmap is None:
        return None

    # After checkpoint deserialization, roadmap may be a dict
    if isinstance(roadmap, dict):
        topics_raw = roadmap.get("topics", [])
    else:
        topics_raw = roadmap.topics

    idx = state.get("current_topic_index", 0)
    if idx >= len(topics_raw):
        return None

    t = topics_raw[idx]
    # Individual topics may also be dicts after deserialization
    if isinstance(t, dict):
        return Topic.from_dict(t)
    return t

def get_latest_quiz_result(state: dict) -> QuizResult | None:
    """Get the most recent quiz result, or None if no quizzes have run."""
    results = state.get("quiz_results", [])
    if not results:
        return None

    latest = results[-1]

    # After msgpack checkpoint deserialization, quiz results may come
    # back as plain dicts. Reconstruct them using from_dict().
    if isinstance(latest, dict):
        return QuizResult.from_dict(latest)

    return latest

def session_is_complete(state: dict) -> bool:
    """True when all topics have been studied."""
    roadmap = state.get("roadmap")
    if roadmap is None:
        return True
    topics = roadmap.get("topics", []) if isinstance(roadmap, dict) else roadmap.topics
    idx = state.get("current_topic_index", 0)
    return idx >= len(topics)



