# src/grading/models.py
"""
Domain models and schemas for the question grading engine.

Defines Pydantic v2 models representing rubric criteria, evaluation statuses,
structured judge outputs, and final grading results.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class CriterionStatus(str, Enum):
    """Evaluation status for a single rubric criterion."""
    MET = "met"
    PARTIALLY_MET = "partially_met"
    MISSED = "missed"


class RubricCriterion(BaseModel):
    """A single evaluation criterion within a rubric."""
    id: str = Field(description="Unique identifier for the criterion (e.g. 'crit_1')")
    description: str = Field(description="Actionable statement of what the student answer must demonstrate")
    weight: float = Field(default=1.0, ge=0.0, description="Relative weight of this criterion in the total score")


class Rubric(BaseModel):
    """Rubric comprising weighted criteria and scoring configuration."""
    criteria: list[RubricCriterion] = Field(default_factory=list)
    passing_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum score for a pass verdict")

    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria) if self.criteria else 0.0


class CriterionEvaluation(BaseModel):
    """Evaluation of one rubric criterion by the judge."""
    criterion_id: str = Field(description="ID of the evaluated criterion")
    status: CriterionStatus = Field(description="Whether the point was met, partially met, or missed")
    evidence: str = Field(default="", description="Direct quotation or specific evidence from the student answer")
    reasoning: str = Field(default="", description="Brief explanation for the assigned status")


class LLMJudgeOutput(BaseModel):
    """Structured response expected from the LLM judge."""
    evaluations: list[CriterionEvaluation] = Field(
        description="Individual evaluation for each criterion in the rubric"
    )
    feedback: str = Field(
        description="Constructive, specific 1-2 sentence feedback tailored to the student's answer"
    )
    missing_concepts: list[str] = Field(
        default_factory=list,
        description="Concise labels or phrases for key concepts that were missed or incomplete"
    )
    misconceptions: list[str] = Field(
        default_factory=list,
        description="Any specific incorrect assertions or misconceptions identified in the answer"
    )


class GradeResult(BaseModel):
    """
    Final evaluated grade result for a student submission.

    Includes calculated score, pass/fail status, actionable feedback,
    and detailed criterion breakdowns. Provides .to_dict() for backward
    compatibility with existing dictionary consumers.
    """
    score: float = Field(ge=0.0, le=1.0, description="Final normalized score between 0.0 and 1.0")
    correct: bool = Field(description="True if score >= passing_threshold")
    feedback: str = Field(description="User-facing summary feedback")
    missing_concept: str = Field(default="", description="Primary missing concept for legacy compatibility")
    missing_concepts: list[str] = Field(default_factory=list, description="All identified missing concepts")
    misconceptions: list[str] = Field(default_factory=list, description="Identified misconceptions")
    criterion_evaluations: list[CriterionEvaluation] = Field(default_factory=list)
    graded_by: str = Field(default="llm", description="Origin of grade: 'pre_grader', 'llm', or 'fallback'")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary matching legacy grade_answer output format."""
        return {
            "correct": self.correct,
            "score": self.score,
            "feedback": self.feedback,
            "missing_concept": self.missing_concept,
            "missing_concepts": list(self.missing_concepts),
            "misconceptions": list(self.misconceptions),
            "criterion_evaluations": [e.model_dump() for e in self.criterion_evaluations],
            "graded_by": self.graded_by,
        }
