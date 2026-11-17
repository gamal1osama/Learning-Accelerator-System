# src/grading/__init__.py
"""
Production-grade Question Grading Engine.

Provides domain models, rubric construction, deterministic scoring,
and pipeline orchestration.
"""

from .engine import GradingEngine, get_default_grader
from .models import (
    CriterionEvaluation,
    CriterionStatus,
    GradeResult,
    LLMJudgeOutput,
    Rubric,
    RubricCriterion,
)
from .pre_grader import PreGrader
from .rubric import RubricBuilder, build_rubric
from .scorers import ScoreAggregator

__all__ = [
    "GradingEngine",
    "get_default_grader",
    "GradeResult",
    "Rubric",
    "RubricCriterion",
    "CriterionStatus",
    "CriterionEvaluation",
    "LLMJudgeOutput",
    "PreGrader",
    "RubricBuilder",
    "build_rubric",
    "ScoreAggregator",
]
