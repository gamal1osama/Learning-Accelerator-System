# src/grading/pre_grader.py
"""
Pre-grader guardrails for fast-path deterministic evaluation.

Evaluates trivial cases (empty answers, non-answers/refusals, and self-declared
wrong answers) instantaneously without invoking an LLM.
"""

from __future__ import annotations

import re
from .models import GradeResult

_NON_ANSWER_EXACT = {
    "",
    "i dont know",
    "i don't know",
    "idk",
    "no idea",
    "not sure",
    "im not sure",
    "i'm not sure",
    "n a",
    "n/a",
    "na",
    "none",
    "no answer",
    "no answer provided",
    "skip",
    "pass",
    "dont know",
    "dunno",
    "haven't learned this yet",
    "havent learned this yet",
}

_SELF_DECLARED_WRONG_PHRASES = (
    "intentionally wrong",
    "deliberately wrong",
    "wrong on purpose",
    "this is a wrong answer",
    "this answer is wrong",
    "intentionally incorrect",
    "deliberately incorrect",
)


class PreGrader:
    """Deterministic fast-path checks before invoking the LLM judge."""

    @classmethod
    def evaluate(cls, student_answer: str) -> GradeResult | None:
        """
        Check if the student answer can be evaluated deterministically.

        Returns:
            GradeResult if a fast-path condition matches, otherwise None.
        """
        raw = student_answer.strip()
        if not raw:
            return GradeResult(
                score=0.0,
                correct=False,
                feedback="No answer was provided, so no credit can be given.",
                missing_concept="",
                missing_concepts=[],
                misconceptions=[],
                criterion_evaluations=[],
                graded_by="pre_grader",
            )

        # Normalize text: lowercased, letters and numbers with spaces
        normalized = " ".join(raw.lower().split()).strip(" .,!?:;\"'()[]{}")

        # 1. Exact non-answer or short refusal
        if normalized in _NON_ANSWER_EXACT:
            return GradeResult(
                score=0.0,
                correct=False,
                feedback="No answer was provided, so no credit can be given.",
                missing_concept="",
                missing_concepts=[],
                misconceptions=[],
                criterion_evaluations=[],
                graded_by="pre_grader",
            )

        # Refusal in very short responses (<= 30 chars, e.g. "I don't know this")
        if len(normalized) <= 30 and (
            normalized.startswith("i dont know")
            or normalized.startswith("i don't know")
            or normalized.startswith("no idea")
            or normalized.startswith("not sure")
        ):
            return GradeResult(
                score=0.0,
                correct=False,
                feedback="No substantive answer was provided, so no credit can be given.",
                missing_concept="",
                missing_concepts=[],
                misconceptions=[],
                criterion_evaluations=[],
                graded_by="pre_grader",
            )

        # 2. Self-declared intentionally wrong answers
        opening = normalized[:160]
        if any(phrase in opening for phrase in _SELF_DECLARED_WRONG_PHRASES):
            return GradeResult(
                score=0.0,
                correct=False,
                feedback="The answer indicates it is intentionally incorrect, so no credit can be given.",
                missing_concept="",
                missing_concepts=[],
                misconceptions=["Self-declared incorrect solution"],
                criterion_evaluations=[],
                graded_by="pre_grader",
            )

        return None
