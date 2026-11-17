# src/grading/rubric.py
"""
Rubric builder for the grading engine.

Converts explicit key points or expected answers into a structured, weighted Rubric.
"""

from __future__ import annotations

import re
from .models import Rubric, RubricCriterion


class RubricBuilder:
    """Builds a Rubric from explicit key points or derives one from expected answer text."""

    @classmethod
    def build(
        cls,
        expected_answer: str,
        key_points: list[str] | None = None,
        passing_threshold: float = 0.5,
    ) -> Rubric:
        """
        Construct a structured Rubric.

        Args:
            expected_answer: The canonical reference answer.
            key_points: Optional explicit list of required concepts or criteria.
            passing_threshold: Score threshold for passing (default 0.5).

        Returns:
            A populated Rubric instance.
        """
        cleaned_points = cls._clean_points(key_points)
        if not cleaned_points:
            cleaned_points = cls._derive_points_from_text(expected_answer)

        criteria = [
            RubricCriterion(
                id=f"crit_{idx}",
                description=point,
                weight=1.0,
            )
            for idx, point in enumerate(cleaned_points, 1)
        ]

        return Rubric(criteria=criteria, passing_threshold=passing_threshold)

    @staticmethod
    def _clean_points(points: list[str] | None) -> list[str]:
        if not points or not isinstance(points, list):
            return []
        cleaned = []
        for p in points:
            text = str(p).strip()
            if text:
                cleaned.append(text)
        return cleaned

    @classmethod
    def _derive_points_from_text(cls, text: str) -> list[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return ["Demonstrates understanding of the topic."]

        # Check for numbered items: e.g. "1) ... 2) ..." or "1. ... 2. ..."
        parts = re.split(r"(?:^|\s)\d+[\).]\s+", cleaned)
        if len(parts) > 1:
            # If the original text didn't start with a number, parts[0] is introductory preamble
            if not re.match(r"^\d+[\).]", cleaned.strip()):
                parts = parts[1:]

            valid_parts = [p.strip(" .,;:") for p in parts if len(p.strip(" .,;:")) >= 10]
            if valid_parts:
                return valid_parts

        # Split on sentence boundaries or semicolons
        sentence_parts = re.split(r"(?<=[.!?])\s+|;\s*", cleaned)
        filtered_sentences = [
            s.strip(" .,;:") for s in sentence_parts if len(s.strip(" .,;:")) >= 12
        ]
        if len(filtered_sentences) >= 2:
            return filtered_sentences

        return [cleaned]


def build_rubric(
    expected_answer: str,
    key_points: list[str] | None = None,
    passing_threshold: float = 0.5,
) -> Rubric:
    """Convenience factory function."""
    return RubricBuilder.build(
        expected_answer=expected_answer,
        key_points=key_points,
        passing_threshold=passing_threshold,
    )
