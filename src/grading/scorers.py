# src/grading/scorers.py
"""
Deterministic score calculation and consistency enforcement.

Computes mathematically consistent scores from evaluated rubric criteria,
eliminating judge self-contradiction.
"""

from __future__ import annotations

from .models import (
    CriterionStatus,
    GradeResult,
    LLMJudgeOutput,
    Rubric,
)

_STATUS_WEIGHTS = {
    CriterionStatus.MET: 1.0,
    CriterionStatus.PARTIALLY_MET: 0.5,
    CriterionStatus.MISSED: 0.0,
}

_GENERIC_PRAISE = {
    "",
    "correct",
    "good",
    "good answer",
    "well done",
    "great job",
    "correct answer",
    "meets requirements",
}


class ScoreAggregator:
    """Calculates weighted scores and ensures output consistency."""

    @classmethod
    def aggregate(
        cls,
        judge_output: LLMJudgeOutput,
        rubric: Rubric,
    ) -> GradeResult:
        """
        Compute final grade from LLM criterion evaluations and rubric weights.

        Args:
            judge_output: The parsed output from the LLM judge.
            rubric: The rubric defining criteria and weights.

        Returns:
            Normalized GradeResult.
        """
        eval_map = {ev.criterion_id: ev for ev in judge_output.evaluations}

        total_weight = rubric.total_weight()
        weighted_sum = 0.0
        missed_criteria_descriptions = []

        for crit in rubric.criteria:
            evaluation = eval_map.get(crit.id)
            if evaluation is None:
                # If a criterion was not evaluated, count as missed
                status = CriterionStatus.MISSED
            else:
                status = evaluation.status

            multiplier = _STATUS_WEIGHTS.get(status, 0.0)
            weighted_sum += crit.weight * multiplier

            if status in (CriterionStatus.MISSED, CriterionStatus.PARTIALLY_MET):
                missed_criteria_descriptions.append(crit.description)

        if total_weight > 0:
            raw_score = weighted_sum / total_weight
        else:
            raw_score = 0.0

        # Clamp between 0.0 and 1.0, round to 2 decimal places
        final_score = round(max(0.0, min(1.0, raw_score)), 2)
        correct = final_score >= rubric.passing_threshold

        feedback = judge_output.feedback.strip()
        missing_concepts = [c.strip() for c in judge_output.missing_concepts if c.strip()]

        # Defensive massaging of placeholder strings
        generic_placeholders = {"key concept missed", "none", "n/a", "empty string"}
        missing_concepts = [
            c for c in missing_concepts
            if c.lower() not in generic_placeholders and len(c) <= 120
        ]

        # Normalization rules:
        if final_score >= 1.0:
            missing_concepts = []
            feedback = "Your answer covers all the key points."
        elif not missing_concepts and missed_criteria_descriptions:
            # Fall back to missed criteria descriptions if LLM forgot to list them
            missing_concepts = missed_criteria_descriptions[:2]

        primary_missing = missing_concepts[0] if missing_concepts else ""

        return GradeResult(
            score=final_score,
            correct=correct,
            feedback=feedback or ("Answer meets requirements." if correct else "Answer needs improvement."),
            missing_concept=primary_missing,
            missing_concepts=missing_concepts,
            misconceptions=judge_output.misconceptions,
            criterion_evaluations=judge_output.evaluations,
            graded_by="llm",
        )
