# tests/test_grading.py
"""
Comprehensive unit tests for the production grading engine (src/grading/).
"""

from unittest.mock import MagicMock, patch
import pytest

from grading.models import (
    CriterionEvaluation,
    CriterionStatus,
    GradeResult,
    LLMJudgeOutput,
    Rubric,
    RubricCriterion,
)
from grading.pre_grader import PreGrader
from grading.rubric import RubricBuilder, build_rubric
from grading.scorers import ScoreAggregator
from grading.engine import GradingEngine
from agents.quiz_generator import grade_answer


# =============================================================================
# PreGrader Tests
# =============================================================================

class TestPreGrader:
    def test_empty_or_whitespace_answer_scores_zero(self):
        for empty_val in ["", "   ", "\n\t  \n"]:
            res = PreGrader.evaluate(empty_val)
            assert res is not None
            assert res.score == 0.0
            assert res.correct is False
            assert res.graded_by == "pre_grader"
            assert "No answer" in res.feedback

    @pytest.mark.parametrize("phrase", [
        "I don't know",
        "i dont know",
        "idk",
        "IDK.",
        "no idea",
        "Not sure!",
        "n/a",
        "N A",
        "na",
        "none",
        "skip",
        "pass",
        "no answer provided",
    ])
    def test_non_answer_refusals_score_zero(self, phrase):
        res = PreGrader.evaluate(phrase)
        assert res is not None
        assert res.score == 0.0
        assert res.correct is False
        assert res.graded_by == "pre_grader"

    def test_short_refusal_phrase_scores_zero(self):
        res = PreGrader.evaluate("I don't know this concept")
        assert res is not None
        assert res.score == 0.0
        assert res.correct is False
        assert res.graded_by == "pre_grader"

    def test_self_declared_intentionally_wrong_answer_scores_zero(self):
        answer = (
            "Intentionally wrong answer:\n\n"
            "import sys\n"
            "if sys.version.split()[0] < '3.8':\n"
            "    print('incompatible')"
        )
        res = PreGrader.evaluate(answer)
        assert res is not None
        assert res.score == 0.0
        assert res.correct is False
        assert res.graded_by == "pre_grader"
        assert "intentionally incorrect" in res.feedback

    def test_valid_attempt_returns_none(self):
        res = PreGrader.evaluate("A closure is a nested function that captures outer scope.")
        assert res is None


# =============================================================================
# RubricBuilder Tests
# =============================================================================

class TestRubricBuilder:
    def test_build_from_explicit_key_points(self):
        points = ["Nested function", "Captures variable", "Returned by outer function"]
        rubric = build_rubric("Expected text", points)
        assert len(rubric.criteria) == 3
        assert rubric.criteria[0].id == "crit_1"
        assert rubric.criteria[0].description == "Nested function"
        assert rubric.criteria[1].id == "crit_2"
        assert rubric.criteria[2].id == "crit_3"
        assert rubric.total_weight() == 3.0

    def test_build_derives_numbered_points_from_expected(self):
        expected = (
            "A closure requires: 1) a nested inner function, "
            "2) the inner function must reference the enclosing scope, "
            "3) the enclosing function must return the inner function."
        )
        rubric = build_rubric(expected, key_points=None)
        assert len(rubric.criteria) == 3
        assert "nested inner function" in rubric.criteria[0].description

    def test_build_derives_sentences_from_expected(self):
        expected = (
            "Late binding looks up values at call time. "
            "The fix is to use a default argument for each lambda."
        )
        rubric = build_rubric(expected, key_points=[])
        assert len(rubric.criteria) == 2
        assert "Late binding" in rubric.criteria[0].description
        assert "default argument" in rubric.criteria[1].description

    def test_empty_expected_answer_creates_fallback_criterion(self):
        rubric = build_rubric("", key_points=[])
        assert len(rubric.criteria) == 1
        assert rubric.criteria[0].weight == 1.0


# =============================================================================
# ScoreAggregator Tests
# =============================================================================

class TestScoreAggregator:
    def test_all_met_produces_perfect_score(self):
        rubric = Rubric(criteria=[
            RubricCriterion(id="c1", description="Point 1", weight=1.0),
            RubricCriterion(id="c2", description="Point 2", weight=1.0),
        ])
        judge_out = LLMJudgeOutput(
            evaluations=[
                CriterionEvaluation(criterion_id="c1", status=CriterionStatus.MET, evidence="pt1"),
                CriterionEvaluation(criterion_id="c2", status=CriterionStatus.MET, evidence="pt2"),
            ],
            feedback="Good answer",
            missing_concepts=["should be cleared"],
            misconceptions=[],
        )

        res = ScoreAggregator.aggregate(judge_out, rubric)
        assert res.score == 1.0
        assert res.correct is True
        assert res.missing_concepts == []
        assert res.feedback == "Your answer covers all the key points."

    def test_partial_and_missed_produces_proportional_score(self):
        rubric = Rubric(criteria=[
            RubricCriterion(id="c1", description="Point 1", weight=1.0),
            RubricCriterion(id="c2", description="Point 2", weight=1.0),
            RubricCriterion(id="c3", description="Point 3", weight=1.0),
        ])
        judge_out = LLMJudgeOutput(
            evaluations=[
                CriterionEvaluation(criterion_id="c1", status=CriterionStatus.MET, evidence="pt1"),
                CriterionEvaluation(criterion_id="c2", status=CriterionStatus.PARTIALLY_MET, evidence="pt2"),
                CriterionEvaluation(criterion_id="c3", status=CriterionStatus.MISSED, evidence=""),
            ],
            feedback="Missing point 3",
            missing_concepts=["Point 3"],
            misconceptions=[],
        )

        # (1.0 + 0.5 + 0.0) / 3 = 0.5
        res = ScoreAggregator.aggregate(judge_out, rubric)
        assert res.score == 0.5
        assert res.correct is True
        assert res.missing_concept == "Point 3"

    def test_weighted_criteria_calculation(self):
        rubric = Rubric(criteria=[
            RubricCriterion(id="c1", description="Crucial Point", weight=3.0),
            RubricCriterion(id="c2", description="Minor Detail", weight=1.0),
        ])
        judge_out = LLMJudgeOutput(
            evaluations=[
                CriterionEvaluation(criterion_id="c1", status=CriterionStatus.MET, evidence="crucial"),
                CriterionEvaluation(criterion_id="c2", status=CriterionStatus.MISSED, evidence=""),
            ],
            feedback="Missed minor detail",
            missing_concepts=["Minor Detail"],
            misconceptions=[],
        )

        # (3.0 * 1.0 + 1.0 * 0.0) / 4.0 = 0.75
        res = ScoreAggregator.aggregate(judge_out, rubric)
        assert res.score == 0.75
        assert res.correct is True

    def test_unevaluated_criterion_treated_as_missed(self):
        rubric = Rubric(criteria=[
            RubricCriterion(id="c1", description="Point 1"),
            RubricCriterion(id="c2", description="Point 2"),
        ])
        # Judge only evaluated c1
        judge_out = LLMJudgeOutput(
            evaluations=[
                CriterionEvaluation(criterion_id="c1", status=CriterionStatus.MET, evidence="pt1"),
            ],
            feedback="Partial",
            missing_concepts=[],
            misconceptions=[],
        )

        res = ScoreAggregator.aggregate(judge_out, rubric)
        assert res.score == 0.5


# =============================================================================
# GradingEngine Tests
# =============================================================================

class TestGradingEngine:
    def test_pre_grader_intercepts_blank_without_llm(self):
        mock_llm = MagicMock()
        engine = GradingEngine(llm=mock_llm)
        res = engine.grade("Q?", "Exp", "")
        assert res.score == 0.0
        assert res.correct is False
        assert res.graded_by == "pre_grader"
        mock_llm.invoke.assert_not_called()

    def test_llm_judge_success_with_structured_output(self):
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = LLMJudgeOutput(
            evaluations=[
                CriterionEvaluation(criterion_id="crit_1", status=CriterionStatus.MET, evidence="nested"),
                CriterionEvaluation(criterion_id="crit_2", status=CriterionStatus.MET, evidence="returns"),
            ],
            feedback="Accurate answer.",
            missing_concepts=[],
            misconceptions=[],
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        engine = GradingEngine(llm=mock_llm)
        res = engine.grade(
            question="What is a closure?",
            expected="A nested function that is returned.",
            student_answer="A nested function that gets returned.",
            key_points=["nested function", "returned function"],
        )

        assert res.score == 1.0
        assert res.correct is True
        assert res.graded_by == "llm"

    def test_llm_failure_returns_safe_fallback(self):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.side_effect = RuntimeError("Gateway timeout")
        mock_llm.invoke.side_effect = RuntimeError("Gateway timeout")

        engine = GradingEngine(llm=mock_llm)
        res = engine.grade(
            question="What is a closure?",
            expected="A nested function.",
            student_answer="My answer here.",
        )

        assert res.score == 0.5
        assert res.correct is False
        assert res.graded_by == "fallback"
        assert "Could not grade automatically" in res.feedback

    def test_grade_answer_facade_matches_legacy_dict_contract(self):
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = LLMJudgeOutput(
            evaluations=[
                CriterionEvaluation(criterion_id="crit_1", status=CriterionStatus.MET, evidence="nested"),
            ],
            feedback="Great job.",
            missing_concepts=[],
            misconceptions=[],
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("grading.engine.build_llm", return_value=mock_llm):
            result = grade_answer(
                "What is a closure?",
                "Nested function.",
                "A nested function.",
                key_points=["nested function"],
            )

        assert isinstance(result, dict)
        assert result["correct"] is True
        assert result["score"] == 1.0
        assert "feedback" in result
        assert "missing_concept" in result
        assert "criterion_evaluations" in result
