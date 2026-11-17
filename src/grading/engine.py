# src/grading/engine.py
"""
Core GradingEngine orchestrating the multi-stage question evaluation pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from llm import build_llm
from .models import (
    CriterionEvaluation,
    CriterionStatus,
    GradeResult,
    LLMJudgeOutput,
    Rubric,
)
from .pre_grader import PreGrader
from .prompts import GRADING_SYSTEM_PROMPT, format_grading_user_prompt
from .rubric import RubricBuilder
from .scorers import ScoreAggregator

logger = logging.getLogger(__name__)


class GradingEngine:
    """
    Production question grading engine.

    Pipeline:
    1. PreGrader fast-path filter (empty, non-answers, self-declared wrong).
    2. RubricBuilder: constructs weighted rubric criteria.
    3. LLM Judge: evaluates rubric criteria with structured output schema.
    4. ScoreAggregator: deterministically calculates weighted scores and enforces consistency.
    """

    def __init__(self, llm: Any = None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        return build_llm(temperature=0.1, json_mode=True)

    def grade(
        self,
        question: str,
        expected: str,
        student_answer: str,
        key_points: list[str] | None = None,
        passing_threshold: float = 0.5,
    ) -> GradeResult:
        """
        Grade a student answer against an expected answer or rubric key points.

        Args:
            question: The question text.
            expected: The reference/model answer.
            student_answer: The answer submitted by the student.
            key_points: Optional explicit list of rubric requirements.
            passing_threshold: Score required for a passing grade (default 0.5).

        Returns:
            A validated GradeResult.
        """
        # Step 1: Pre-grading fast-path guardrails
        fast_result = PreGrader.evaluate(student_answer)
        if fast_result is not None:
            return fast_result

        # Step 2: Build structured rubric
        rubric = RubricBuilder.build(
            expected_answer=expected,
            key_points=key_points,
            passing_threshold=passing_threshold,
        )

        # Step 3: Invoke LLM Judge with structured output
        system_msg = SystemMessage(content=GRADING_SYSTEM_PROMPT)
        user_msg = HumanMessage(
            content=format_grading_user_prompt(
                question=question,
                expected_answer=expected,
                rubric=rubric,
                student_answer=student_answer,
            )
        )

        llm = self._get_llm()

        try:
            judge_output = self._invoke_judge(llm, system_msg, user_msg, rubric)
            return ScoreAggregator.aggregate(judge_output, rubric)
        except Exception as e:
            logger.error(f"[GradingEngine] Judge evaluation failed: {e}", exc_info=True)
            return GradeResult(
                score=0.5,
                correct=False,
                feedback="Could not grade automatically. Please review manually.",
                missing_concept="",
                missing_concepts=[],
                misconceptions=[],
                criterion_evaluations=[],
                graded_by="fallback",
            )

    def _invoke_judge(
        self,
        llm: Any,
        system_msg: SystemMessage,
        user_msg: HumanMessage,
        rubric: Rubric,
    ) -> LLMJudgeOutput:
        """Invoke the LLM judge using structured output or JSON validation."""
        # Check if llm has with_structured_output and is not a plain Mock without explicit setup
        if hasattr(llm, "with_structured_output"):
            try:
                structured_model = llm.with_structured_output(LLMJudgeOutput)
                result = structured_model.invoke([system_msg, user_msg])
                if isinstance(result, LLMJudgeOutput):
                    return result
                if isinstance(result, dict) and "evaluations" in result:
                    return LLMJudgeOutput.model_validate(result)
            except Exception as e:
                logger.warning(
                    f"[GradingEngine] with_structured_output invocation failed, trying raw json: {e}"
                )

        # Fallback / standard raw invocation:
        response = llm.invoke([system_msg, user_msg])
        content = response.content if hasattr(response, "content") else str(response)

        # Handle raw json string
        if isinstance(content, str):
            try:
                data = json.loads(content)
            except Exception:
                parser = PydanticOutputParser(pydantic_object=LLMJudgeOutput)
                return parser.parse(content)

            if isinstance(data, dict):
                # Standard structured output
                if "evaluations" in data:
                    return LLMJudgeOutput.model_validate(data)

                # Legacy JSON format compatibility: {"correct": true, "score": 0.85, ...}
                if "score" in data or "correct" in data:
                    score_val = float(data.get("score", 0.0))
                    status = (
                        CriterionStatus.MET
                        if score_val >= 0.7
                        else (CriterionStatus.PARTIALLY_MET if score_val >= 0.4 else CriterionStatus.MISSED)
                    )
                    evals = [
                        CriterionEvaluation(
                            criterion_id=crit.id,
                            status=status,
                            evidence="",
                            reasoning=str(data.get("feedback", "")),
                        )
                        for crit in rubric.criteria
                    ]
                    missing = str(data.get("missing_concept", "")).strip()
                    return LLMJudgeOutput(
                        evaluations=evals,
                        feedback=str(data.get("feedback", "")),
                        missing_concepts=[missing] if missing else [],
                        misconceptions=[],
                    )

            return LLMJudgeOutput.model_validate(data)

        if isinstance(content, dict):
            return LLMJudgeOutput.model_validate(content)

        raise ValueError(f"Unparseable LLM judge response: {content}")


_DEFAULT_GRADER: GradingEngine | None = None


def get_default_grader() -> GradingEngine:
    """Retrieve or create the singleton default GradingEngine."""
    global _DEFAULT_GRADER
    if _DEFAULT_GRADER is None:
        _DEFAULT_GRADER = GradingEngine()
    return _DEFAULT_GRADER
