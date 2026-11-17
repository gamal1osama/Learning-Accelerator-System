# src/grading/prompts.py
"""
Prompts for the LLM rubric judge.
"""

from __future__ import annotations
from .models import Rubric

GRADING_SYSTEM_PROMPT = """You are an objective, pedagogical grading expert evaluating a student's answer against a structured rubric.

Guidelines:
1. Evaluate each criterion independently based solely on what the student asserts as their solution.
2. Evidence: Cite exact words or brief evidence from the student answer for each criterion.
3. Status definitions:
   - "met": The student's answer clearly and correctly demonstrates this criterion.
   - "partially_met": The student demonstrates partial understanding or states the concept imprecisely.
   - "missed": The student omits, contradicts, or demonstrates no understanding of this criterion.
4. Crucial rules:
   - Do NOT award credit for a concept that appears only as part of an explanation of an incorrect or discarded approach (e.g. "WRONG: compares version strings...").
   - If the student's answer is off-topic, evasive, or explains an incorrect procedure, mark the relevant criteria as "missed".
5. Feedback & Concepts:
   - Provide 1-2 constructive, encouraging sentences of feedback.
   - List concise labels for any missing key concepts.
   - List any misconceptions or false statements identified in the answer.
"""

def format_rubric_section(rubric: Rubric) -> str:
    """Format criteria for injection into the grading prompt."""
    lines = []
    for criterion in rubric.criteria:
        lines.append(f"- ID: {criterion.id} | Requirement: {criterion.description}")
    return "\n".join(lines)


def format_grading_user_prompt(
    question: str,
    expected_answer: str,
    rubric: Rubric,
    student_answer: str,
) -> str:
    """Format the user prompt for the grading judge."""
    criteria_text = format_rubric_section(rubric)
    return (
        f"Question:\n{question}\n\n"
        f"Model Answer:\n{expected_answer}\n\n"
        f"Rubric Criteria:\n{criteria_text}\n\n"
        f"Student Answer:\n{student_answer}\n\n"
        "Please evaluate each criterion and provide constructive feedback."
    )
