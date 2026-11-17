# src/agents/quiz_generator.py
"""
Quiz Generator Agent.

Generates conceptual quiz questions from study materials and coordinates
quiz sessions. Evaluation and grading are delegated to the production
GradingEngine in src/grading/.
"""

import json
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from grading import get_default_grader
from graph.state import QuizQuestion, QuizResult, get_current_topic
from llm import build_llm

GENERATION_PROMPT = """You are a quiz designer for a student learning programming.

Given a topic and explanation, generate {n} quiz questions that test
genuine understanding, not just the ability to repeat memorized phrases.

Good questions require the student to:
  - Apply a concept to a new situation
  - Explain WHY something works, not just WHAT it does
  - Identify edge cases or common mistakes
  - Compare related concepts

Return ONLY valid JSON with no prose or markdown:
{{
  "questions": [
    {{
      "question": "Clear, specific question text ending with ?",
      "expected_answer": "Model answer in 1-3 sentences",
      "key_points": [
        "Specific concept or step the answer must mention",
        "Second required concept or step"
      ],
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Rules:
  - Include at least one question about a common mistake or gotcha
  - expected_answer should be concise but complete
  - Provide 2-4 key_points per question
  - Each key point must be independently checkable and together cover
    the expected_answer, not just repeat it
  - Avoid yes/no questions. Ask for explanation or demonstration
"""


def _normalize_questions(questions: list) -> list[dict]:
    """Validate and clean generated questions structure."""
    normalized = []
    for question in questions:
        if not isinstance(question, dict):
            continue

        question_text = str(question.get("question", "")).strip()
        expected = str(question.get("expected_answer", "")).strip()
        if not question_text or not expected:
            continue

        key_points = [
            str(kp).strip()
            for kp in question.get("key_points", [])
            if str(kp).strip()
        ]

        normalized.append({
            "question": question_text,
            "expected_answer": expected,
            "key_points": key_points,
            "difficulty": str(question.get("difficulty", "medium")).strip() or "medium",
        })

    return normalized


def generate_questions(topic: str, explanation: str, n: int = 3) -> list[dict]:
    """Generate n quiz questions with rubric key points from the Explainer's output."""
    llm = build_llm(temperature=0.4, json_mode=True)

    prompt = GENERATION_PROMPT.format(n=n)
    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Topic: {topic}\n\nExplanation:\n{explanation}"),
        ])
        data = json.loads(response.content)
        questions = data.get("questions", [])
        if questions and isinstance(questions, list):
            normalized = _normalize_questions(questions)
            if normalized:
                return normalized
    except Exception as e:
        print(f"[Quiz Generator] LLM call failed during question generation: {e}")

    # Fallback: one generic question with explicit rubric points
    return [{
        "question": f"In your own words, explain the key concept of {topic} and why it matters.",
        "expected_answer": "A clear explanation demonstrating conceptual understanding.",
        "key_points": [
            "Explains the core idea in their own words",
            "Explains why the idea matters",
        ],
        "difficulty": "medium",
    }]


def grade_answer(
    question: str,
    expected: str,
    student_answer: str,
    key_points: list[str] | None = None,
) -> dict:
    """
    Grade a student's answer using the production GradingEngine.

    Maintains backward compatibility with all callers expecting a dictionary.
    """
    grader = get_default_grader()
    result = grader.grade(
        question=question,
        expected=expected,
        student_answer=student_answer,
        key_points=key_points,
    )
    return result.to_dict()


def run_quiz(topic: str, explanation: str) -> QuizResult:
    """Run an interactive quiz session in the terminal."""
    print(f"\n{'='*60}")
    print(f"Quiz: {topic}")
    print(f"{'='*60}")
    print("Answer each question in your own words. Press Enter to submit.\n")

    questions_data = generate_questions(topic, explanation, n=3)
    graded_questions = []
    total_score = 0.0
    weak_areas = []

    for i, q_data in enumerate(questions_data, 1):
        question_text = q_data["question"]
        expected = q_data["expected_answer"]
        difficulty = q_data.get("difficulty", "medium")

        print(f"Question {i} [{difficulty}]: {question_text}")
        user_answer = input("Your answer: ").strip()
        if not user_answer:
            user_answer = "(no answer provided)"

        print("Grading...")
        grade = grade_answer(
            question_text,
            expected,
            user_answer,
            q_data.get("key_points", []),
        )

        score = float(grade.get("score", 0.0))
        correct = bool(grade.get("correct", False))
        feedback = grade.get("feedback", "")
        missing = grade.get("missing_concept", "")

        total_score += score
        status = "✓" if correct else "✗"
        print(f"{status} Score: {score:.0%}. {feedback}\n")

        if missing:
            weak_areas.append(missing)

        graded_questions.append(QuizQuestion(
            question=question_text,
            expected_answer=expected,
            user_answer=user_answer,
            correct=correct,
            feedback=feedback,
            score=score,
        ))

    avg_score = total_score / len(questions_data) if questions_data else 0.0
    correct_count = sum(1 for q in graded_questions if q.correct)

    print(f"{'='*60}")
    print(f"Quiz complete! Score: {avg_score:.0%} ({correct_count}/{len(graded_questions)} correct)")
    if weak_areas:
        print(f"Areas to review: {', '.join(set(weak_areas))}")
    print(f"{'='*60}\n")

    return QuizResult(
        topic=topic,
        questions=graded_questions,
        score=avg_score,
        weak_areas=list(set(weak_areas)),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def quiz_generator_node(state: dict) -> dict:
    """
    LangGraph node: Quiz Generator

    Reads:  state["roadmap"], state["current_topic_index"], state["messages"]
    Writes: state["quiz_results"], state["weak_areas"], state["error"]
    """
    topic = get_current_topic(state)
    if topic is None:
        return {"error": "No current topic. Curriculum Planner must run first"}

    # Extract the Explainer's final response from message history.
    messages = state.get("messages", [])
    explanation = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            text = str(msg.content).strip()
            is_meta = len(text) < 250 and any(
                p in text.lower()
                for p in ("stored the explanation", "session memory", "stored in memory", "saved the explanation")
            )
            if not is_meta:
                explanation = text
                break

    if not explanation:
        print("[Quiz Generator] Warning: no explanation found, generating generic quiz")
        explanation = f"Topic: {topic.title}. {topic.description}"

    print(f"\n[Quiz Generator] Generating quiz for: '{topic.title}'")
    quiz_result = run_quiz(topic.title, explanation)

    existing_results = state.get("quiz_results", [])
    all_weak_areas = list(set(
        state.get("weak_areas", []) + quiz_result.weak_areas
    ))

    return {
        "quiz_results": existing_results + [quiz_result],
        "weak_areas": all_weak_areas,
        "error": None,
    }
