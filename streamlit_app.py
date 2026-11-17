"""
streamlit_app.py

Streamlit web interface for the Learning Accelerator.

Runs the same LangGraph graph as main.py, only the I/O mechanism
changes. Instead of terminal input/output, the app uses Streamlit
widgets and session state.

Run:
    streamlit run streamlit_app.py

Architecture:
    The app is a state machine with five screens:
    GOAL_INPUT → ROADMAP_APPROVAL → EXPLAINING → QUIZZING → COMPLETE

    A separate graph instance (ui_graph) is compiled with
    interrupt_before=["quiz_generator"] so the graph pauses before the
    quiz step and returns control to Streamlit. The UI handles quiz I/O
    directly (calling generate_questions and grade_answer), then injects
    the QuizResult into the checkpoint via graph.update_state() and
    resumes execution from progress_coach onward.

    This means:
    - Zero changes to quiz_generator_node or run_quiz()
    - The terminal interface (main.py) is completely unaffected
    - The LangGraph graph code is identical, only I/O changes

    Explanations are streamed token by token through LangGraph's
    stream_mode="messages". Only the Explainer's final answer is rendered;
    tool-calling turns update the status line and are otherwise discarded.
"""

import sys
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
# Explicit path: don't depend on cwd when streamlit runs the script
load_dotenv(override=True, dotenv_path=Path(__file__).parent / ".env")

import time
import uuid
import streamlit as st
from langgraph.types import Command

from graph.workflow import build_graph
from graph.state import initial_state, StudyRoadmap, QuizResult, QuizQuestion
from graph.streaming import (
    extract_coaching,
    extract_explanation,
    iter_explanation_updates,
)
from observability.langfuse_setup import get_langfuse_config, flush_langfuse
from agents.quiz_generator import generate_questions, grade_answer


# Page config
st.set_page_config(
    page_title="Learning Accelerator",
    page_icon="🎓",
    layout="centered",
)


# Build a UI-specific graph with interrupt_before=["quiz_generator"]
# This stops the graph before quiz_generator runs so the UI can handle
# quiz I/O without calling input() which would block Streamlit.
#
# Cached: Streamlit re-runs this whole script on every interaction. With
# Postgres checkpointing an uncached build would open a new pool (and
# re-verify the schema) on every click. Caching is safe because the graph
# object is stateless per session: all session state lives in the external
# checkpoint store behind the thread_id.

@st.cache_resource(show_spinner=False)
def get_ui_graph():
    return build_graph(
        db_path="data/checkpoints_ui.db",
        interrupt_before=["quiz_generator"],
        dsn_env_var="CHECKPOINT_DB_URL_UI",
    )


ui_graph = get_ui_graph()


# Session state initialisation

def init_state():
    defaults = {
        "screen": "GOAL_INPUT",
        "session_id": None,
        "graph_config": None,
        "roadmap": None,
        "current_topic_index": 0,
        "quiz_questions": [],
        "current_question_idx": 0,
        "graded_answers": [],
        "current_quiz_missing_concepts": [],
        "quiz_results": [],
        "weak_areas": [],
        "explanation": "",
        "topic_title": "",
        "topic_description": "",
        "coaching_message": "",
        "error": None,
        "goal": "",
        "pending_resume": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()


# Helpers

def go_to(screen: str):
    st.session_state.screen = screen


def get_roadmap() -> StudyRoadmap | None:
    r = st.session_state.roadmap
    if r is None:
        return None
    if isinstance(r, dict):
        return StudyRoadmap.from_dict(r)
    return r



def get_topic_info(result: dict, idx: int):
    """Return (title, description) for topic at idx from result or session state."""
    roadmap = result.get("roadmap") or st.session_state.roadmap
    rm = roadmap
    if isinstance(rm, dict):
        rm = StudyRoadmap.from_dict(rm)
    if rm and idx < len(rm.topics):
        topic = rm.topics[idx]
        title = topic.title if hasattr(topic, "title") else topic.get("title", "")
        desc = topic.description if hasattr(topic, "description") else topic.get("description", "")
        return title, desc
    return "", ""


def new_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def stream_until_pause(graph_input, status_label: str) -> dict:
    """
    Run ui_graph while rendering Explainer tokens in the current screen.

    Returns the checkpoint state after the graph pauses at its next
    interrupt_before point.
    """
    config = st.session_state.graph_config
    status_slot = st.empty()
    status_slot.caption(status_label)
    placeholder = st.empty()

    full_text = ""
    last_paint = 0.0
    status_visible = True
    try:
        for update in iter_explanation_updates(ui_graph, graph_input, config):
            if update.kind == "status":
                status_slot.caption(update.status)
                continue

            if status_visible:
                status_slot.empty()
                status_visible = False

            full_text = update.full_text
            if update.kind == "reset":
                placeholder.empty()
                continue

            now = time.monotonic()
            if now - last_paint >= 0.05:
                placeholder.markdown(f"{update.full_text}▌")
                last_paint = now
    finally:
        if full_text:
            placeholder.markdown(full_text)
        if status_visible:
            status_slot.empty()

    return ui_graph.get_state(config).values


# Graph interaction

def start_session(goal: str):
    """
    Start a new session. Runs: curriculum_planner → human_approval (interrupt).
    """
    session_id = str(uuid.uuid4())[:8]
    config = get_langfuse_config(session_id)
    st.session_state.session_id = session_id
    st.session_state.graph_config = config
    st.session_state.goal = goal

    state = initial_state(goal, session_id)

    with st.spinner("Building your study roadmap..."):
        result = ui_graph.invoke(state, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        go_to("ROADMAP_APPROVAL")
    elif result.get("error"):
        st.session_state.error = result["error"]
    else:
        st.session_state.error = "Unexpected: no interrupt after planner."


def approve_roadmap(approved: bool):
    """
    Resume after roadmap decision.

    If approved:
        Prepares state for topic 0, sets pending resume, and navigates
        immediately to the EXPLAINING screen so the explanation streams live there.

    If rejected:
        Graph runs: human_approval_node → curriculum_planner → interrupt
        New roadmap is shown.
    """
    decision = "yes" if approved else "no"

    if not approved:
        with st.spinner("Generating a new plan..."):
            result = ui_graph.invoke(
                Command(resume=decision),
                config=st.session_state.graph_config,
            )

        payload = result["__interrupt__"][0].value
        st.session_state.roadmap = payload.get("roadmap")
        go_to("ROADMAP_APPROVAL")
        return

    # Transition to EXPLAINING screen immediately and queue the resume command
    rm = get_roadmap()
    st.session_state.current_topic_index = 0
    st.session_state.explanation = ""
    st.session_state.coaching_message = ""
    st.session_state.quiz_questions = []
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []
    if rm and rm.topics:
        t0 = rm.topics[0]
        st.session_state.topic_title = t0.title if hasattr(t0, "title") else t0.get("title", "")
        st.session_state.topic_description = t0.description if hasattr(t0, "description") else t0.get("description", "")
    st.session_state.pending_resume = Command(resume="yes")
    go_to("EXPLAINING")


def advance_after_quiz(quiz_result: QuizResult):
    """
    After the UI-handled quiz is complete:
    1. Inject the QuizResult into the checkpoint as if quiz_generator ran.
    2. Advance topic index.
    3. If more topics, transition to EXPLAINING to stream next topic explanation.
    4. If done, go to COMPLETE.
    """
    config = st.session_state.graph_config
    existing = st.session_state.quiz_results
    all_weak = list(set(st.session_state.weak_areas + quiz_result.weak_areas))

    # Tell LangGraph that quiz_generator has already run with this result
    ui_graph.update_state(
        config,
        {
            "quiz_results": existing + [quiz_result],
            "weak_areas": all_weak,
            "roadmap": st.session_state.roadmap,
            "current_topic_index": st.session_state.current_topic_index,
            "error": None,
        },
        as_node="quiz_generator",
    )

    st.session_state.quiz_results = existing + [quiz_result]
    st.session_state.weak_areas = all_weak
    new_idx = st.session_state.current_topic_index + 1

    rm = get_roadmap()
    if rm is None or new_idx >= len(rm.topics):
        flush_langfuse()
        go_to("COMPLETE")
        return

    # Set up next topic and transition to EXPLAINING screen
    st.session_state.current_topic_index = new_idx
    st.session_state.explanation = ""
    st.session_state.coaching_message = ""
    st.session_state.quiz_questions = []
    st.session_state.graded_answers = []
    st.session_state.current_quiz_missing_concepts = []
    title, desc = get_topic_info({"roadmap": rm}, new_idx)
    st.session_state.topic_title = title
    st.session_state.topic_description = desc
    st.session_state.pending_resume = "resume_next"
    go_to("EXPLAINING")


# Screens

def screen_goal_input():
    st.title("🎓 Learning Accelerator")
    st.markdown(
        "Enter a learning goal and the system will build a personalised "
        "study plan, explain each topic using your notes, and quiz you "
        "as you go, all running locally with Ollama."
    )

    with st.form("goal_form"):
        goal = st.text_input(
            "What do you want to learn?",
            placeholder="e.g. Learn Python closures and decorators from scratch",
        )
        submitted = st.form_submit_button("Build Study Plan →", type="primary")

    if submitted:
        if not goal.strip():
            st.error("Please enter a learning goal.")
        else:
            start_session(goal.strip())
            st.rerun()

    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Try again"):
            st.session_state.error = None
            st.rerun()


def screen_roadmap_approval():
    st.title("📋 Your Study Plan")
    rm = get_roadmap()

    if rm is None:
        st.error("No roadmap found.")
        if st.button("Start over"):
            new_session()
            st.rerun()
        return

    st.markdown(f"**Goal:** {rm.goal}")
    st.markdown(f"**Duration:** {rm.total_weeks} weeks @ {rm.weekly_hours} hrs/week")
    st.markdown("---")

    for i, topic in enumerate(rm.topics, 1):
        title = topic.title if hasattr(topic, "title") else topic.get("title", "")
        desc = topic.description if hasattr(topic, "description") else topic.get("description", "")
        mins = topic.estimated_minutes if hasattr(topic, "estimated_minutes") else topic.get("estimated_minutes", "?")
        prereqs = topic.prerequisites if hasattr(topic, "prerequisites") else topic.get("prerequisites", [])
        prereq_text = f" *(needs: {', '.join(prereqs)})*" if prereqs else ""
        st.markdown(f"**{i}. {title}**, {mins} min{prereq_text}")
        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{desc}")

    st.markdown("---")
    st.markdown("Does this study plan look good?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, start studying", type="primary", use_container_width=True):
            approve_roadmap(True)
            st.rerun()
    with col2:
        if st.button("🔄 No, generate a different plan", use_container_width=True):
            approve_roadmap(False)
            st.rerun()


def screen_explaining():
    rm = get_roadmap()
    total = len(rm.topics) if rm else 1
    idx = st.session_state.current_topic_index

    st.progress(idx / total, text=f"Topic {idx + 1} of {total}")
    st.title(f"📖 {st.session_state.topic_title}")
    st.caption(st.session_state.topic_description)
    st.markdown("---")

    if st.session_state.coaching_message:
        st.info(f"💬 **Coach:** {st.session_state.coaching_message}")
        st.markdown("---")

    st.markdown("### Explanation")

    # If streaming is needed for this topic:
    if st.session_state.get("pending_resume"):
        resume_cmd = st.session_state.pending_resume
        st.session_state.pending_resume = None  # Consume pending resume

        graph_input = resume_cmd if isinstance(resume_cmd, Command) else None
        status_label = (
            "Explainer is researching notes and preparing your explanation..."
            if graph_input is not None
            else "Getting coaching feedback and researching next topic..."
        )

        result = stream_until_pause(graph_input, status_label)

        messages = result.get("messages", [])
        explanation = extract_explanation(messages)
        st.session_state.explanation = explanation

        coaching = extract_coaching(messages, explanation)
        st.session_state.coaching_message = coaching

        roadmap = result.get("roadmap") or st.session_state.roadmap
        st.session_state.roadmap = roadmap

        with st.spinner("Preparing quiz questions..."):
            questions = generate_questions(
                st.session_state.topic_title,
                explanation,
                n=3,
            )
            st.session_state.quiz_questions = questions
            st.session_state.current_question_idx = 0
            st.session_state.graded_answers = []
            st.session_state.current_quiz_missing_concepts = []

        st.rerun()

    # Display complete explanation
    if st.session_state.explanation:
        st.markdown(st.session_state.explanation)
    else:
        st.warning("No explanation available, starting quiz with topic context.")

    st.markdown("---")
    st.markdown(f"**Ready to test your knowledge of *{st.session_state.topic_title}*?**")

    if st.button("Start Quiz →", type="primary"):
        st.session_state.coaching_message = ""
        go_to("QUIZZING")
        st.rerun()


def screen_quizzing():
    questions = st.session_state.quiz_questions
    q_idx = st.session_state.current_question_idx
    total_q = len(questions)
    rm = get_roadmap()
    total_topics = len(rm.topics) if rm else 1
    topic_idx = st.session_state.current_topic_index

    st.progress(topic_idx / total_topics, text=f"Topic {topic_idx + 1} of {total_topics}")
    # All-done branch shows question idx == total, so clamp the display
    shown_q = min(q_idx, total_q - 1) if total_q else 0
    if total_q > 0:
        done_text = (f"Question {q_idx} of {total_q} completed"
                     if q_idx >= total_q
                     else f"Question {q_idx + 1} of {total_q}")
        st.progress(min(1.0, q_idx / total_q), text=done_text)

    st.title(f"🧠 Quiz: {st.session_state.topic_title}")
    st.markdown("---")

    # Show already-graded answers
    for i, graded in enumerate(st.session_state.graded_answers):
        status = "✅" if graded.correct else "❌"
        with st.expander(f"{status} Q{i+1}: {graded.question[:80]}...", expanded=False):
            st.markdown(f"**Your answer:** {graded.user_answer}")
            st.markdown(f"**Score:** {graded.score:.0%}")
            st.markdown(f"**Feedback:** {graded.feedback}")

    # Current question
    if q_idx < total_q:
        q = questions[q_idx]
        question_text = q.get("question", "")
        difficulty = q.get("difficulty", "medium")

        st.markdown(f"**Question {q_idx + 1} [{difficulty}]:**")
        st.markdown(question_text)

        with st.form(f"answer_form_{q_idx}"):
            answer = st.text_area(
                "Your answer:",
                placeholder="Type your answer here...",
                height=120,
                key=f"answer_input_{q_idx}",
            )
            submitted = st.form_submit_button("Submit Answer →", type="primary")

        if submitted:
            user_answer = answer.strip() or "(no answer provided)"
            expected = q.get("expected_answer", "")

            with st.spinner("Grading your answer..."):
                grade = grade_answer(
                    question_text,
                    expected,
                    user_answer,
                    q.get("key_points", []),
                )

            graded_q = QuizQuestion(
                question=question_text,
                expected_answer=expected,
                user_answer=user_answer,
                correct=bool(grade.get("correct", False)),
                feedback=grade.get("feedback", ""),
                score=float(grade.get("score", 0.0)),
            )
            st.session_state.graded_answers.append(graded_q)
            # Capture the LLM's identified missing concept (short topic-area phrase)
            # rather than the full feedback sentence, so the "Topics to Revisit"
            # list in screen_complete shows useful labels.
            missing = grade.get("missing_concept", "").strip()
            if missing:
                st.session_state.current_quiz_missing_concepts.append(missing)
            st.session_state.current_question_idx = q_idx + 1
            st.rerun()

    else:
        # All questions done
        st.markdown("---")
        graded = st.session_state.graded_answers
        avg_score = sum(q.score for q in graded) / len(graded) if graded else 0.0
        # Deduplicated list of identified weak areas across all questions in this quiz
        weak_areas = list(dict.fromkeys(
            st.session_state.current_quiz_missing_concepts
        ))

        st.success("✅ Quiz complete!")
        st.metric("Your score", f"{avg_score:.0%}")

        quiz_result = QuizResult(
            topic=st.session_state.topic_title,
            questions=graded,
            score=avg_score,
            weak_areas=weak_areas,
        )

        if st.button("Continue →", type="primary"):
            advance_after_quiz(quiz_result)
            st.rerun()


def screen_complete():
    st.title("🎉 Session Complete!")
    st.markdown("---")

    rm = get_roadmap()
    quiz_results = st.session_state.quiz_results

    if rm:
        st.markdown(f"**Goal:** {rm.goal}")

    if quiz_results:
        avg = sum(
            (r.score if hasattr(r, "score") else r.get("score", 0))
            for r in quiz_results
        ) / len(quiz_results)
        st.metric("Overall Average", f"{avg:.0%}")
        st.markdown("---")
        st.markdown("### Results by Topic")
        for r in quiz_results:
            if isinstance(r, dict):
                r = QuizResult.from_dict(r)
            status = "✅" if r.score >= 0.5 else "❌"
            weak = f", review: {', '.join(r.weak_areas[:2])}" if r.weak_areas else ""
            st.markdown(f"{status} **{r.topic}**: {r.score:.0%}{weak}")

    if st.session_state.weak_areas:
        st.markdown("---")
        st.markdown("### Topics to Revisit")
        for w in st.session_state.weak_areas[:5]:
            st.markdown(f"- {w}")

    st.markdown("---")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")
    st.caption("Resume via terminal: `python main.py --resume <session-id>`")

    if st.button("🔄 Start a New Session", type="primary"):
        new_session()
        st.rerun()


# Error banner

def display_error():
    if st.session_state.error:
        st.error(f"Something went wrong: {st.session_state.error}")
        if st.button("← Start over"):
            new_session()
            st.rerun()


# Router

screen = st.session_state.screen

if screen == "GOAL_INPUT":
    screen_goal_input()
elif screen == "ROADMAP_APPROVAL":
    display_error()
    screen_roadmap_approval()
elif screen == "EXPLAINING":
    display_error()
    screen_explaining()
elif screen == "QUIZZING":
    display_error()
    screen_quizzing()
elif screen == "COMPLETE":
    screen_complete()
else:
    st.error(f"Unknown screen: {screen}")
    if st.button("Reset"):
        new_session()
        st.rerun()
