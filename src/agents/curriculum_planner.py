import json

from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import StudyRoadmap, Topic
from llm import build_llm, get_model_name

PLANNER_SYSTEM_PROMPT = """You are an expert curriculum designer. Your job is to
create a structured study roadmap when given a learning goal.

Return ONLY valid JSON with no prose, no markdown code fences, no explanation.
The JSON must match this exact schema:

{
  "goal": "the original learning goal exactly as given",
  "total_weeks": <integer between 1 and 12>,
  "weekly_hours": <integer between 3 and 10>,
  "topics": [
    {
      "title": "Short topic name (3-6 words)",
      "description": "One clear sentence explaining what this topic covers",
      "estimated_minutes": <integer between 30 and 120>,
      "prerequisites": ["title of earlier topic if required, else empty list"],
      "status": "pending"
    }
  ]
}

Rules:
- Order topics from foundational to advanced
- prerequisites must reference earlier topic titles exactly as written
- Aim for 4 to 6 topics
- status must always be "pending"
"""






def build_planner_llm():
    """Create the LLM client for the Curriculum Planner."""
    return build_llm(temperature=0.1, json_mode=True)




def parse_roadmap_json(json_string: str) -> StudyRoadmap:
    """Parse the LLM's JSON output into a StudyRoadmap dataclass."""
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(
            "LLM returned invalid JSON.\n"
            f"Error: {e}\n"
            f"Raw output (first 300 chars): {json_string[:300]}"
        )

    # Validate required top-level fields
    required = ["goal", "total_weeks", "topics"]
    for field in required:
        if field not in data:
            raise ValueError(f"LLM JSON missing required field: '{field}'")

    if not isinstance(data["topics"], list) or len(data["topics"]) == 0:
        raise ValueError("LLM JSON 'topics' must be a non-empty list")

    # Build Topic objects
    topics = []
    for i, t in enumerate(data["topics"]):
        # Validate each topic has required fields
        for field in ["title", "description", "estimated_minutes"]:
            if field not in t:
                raise ValueError(
                    f"Topic {i} missing required field: '{field}'"
                )
        topics.append(Topic(
            title=t["title"],
            description=t["description"],
            estimated_minutes=int(t["estimated_minutes"]),
            prerequisites=t.get("prerequisites", []),
            status=t.get("status", "pending"),
        ))

    return StudyRoadmap(
        goal=data["goal"],
        total_weeks=int(data["total_weeks"]),
        weekly_hours=int(data.get("weekly_hours", 5)),
        topics=topics,
    )


def curriculum_planner_node(state: dict) -> dict:
    """
    LangGraph node: Curriculum Planner
    Reads: state["goal"]: the user's learning goal
    Writes: state["roadmap"], state["messages"], state["error"]
    """
    goal = state.get("goal", "").strip()

    if not goal:
        return {"error": "No learning goal provided. Set state['goal'] before running."}

    print(f"\n[Curriculum Planner] Building roadmap for: '{goal}'")

    llm = build_planner_llm()

    # Build the message list.
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"Create a study roadmap for this learning goal: {goal}"),
    ]

    # Call the LLM. This is a synchronous blocking call.
    print(f"[Curriculum Planner] Calling {get_model_name()}...")
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"[Curriculum Planner] LLM call failed: {e}")
        return {
            "error": f"LLM call failed: {e}",
            "messages": messages,
        }


    try:
        roadmap = parse_roadmap_json(response.content)
    except ValueError as e:
        print(f"[Curriculum Planner] Parse error: {e}")
        # Return error state, the graph can route to an error handler
        return {
            "error": str(e),
            "messages": messages + [response],
        }

    # Success, log what was created
    print(f"[Curriculum Planner] Created roadmap: {len(roadmap.topics)} topics, "
          f"{roadmap.total_weeks} weeks")
    for i, topic in enumerate(roadmap.topics, 1):
        prereqs = f" (needs: {', '.join(topic.prerequisites)})" if topic.prerequisites else ""
        print(f"  {i}. {topic.title}{prereqs}, {topic.estimated_minutes} min")


    return {
        "roadmap": roadmap,
        "messages": messages + [response],
        "error": None,
    }
