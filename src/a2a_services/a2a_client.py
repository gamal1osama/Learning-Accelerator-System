# src/a2a_services/a2a_client.py

import httpx
import json
import uuid
import os

from a2a_services.auth import auth_header, unauthorized_digest

QUIZ_SERVICE_URL  = os.getenv("QUIZ_SERVICE_URL",  "http://localhost:9001")
STUDY_BUDDY_URL   = os.getenv("STUDY_BUDDY_URL",   "http://localhost:9002")
DEFAULT_TIMEOUT   = 120.0


def _request_headers() -> dict:
    """
    Headers for outgoing A2A calls.

    Mirrors the services' declared securitySchemes: a Bearer token in the
    HTTP Authorization header (from A2A_BEARER_TOKEN). Empty dict when auth
    is disabled, so local dev calls stay identical to before.
    """
    header = auth_header()
    if not header:
        return {}
    return {"Authorization": header}


def discover_agent(base_url: str) -> dict:
    """Fetch an Agent Card to discover capabilities. Returns {} if unreachable."""
    card_url = f"{base_url.rstrip('/')}/.well-known/agent-card.json"
    try:
        response = httpx.get(card_url, headers=_request_headers(), timeout=5.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[A2A Client] Cannot reach {card_url}: {e}")
        return {}


def send_task(
    base_url: str,
    message_text: str,
    task_id: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """
    Submit a task to an A2A agent via JSON-RPC 2.0 (message/send).

    The JSON-RPC envelope is what A2A requires. Your caller doesn't
    need to know about the envelope. It just passes a text payload.
    Pass an explicit task_id when you need an idempotency key; otherwise
    a UUID is generated for you.
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "message/send",
        "params": {
            "message": {
                "role":       "user",
                "messageId":  task_id or str(uuid.uuid4()),
                "parts":      [{"kind": "text", "text": message_text}],
            },
        },
    }

    url = f"{base_url.rstrip('/')}/"
    try:
        response = httpx.post(url, json=payload, headers=_request_headers(), timeout=timeout)

        # Auth failures never raise: surface them as error dicts like the
        # rest of this client's failure paths.
        if response.status_code in (401, 403):
            try:
                body = response.json()
            except Exception:
                body = None
            return {"error": unauthorized_digest(response.status_code, body)}

        response.raise_for_status()
        data = response.json()

        # Extract text from the A2A response envelope:
        # result.parts[0].text  (result is a Message for message/send)
        result    = data.get("result", {})
        for part in result.get("parts", []):
            if part.get("kind") == "text":
                try:
                    return json.loads(part["text"])
                except json.JSONDecodeError:
                    return {"text": part["text"]}

        # Legacy fallbacks: artifacts / status message
        artifacts = result.get("artifacts", [])
        if artifacts:
            for part in artifacts[0].get("parts", []):
                if part.get("kind") in ("text", "type") or part.get("type") == "text":
                    try:
                        return json.loads(part.get("text", "{}"))
                    except json.JSONDecodeError:
                        return {"text": part.get("text", "")}

        status = result.get("status", {})
        for part in status.get("message", {}).get("parts", []):
            if part.get("kind") == "text":
                try:
                    return json.loads(part["text"])
                except json.JSONDecodeError:
                    return {"text": part["text"]}

        return result

    except httpx.TimeoutException:
        return {"error": f"Service timed out after {timeout}s"}
    except httpx.ConnectError:
        return {"error": f"Cannot connect to {url}"}
    except Exception as e:
        return {"error": f"A2A task failed: {e}"}


def delegate_quiz_task(
    topic: str,
    explanation: str,
    answers: list[str] | None = None,
    quiz_service_url: str = QUIZ_SERVICE_URL,
) -> dict:
    """High-level helper: delegate a quiz task to the Quiz A2A service."""
    payload = json.dumps({
        "topic":       topic,
        "explanation": explanation,
        "answers":     answers or [],
    })
    return send_task(quiz_service_url, payload)


def request_study_assistance(
    topic: str,
    explanation: str = "",
    weak_areas: list[str] | None = None,
    study_buddy_url: str = STUDY_BUDDY_URL,
) -> dict:
    """High-level helper: request study help from the CrewAI Study Buddy."""
    payload = json.dumps({
        "topic":       topic,
        "explanation": explanation,
        "weak_areas":  weak_areas or [],
    })
    return send_task(study_buddy_url, payload)


def is_quiz_service_available(quiz_service_url: str = QUIZ_SERVICE_URL) -> bool:
    """Quick health check: is the quiz service reachable?"""
    return bool(discover_agent(quiz_service_url))


def is_study_buddy_available(study_buddy_url: str = STUDY_BUDDY_URL) -> bool:
    """Quick health check: is the CrewAI study buddy reachable?"""
    return bool(discover_agent(study_buddy_url))


