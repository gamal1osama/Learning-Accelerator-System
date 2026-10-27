"""
tests/test_a2a_auth.py

Tests for the A2A bearer-token authentication layer.

Covers three layers of the feature (see docs/ARCHITECTURE.md):
  1. Declaration: both Agent Cards reference a shared HTTP bearer scheme
  2. Enforcement: BearerAuthMiddleware on the real quiz service app
     (Starlette TestClient — no live servers)
  3. Client side: a2a_client attaches the Authorization header and turns
     401/403 into the standard error dict

All modeled after tests/test_a2a.py conventions (mocked network where the
client is concerned; TestClient instead of live uvicorn).

Run: python -m pytest tests/test_a2a_auth.py -v
"""

import json
import os
from unittest.mock import patch, MagicMock

import pytest
from starlette.testclient import TestClient

# Reuse one token constant throughout; never commit a real secret.
TEST_TOKEN = "test-bearer-token-12345"
ENCODING = "utf-8"

JSONRPC_ENVELOPE = {
    "jsonrpc": "2.0",
    "id":      1,
    "method":  "message/send",
    "params": {
        "message": {
            "role":      "user",
            "messageId": "test-message-id",
            "parts":     [{"kind": "text", "text": json.dumps({"topic": "closures"})}],
        },
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Agent Card declaration layer
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentCardSecurityDeclaration:
    """Agent Cards must declare the shared bearer scheme via securitySchemes."""

    def test_quiz_card_declares_bearer_scheme(self):
        from a2a_services.quiz_service import QUIZ_AGENT_CARD

        assert QUIZ_AGENT_CARD.security_schemes["bearer"].root.type == "http"
        assert QUIZ_AGENT_CARD.security_schemes["bearer"].root.scheme == "bearer"

    def test_quiz_card_requires_bearer(self):
        from a2a_services.quiz_service import QUIZ_AGENT_CARD

        assert QUIZ_AGENT_CARD.security == [{"bearer": []}]

    def test_study_buddy_card_declares_bearer_scheme(self):
        from crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert STUDY_BUDDY_CARD.security_schemes["bearer"].root.type == "http"
        assert STUDY_BUDDY_CARD.security_schemes["bearer"].root.scheme == "bearer"
        assert STUDY_BUDDY_CARD.security == [{"bearer": []}]

    def test_cards_share_one_scheme_object(self):
        """Both cards reference the same scheme definition (single source)."""
        from a2a_services.quiz_service import QUIZ_AGENT_CARD
        from crewai_agent.study_buddy import STUDY_BUDDY_CARD

        assert (
            QUIZ_AGENT_CARD.security_schemes["bearer"]
            == STUDY_BUDDY_CARD.security_schemes["bearer"]
        )

    def test_scheme_serializes_to_openapi_style(self):
        """The serialized card must carry standard OpenAPI-style fields."""
        from a2a_services.quiz_service import QUIZ_AGENT_CARD

        card = json.loads(QUIZ_AGENT_CARD.model_dump_json(by_alias=True))
        scheme = card["securitySchemes"]["bearer"]
        assert scheme["type"] == "http"
        assert scheme["scheme"] == "bearer"
        assert card["security"] == [{"bearer": []}]


class TestAuthConfig:
    """Token config helpers read the environment, default to off."""

    def test_disabled_when_unset(self, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        from a2a_services.auth import auth_enabled, auth_header

        assert auth_enabled() is False
        assert auth_header() is None

    def test_disabled_when_empty(self, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", "")
        from a2a_services.auth import auth_enabled, auth_header

        assert auth_enabled() is False
        assert auth_header() is None

    def test_enabled_with_token(self, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", TEST_TOKEN)
        from a2a_services.auth import auth_enabled, auth_header

        assert auth_enabled() is True
        assert auth_header() == f"Bearer {TEST_TOKEN}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Enforcement layer (real quiz service app over TestClient)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def quiz_test_client():
    """The actual A2A app as served by quiz_service (middleware attached)."""
    from a2a_services.quiz_service import create_quiz_server

    return TestClient(create_quiz_server())


class TestBearerAuthMiddleware:
    """In-process verification against the real create_quiz_server() app."""

    def test_dev_mode_passes_through_without_token(self, quiz_test_client, monkeypatch):
        """No A2A_BEARER_TOKEN -> identical to the old unauthenticated app."""
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)

        response = quiz_test_client.post(
            "/", json=JSONRPC_ENVELOPE, headers={}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["jsonrpc"] == "2.0"

    def test_rejects_missing_header_when_auth_on(self, quiz_test_client, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", TEST_TOKEN)

        response = quiz_test_client.post("/", json=JSONRPC_ENVELOPE, headers={})
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"
        assert "agent-card" in response.json()["error"]

    def test_rejects_wrong_token(self, quiz_test_client, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", TEST_TOKEN)

        response = quiz_test_client.post(
            "/",
            json=JSONRPC_ENVELOPE,
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert response.status_code == 401

    def test_rejects_non_bearer_scheme(self, quiz_test_client, monkeypatch):
        """Basic/Digest and friends must not authenticate."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", TEST_TOKEN)

        response = quiz_test_client.post(
            "/",
            json=JSONRPC_ENVELOPE,
            headers={"Authorization": f"Basic {TEST_TOKEN}"},
        )
        assert response.status_code == 401

    def test_accepts_correct_token(self, quiz_test_client, monkeypatch):
        """Valid token reaches the handler: JSON-RPC envelope response (200)."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", TEST_TOKEN)

        response = quiz_test_client.post(
            "/",
            json=JSONRPC_ENVELOPE,
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["jsonrpc"] == "2.0"

    def test_card_endpoint_stays_public(self, quiz_test_client, monkeypatch):
        """Discovery must remain unauthenticated (clients read it to learn auth)."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", TEST_TOKEN)

        response = quiz_test_client.get("/.well-known/agent-card.json")
        assert response.status_code == 200

        card = response.json()
        assert card["securitySchemes"]["bearer"]["scheme"] == "bearer"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Client layer
# ─────────────────────────────────────────────────────────────────────────────

class TestClientAuthHeaders:
    """a2a_client mirrors the card's declared scheme it learned at runtime."""

    @patch("a2a_services.a2a_client.httpx.get")
    def test_discover_sends_no_header_in_dev_mode(self, mock_get, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        from a2a_services.a2a_client import discover_agent

        mock_get.return_value = _mock_response(200, {"name": "X"})
        discover_agent("http://localhost:9001")

        assert mock_get.call_args.kwargs["headers"] == {}

    @patch("a2a_services.a2a_client.httpx.post")
    def test_send_task_attaches_bearer_header(self, mock_post, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", TEST_TOKEN)
        from a2a_services.a2a_client import send_task

        mock_post.return_value = _mock_response(
            200, {"result": {}}
        )
        send_task("http://localhost:9001", "{}")

        headers = mock_post.call_args.kwargs["headers"]
        assert headers == {"Authorization": f"Bearer {TEST_TOKEN}"}

    @patch("a2a_services.a2a_client.httpx.post")
    def test_send_task_no_header_without_token(self, mock_post, monkeypatch):
        monkeypatch.delenv("A2A_BEARER_TOKEN", raising=False)
        from a2a_services.a2a_client import send_task

        mock_post.return_value = _mock_response(200, {"result": {}})
        send_task("http://localhost:9001", "{}")

        assert mock_post.call_args.kwargs["headers"] == {}

    @patch("a2a_services.a2a_client.httpx.post")
    def test_401_becomes_error_dict(self, mock_post, monkeypatch):
        """Auth failures surface as error dicts, never raise."""
        monkeypatch.setenv("A2A_BEARER_TOKEN", TEST_TOKEN)
        from a2a_services.a2a_client import send_task

        mock_post.return_value = _mock_response(401, {"detail": "nope"})
        result = send_task("http://localhost:9001", "{}")

        assert "error" in result
        assert "401" in result["error"]
        assert "nope" in result["error"]

    @patch("a2a_services.a2a_client.httpx.post")
    def test_403_becomes_error_dict(self, mock_post, monkeypatch):
        monkeypatch.setenv("A2A_BEARER_TOKEN", TEST_TOKEN)
        from a2a_services.a2a_client import send_task

        mock_post.return_value = _mock_response(403, {"detail": "forbidden"})
        result = send_task("http://localhost:9001", "{}")

        assert "error" in result
        assert "403" in result["error"]


def _mock_response(status_code: int, body: dict) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body
    response.raise_for_status = MagicMock()
    return response
