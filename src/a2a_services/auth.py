# src/a2a_services/auth.py

"""
Shared bearer-token authentication for the A2A services.

How it works
------------
A2A declares authentication like any OpenAPI service: the Agent Card
includes a `securitySchemes` entry and the caller sends the credential in
the HTTP "Authorization" header on every request (not inside the JSON-RPC
envelope). This module holds the single source of truth for that scheme:

  - BEARER_SECURITY_SCHEME / BEARER_SECURITY  -> declared on both Agent Cards
  - BearerAuthMiddleware                      -> enforced on both servers
  - auth_header()/auth_enabled()              -> used by the A2A client

Configuration (from the environment / .env):
  A2A_BEARER_TOKEN   The token callers must present.
                     Empty or unset = auth disabled (safe local dev mode:
                     the middleware is a pass-through and the cards declare
                     no security). This keeps the default dev workflow
                     unchanged; production sets the token.

  A2A_PUBLIC_PATHS   Optional comma-separated extra paths that stay public
                     (the well-known Agent Card path is always public so
                     callers can discover the security scheme).
"""

import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from a2a.types import (
    HTTPAuthSecurityScheme,
    SecurityScheme,
)

# Bearer scheme definition (OpenAPI-style, serialized onto the Agent Card)
# ---------------------------------------------------------------------------
# Matches what the middleware enforces below: scheme=bearer over the standard
# "Authorization" header (RFC 7235). Token format hint is generic: services
# accept any opaque token from A2A_BEARER_TOKEN, not necessarily JWT.

BEARER_SCHEME_NAME = "bearer"

BEARER_SECURITY_SCHEME = SecurityScheme(
    root=HTTPAuthSecurityScheme(
        type="http",
        scheme="bearer",
        description=(
            "Bearer token from the A2A_BEARER_TOKEN secret, sent in the "
            "HTTP Authorization header on every request."
        ),
    )
)

# The card-level requirement: all RPC calls must present this scheme.
BEARER_SECURITY: list[dict] = [{BEARER_SCHEME_NAME: []}]


# Configuration
# ---------------------------------------------------------------------------

AUTH_TOKEN_ENV_VAR = "A2A_BEARER_TOKEN"

# The Agent Card must stay discoverable: A2A clients fetch it first to learn
# which security scheme to apply.
AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"

_EXTRA_PUBLIC = os.getenv("A2A_PUBLIC_PATHS", "")
PUBLIC_PATHS = tuple(
    "/" + p.lstrip("/")
    for p in _EXTRA_PUBLIC.split(",")
    if p.strip()
) + (AGENT_CARD_WELL_KNOWN_PATH,)


def get_auth_token() -> str:
    """The configured bearer token ('' when auth is disabled)."""
    return os.getenv(AUTH_TOKEN_ENV_VAR, "").strip()


def auth_enabled() -> bool:
    """True when a bearer token is configured (auth is enforced)."""
    return bool(get_auth_token())


def auth_header() -> str | None:
    """Authorization header value for outgoing calls, or None if disabled."""
    token = get_auth_token()
    if not token:
        return None
    return f"Bearer {token}"


# Server-side middleware
# ---------------------------------------------------------------------------

class BearerAuthMiddleware(BaseHTTPMiddleware):
    """
    Enforces the bearer scheme (declared in the Agent Card's securitySchemes)
    on A2A RPC traffic while keeping discovery endpoints public.

    When no token is configured the middleware is a pass-through, so the
    identical app binary serves unauthenticated local development.
    """

    async def dispatch(self, request: Request, call_next):
        if not auth_enabled() or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization")
        presented = header[7:].strip() if header and header.startswith("Bearer ") else ""
        expected = get_auth_token()
        if presented and secrets.compare_digest(presented.encode(), expected.encode()):
            request.state.a2a_user = _RemoteCaller()
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={
                "error": "Unauthorized: missing or invalid bearer token. "
                         "Send 'Authorization: Bearer <token>' (see the "
                         "agent card's securitySchemes at "
                         f"{request.url.scheme}://{request.url.netloc}"
                         f"{AGENT_CARD_WELL_KNOWN_PATH}).",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )


class _RemoteCaller:
    """Minimal authenticated-user record stored on request.state."""

    is_authenticated = True
    user_name = "bearer-caller"


# Health-check friendly error digest for the client
# ---------------------------------------------------------------------------

def unauthorized_digest(status_code: int, body: object = None) -> str:
    """Short message used by the client helpers when auth fails (401/403)."""
    detail = ""
    if isinstance(body, dict):
        detail = body.get("error") or body.get("detail") or ""
    detail = f": {str(detail)[:200]}" if detail else ""
    return f"Authentication rejected (HTTP {status_code}){detail}"
