# src/llm/client.py
"""
Single source of truth for LLM client construction.

Every agent in the system (LangGraph nodes, the CrewAI service, the eval
judge) builds its chat model through this module instead of constructing
clients directly. All traffic goes to the LiteLLM gateway, which fronts
the configured provider (Ollama today, optional cloud fallbacks later
via litellm_config.yaml). Agents never see provider details.
"""

import os

from langchain_openai import ChatOpenAI

DEFAULT_GATEWAY_URL = "http://localhost:4000"
DEFAULT_GATEWAY_KEY = "sk-local-gateway"
DEFAULT_GATEWAY_MODEL = "notes-helper"


def get_base_url() -> str:
    """Gateway endpoint the agents talk to."""
    return os.getenv("LITELLM_BASE_URL", DEFAULT_GATEWAY_URL).rstrip("/")


def get_api_key() -> str:
    """Bearer key sent to the gateway. Any non-empty value works unless
    the gateway enforces a master/virtual key."""
    return os.getenv("LITELLM_API_KEY", DEFAULT_GATEWAY_KEY)


def get_model_name() -> str:
    """Model alias registered in litellm_config.yaml."""
    return os.getenv("LITELLM_MODEL", DEFAULT_GATEWAY_MODEL)


def build_llm(temperature: float = 0.3, json_mode: bool = False) -> ChatOpenAI:
    """
    Build the chat model every agent uses.

    Args:
        temperature: Sampling temperature. Keep structured-output agents
            at 0.1-0.2 (see docs/MODEL_SELECTION.md).
        json_mode: When True, request JSON output from the gateway. The
            gateway translates this per provider (Ollama gets
            format=json-equivalent behaviour, OpenAI-style providers get
            response_format json_object).

    Returns a LangChain chat model. Tool binding stays with the caller:
    `build_llm(...).bind_tools(tools)`.
    """
    kwargs: dict = {
        "model": get_model_name(),
        "api_key": get_api_key(),
        "base_url": get_base_url(),
        "temperature": temperature,
    }
    if json_mode:
        # Passed via model_kwargs: in this langchain-openai release
        # response_format is not a first-class class field, setting it
        # directly triggers a UserWarning on every construction.
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)
