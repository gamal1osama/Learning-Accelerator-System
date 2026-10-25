# tests/test_llm_factory.py

"""
tests/test_llm_factory.py

Unit tests for the shared LLM factory (src/llm/client.py).

Verifies that every agent gets an OpenAI-compatible client pointed at
the LiteLLM gateway with the correct alias, base URL, temperature, and
JSON-mode mapping. All tests are mocked, no gateway needed.

Run: python -m pytest tests/test_llm_factory.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock

import importlib

from llm import client as llm_client
from llm.client import build_llm, get_api_key, get_base_url, get_model_name


class TestEnvAccessors:
    """Env-var accessors: defaults and overriding."""

    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
        assert get_base_url() == "http://localhost:4000"

    def test_base_url_overridden(self, monkeypatch):
        monkeypatch.setenv("LITELLM_BASE_URL", "http://0.0.0.0:4000/")
        assert get_base_url() == "http://0.0.0.0:4000"  # trailing slash stripped

    def test_default_api_key(self, monkeypatch):
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        assert get_api_key() == "sk-local-gateway"

    def test_default_model_name(self, monkeypatch):
        monkeypatch.delenv("LITELLM_MODEL", raising=False)
        assert get_model_name() == "notes-helper"

    def test_model_name_overridden(self, monkeypatch):
        monkeypatch.setenv("LITELLM_MODEL", "notes-helper-alias2")
        assert get_model_name() == "notes-helper-alias2"


class TestBuildLLM:
    """build_llm routing: correct client type and parameters."""

    @patch("llm.client.ChatOpenAI")
    def test_returns_chat_openai_pointing_at_gateway(self, mock_chat):
        build_llm(temperature=0.3)
        kwargs = mock_chat.call_args.kwargs
        assert kwargs["model"] == "notes-helper"
        assert kwargs["base_url"] == "http://localhost:4000"
        assert kwargs["temperature"] == 0.3
        assert "model_kwargs" not in kwargs

    @patch("llm.client.ChatOpenAI")
    def test_json_mode_sets_response_format(self, mock_chat):
        build_llm(temperature=0.1, json_mode=True)
        kwargs = mock_chat.call_args.kwargs
        assert kwargs["model_kwargs"]["response_format"] == {"type": "json_object"}

    @patch("llm.client.ChatOpenAI")
    def test_no_json_mode_by_default(self, mock_chat):
        build_llm(temperature=0.4)
        assert "model_kwargs" not in mock_chat.call_args.kwargs

    @patch("llm.client.ChatOpenAI")
    def test_env_read_at_call_time(self, mock_chat, monkeypatch):
        """Env values are read per call, not captured at import time."""
        monkeypatch.setenv("LITELLM_MODEL", "alias-one")
        build_llm(temperature=0.1)
        assert mock_chat.call_args.kwargs["model"] == "alias-one"
        monkeypatch.setenv("LITELLM_MODEL", "alias-two")
        build_llm(temperature=0.1)
        assert mock_chat.call_args.kwargs["model"] == "alias-two"

    def test_returns_real_chat_openai_instance(self, monkeypatch):
        """The factory returns an object with the LangChain chat API."""
        monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
        llm = build_llm(temperature=0.2)
        # Bind tools must be available: the Explainer relies on it
        assert callable(llm.bind_tools)

    def test_module_reload_keeps_defaults_valid(self):
        """Enum-free factory stays importable and self-consistent."""
        reloaded = importlib.reload(llm_client)
        assert reloaded.get_model_name()
        assert reloaded.get_base_url().startswith("http")
