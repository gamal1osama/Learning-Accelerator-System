# tests/test_eval.py

import os
from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama


class OllamaJudge(DeepEvalBaseLLM):
    """
    Custom judge model using local Ollama.

    DeepEval supports custom models via the DeepEvalBaseLLM interface.
    We wrap ChatOllama to provide synchronous and async generation.

    The judge runs at temperature=0.0 for consistency. The same answer
    evaluated twice should produce the same score.
    """

    def __init__(self):
        self.model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def load_model(self):
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0,   # Deterministic for evaluation
        )

    def generate(self, prompt: str) -> str:
        return self.load_model().invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"


def get_judge_model():
    """Return an OllamaJudge, or None if deepeval is not installed."""
    try:
        return OllamaJudge()
    except ImportError:
        return None

        