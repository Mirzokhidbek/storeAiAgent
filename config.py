"""Loads settings from .env so the rest of the code never reads os.environ directly."""

import os

from dotenv import load_dotenv

load_dotenv()


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip()

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.75"))
MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "2"))


def require(name: str) -> str:
    """Return a required setting or raise a readable error. Never prints the value."""
    value = os.getenv(name, "").strip()
    if not value:
        # Fail early with a clear message instead of a confusing API error later.
        raise RuntimeError(f"{name} .env faylida ko'rsatilmagan. .env.example'ga qarang.")
    return value
