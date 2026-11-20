"""One place that creates Gemini models, so settings stay consistent across the project."""

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import config


def get_chat_model(temperature: float | None = None) -> ChatGoogleGenerativeAI:
    """Gemini chat model. Pass temperature=0 for deterministic tasks like routing."""
    return ChatGoogleGenerativeAI(
        model=config.require("GEMINI_MODEL"),
        google_api_key=config.require("GOOGLE_API_KEY"),
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        # Caps answer length: protects cost and the free-tier quota.
        max_output_tokens=config.LLM_MAX_TOKENS,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Embedding model that turns text into vectors (used by RAG from Phase 2)."""
    return GoogleGenerativeAIEmbeddings(
        model=config.require("EMBEDDING_MODEL"),
        google_api_key=config.require("GOOGLE_API_KEY"),
    )
