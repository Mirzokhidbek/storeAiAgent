"""Retrieval part of RAG: find policy chunks that are semantically close to a question."""

from pathlib import Path
from typing import TypedDict

from langchain_chroma import Chroma

import config
from agent.llm import get_embeddings

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
POLICIES = "policies"
PRODUCTS = "products"


class Chunk(TypedDict):
    text: str
    source: str   # file name, e.g. "qaytarish.md" (shown to the user as the source)
    section: str  # markdown heading the chunk came from
    score: float  # relevance: 1.0 = identical meaning, 0 = unrelated


def get_store(collection: str) -> Chroma:
    """Open (or create) a persistent Chroma collection.

    "cosine" space matters: Chroma then returns cosine *distance*, and LangChain's
    relevance score becomes 1 - distance, so higher = more similar (what we want).
    """
    return Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def search_policies(query: str, k: int | None = None, threshold: float | None = None) -> list[Chunk]:
    """Top-k policy chunks whose relevance score passes the threshold (best first)."""
    k = k or config.RAG_TOP_K
    threshold = config.RAG_SCORE_THRESHOLD if threshold is None else threshold

    results = get_store(POLICIES).similarity_search_with_relevance_scores(query, k=k)
    return [
        Chunk(text=doc.page_content, source=doc.metadata["source"],
              section=doc.metadata["section"], score=round(score, 3))
        for doc, score in results
        if score >= threshold
    ]
