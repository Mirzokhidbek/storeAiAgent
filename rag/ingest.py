"""Indexing part of RAG: split documents into chunks, embed them, store them in Chroma.

Run from the project root:  python rag/ingest.py
Safe to run many times: both collections are recreated from scratch.
"""

import sys
from pathlib import Path

# Allow `python rag/ingest.py` to import project packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import (  # noqa: E402
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.retriever import POLICIES, PRODUCTS, get_store  # noqa: E402
from shop.db import get_connection  # noqa: E402

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policies"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def split_policy(path: Path) -> list[Document]:
    """Step 1: split by '##' headings (one topic per piece).
    Step 2: only if a section is still longer than CHUNK_SIZE, split it by paragraphs."""
    by_heading = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section")]
    ).split_text(path.read_text(encoding="utf-8"))

    by_size = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],  # paragraph first, then line, then sentence
    )

    chunks = []
    for piece in by_heading:
        section = piece.metadata.get("section") or piece.metadata.get("title", "")
        for text in by_size.split_text(piece.page_content):
            # The heading is added to the text so the embedding knows the topic,
            # e.g. "14 kun" alone means little; "Ishlatilmagan mahsulotni qaytarish: 14 kun" does.
            chunks.append(Document(
                page_content=f"{section}\n{text}",
                metadata={"source": path.name, "section": section, "type": "policy"},
            ))
    return chunks


def load_policy_chunks() -> list[Document]:
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        chunks.extend(split_policy(path))
    return chunks


def load_product_chunks() -> list[Document]:
    """One chunk per product. Price is stored as metadata only for filtering;
    the price shown to users is always re-read from SQLite (hard rule)."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, category, brand, price_uzs, description FROM products"
        ).fetchall()
    return [
        Document(
            page_content=f"{r['name']} ({r['category']}, {r['brand']}). {r['description']}",
            metadata={"product_id": r["id"], "category": r["category"], "brand": r["brand"],
                      "price_uzs": r["price_uzs"], "type": "product"},
        )
        for r in rows
    ]


def rebuild_collection(name: str, docs: list[Document], ids: list[str]) -> None:
    get_store(name).delete_collection()  # idempotent: drop old vectors first
    get_store(name).add_documents(docs, ids=ids)


def main() -> None:
    policy_docs = load_policy_chunks()
    policy_ids = [f"{d.metadata['source']}#{i}" for i, d in enumerate(policy_docs)]
    rebuild_collection(POLICIES, policy_docs, policy_ids)

    product_docs = load_product_chunks()
    rebuild_collection(PRODUCTS, product_docs, [d.metadata["product_id"] for d in product_docs])

    print(f"'{POLICIES}' kolleksiyasi: {len(policy_docs)} bo'lak")
    for source in sorted({d.metadata["source"] for d in policy_docs}):
        docs = [d for d in policy_docs if d.metadata["source"] == source]
        longest = max(len(d.page_content) for d in docs)
        print(f"  {source:20} {len(docs):3} bo'lak (eng uzuni {longest} belgi)")
    print(f"'{PRODUCTS}' kolleksiyasi: {len(product_docs)} bo'lak (har mahsulotga bittadan)")


if __name__ == "__main__":
    main()
