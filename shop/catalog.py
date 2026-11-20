"""Catalog logic: product search, details and comparison.

Shared by the LangChain @tool wrappers (Phase 3) and the MCP server (Phase 6).
Every function returns a JSON-serialisable dict; errors are {"error": "<Uzbek text>"}.
"""

import json
import sqlite3
from collections.abc import Callable

from shop.db import connect

CATEGORIES = ("noutbuk", "telefon", "quloqchin", "aksessuar")

# How many candidates the semantic step returns before SQL filtering.
# Price is no longer filtered in Chroma, so SQL may drop many candidates (too expensive,
# out of stock). The catalog is small (~40 items), so we simply ask for all of them:
# Chroma only decides the ORDER, SQL decides what is shown.
SEMANTIC_CANDIDATES = 50

# A ranker takes (query, chroma_filter, k) and returns product ids, most similar first.
# Tests pass a fake ranker so they run without Chroma or a Gemini API key.
Ranker = Callable[[str, dict | None, int], list[str]]


# ---------- helpers ----------

def _normalize_id(product_id: str) -> str:
    return product_id.strip().upper()  # "lap-001 " -> "LAP-001"


def _summary(row: sqlite3.Row) -> dict:
    """Short product card used in search results."""
    return {
        "id": row["id"],
        "name": row["name"],
        "category": row["category"],
        "brand": row["brand"],
        "price_uzs": row["price_uzs"],
        "stock": row["stock"],
        "weight_kg": row["weight_kg"],
        "description": row["description"],
    }


def _canonical_brand(conn: sqlite3.Connection, brand: str) -> str | None:
    """Map "lenovo" / "LENOVO" to the exact spelling stored in the DB ("Lenovo").
    Needed because Chroma metadata filters are case-sensitive exact matches."""
    row = conn.execute(
        "SELECT brand FROM products WHERE LOWER(brand) = LOWER(?) LIMIT 1", (brand.strip(),)
    ).fetchone()
    return row["brand"] if row else None


# ---------- semantic part (Chroma) ----------

def build_chroma_filter(category: str | None, brand: str | None) -> dict | None:
    """Turn category/brand into Chroma's `where` syntax.
    Chroma wants a single condition as-is, and two or more wrapped in {"$and": [...]}.

    Price is deliberately NOT filtered here: the price in Chroma metadata is a copy made
    at ingest time. If the price drops in SQLite, a Chroma price filter would hide the
    discounted product. Category and brand do not change, so they are safe to filter on."""
    conditions = []
    if category:
        conditions.append({"category": category})
    if brand:
        conditions.append({"brand": brand})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def semantic_rank(query: str, chroma_filter: dict | None, k: int) -> list[str]:
    """Rank products by meaning ("yengil", "talaba uchun") using the `products` collection."""
    # Imported here so that importing shop.catalog does not require Chroma/Gemini settings.
    from rag.retriever import PRODUCTS, get_store

    results = get_store(PRODUCTS).similarity_search_with_relevance_scores(
        query, k=k, filter=chroma_filter
    )
    return [doc.metadata["product_id"] for doc, _score in results]


# ---------- SQL part (source of truth) ----------

def _load_filtered(conn: sqlite3.Connection, ids: list[str], category: str | None,
                   brand: str | None, min_price: int | None, max_price: int | None,
                   in_stock_only: bool) -> dict[str, sqlite3.Row]:
    """Re-read candidates from SQLite and apply the exact filters.

    Price and stock are filtered ONLY here, because SQLite has the current values.
    Category and brand are checked again as a safety net."""
    if not ids:
        return {}
    sql = f"SELECT * FROM products WHERE id IN ({','.join('?' * len(ids))})"
    params: list = list(ids)
    if category:
        sql += " AND category = ?"
        params.append(category)
    if brand:
        sql += " AND brand = ?"
        params.append(brand)
    if min_price is not None:
        sql += " AND price_uzs >= ?"
        params.append(min_price)
    if max_price is not None:
        sql += " AND price_uzs <= ?"
        params.append(max_price)
    if in_stock_only:
        sql += " AND stock > 0"
    return {row["id"]: row for row in conn.execute(sql, params)}


# ---------- public functions (wrapped as tools) ----------

def search_products(query: str, category: str | None = None, min_price: int | None = None,
                    max_price: int | None = None, brand: str | None = None,
                    in_stock_only: bool = True, limit: int = 3,
                    ranker: Ranker | None = None) -> dict:
    """Hybrid search: Chroma ranks by meaning (filtered by category/brand only),
    SQLite filters by exact, current facts (price, stock)."""
    ranker = ranker or semantic_rank
    limit = max(1, min(limit, 10))

    if category:
        category = category.strip().lower()
        if category not in CATEGORIES:
            return {"error": f"Noma'lum kategoriya. Mavjudlari: {', '.join(CATEGORIES)}."}

    with connect() as conn:
        if brand:
            brand = _canonical_brand(conn, brand)
            if brand is None:
                return {"products": [], "message": "Bu brend mahsulotlari do'konimizda yo'q."}

        try:
            chroma_filter = build_chroma_filter(category, brand)
            ranked_ids = ranker(query, chroma_filter, SEMANTIC_CANDIDATES)
        except Exception:  # noqa: BLE001 - network/quota errors must not crash the agent
            return {"error": "Qidiruv vaqtincha ishlamayapti. Birozdan so'ng qayta urinib ko'ring."}

        rows = _load_filtered(conn, ranked_ids, category, brand, min_price, max_price,
                             in_stock_only)

    # Keep the semantic order (best match first), but only ids that survived SQL.
    products = [_summary(rows[pid]) for pid in ranked_ids if pid in rows][:limit]
    if not products:
        return {"products": [], "message": "So'rovingizga mos mahsulot topilmadi."}
    return {"products": products}


def get_product(product_id: str) -> dict:
    """Full product row with specs parsed from JSON."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM products WHERE id = ?", (_normalize_id(product_id),)
        ).fetchone()
    if row is None:
        return {"error": f"'{product_id}' kodli mahsulot topilmadi."}

    product = dict(row)
    product["specs"] = json.loads(product.pop("specs_json") or "{}")
    return product


def compare_products(product_ids: list[str]) -> dict:
    """Side-by-side view of 2-3 products (prices and stock read from SQLite now)."""
    ids = list(dict.fromkeys(_normalize_id(pid) for pid in product_ids))  # dedupe, keep order
    if not 2 <= len(ids) <= 3:
        return {"error": "Solishtirish uchun 2 yoki 3 ta turli mahsulot kodini bering."}

    compared = []
    for pid in ids:
        product = get_product(pid)
        if "error" in product:
            return product
        compared.append({
            "id": product["id"],
            "name": product["name"],
            "price_uzs": product["price_uzs"],
            "weight_kg": product["weight_kg"],
            "stock": product["stock"],
            "specs": product["specs"],
        })
    return {"products": compared}
