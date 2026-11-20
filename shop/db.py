"""SQLite helpers. The database is the single source of truth for prices, stock and orders."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "shop.db"

SCHEMA = """
CREATE TABLE products (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT NOT NULL,         -- noutbuk | telefon | quloqchin | aksessuar
    brand        TEXT NOT NULL,
    price_uzs    INTEGER NOT NULL,
    stock        INTEGER NOT NULL,
    weight_kg    REAL,
    specs_json   TEXT,
    description  TEXT NOT NULL
);

CREATE TABLE customers (
    id     TEXT PRIMARY KEY,
    name   TEXT NOT NULL,
    phone  TEXT NOT NULL UNIQUE
);

CREATE TABLE orders (
    id            TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(id),
    status        TEXT NOT NULL,        -- yangi | tolandi | yolda | yetkazildi | bekor_qilindi
    created_at    TEXT NOT NULL,
    delivered_at  TEXT
);

CREATE TABLE order_items (
    order_id    TEXT NOT NULL REFERENCES orders(id),
    product_id  TEXT NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL,
    price_uzs   INTEGER NOT NULL        -- price at the time of the order
);

CREATE TABLE cart_items (
    customer_id TEXT NOT NULL REFERENCES customers(id),
    product_id  TEXT NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL,
    PRIMARY KEY (customer_id, product_id)
);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection where rows behave like dicts and foreign keys are enforced.

    DB_PATH is read at call time (not as a default argument) so tests can point it
    at a temporary copy and never touch the real shop.db.
    """
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us read row["price_uzs"] instead of row[4]
    conn.execute("PRAGMA foreign_keys = ON")  # SQLite turns this off by default
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """`with connect() as conn:` -> commit if the block succeeds, roll back if it fails,
    and always close the connection. Multi-step writes (e.g. place_order) are all-or-nothing.

    (A plain `with sqlite3.connect(...)` commits but does NOT close the connection.)"""
    conn = get_connection()
    try:
        with conn:
            yield conn
    finally:
        conn.close()
