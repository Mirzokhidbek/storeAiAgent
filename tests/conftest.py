"""Shared pytest fixtures (pytest loads this file automatically)."""

import random

import pytest

import config
import shop.db
from agent import tools
from data import seed_data


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Build a new shop.db in a temp folder and point shop.db at it.
    Tests can change data freely; the real data/shop.db is never touched."""
    db_file = build_seed_db(tmp_path / "shop.db")
    monkeypatch.setattr(shop.db, "DB_PATH", db_file)
    return db_file


def build_seed_db(db_file):
    """The seed data (fixed random seed) in a new file. Same input = same database."""
    rng = random.Random(seed_data.RANDOM_SEED)
    fixed = seed_data.fixed_orders()
    all_orders = fixed + seed_data.random_orders(
        rng, seed_data.TOTAL_ORDERS - len(fixed), {o["id"] for o in fixed})

    conn = shop.db.get_connection(db_file)
    with conn:
        shop.db.create_schema(conn)
        seed_data.insert_all(conn, all_orders)
    conn.close()
    return db_file


@pytest.fixture(autouse=True)
def local_tools(monkeypatch):
    """Unit and graph tests run tools in-process (fast, uses the temp DB above).
    tests/test_mcp.py starts the real MCP servers and checks they give the same results."""
    monkeypatch.setattr(config, "TOOL_BACKEND", "local")
    tools.backend.cache_clear()   # forget a backend made under other settings
