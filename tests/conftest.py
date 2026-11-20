"""Shared pytest fixtures (pytest loads this file automatically)."""

import random

import pytest

import shop.db
from data import seed_data


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Build a new shop.db in a temp folder and point shop.db at it.
    Tests can change data freely; the real data/shop.db is never touched."""
    db_file = tmp_path / "shop.db"
    rng = random.Random(seed_data.RANDOM_SEED)
    fixed = seed_data.fixed_orders()
    all_orders = fixed + seed_data.random_orders(
        rng, seed_data.TOTAL_ORDERS - len(fixed), {o["id"] for o in fixed})

    conn = shop.db.get_connection(db_file)
    with conn:
        shop.db.create_schema(conn)
        seed_data.insert_all(conn, all_orders)
    conn.close()

    monkeypatch.setattr(shop.db, "DB_PATH", db_file)
    return db_file
