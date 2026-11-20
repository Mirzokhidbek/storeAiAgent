"""Unit tests for shop/ logic and agent/tools.py wrappers.

Run:  python -m pytest -v
No API key needed: each test gets a fresh temporary copy of the seed data, and the
semantic (Chroma) step of search_products is replaced by a fake ranker.
"""

import pytest

import shop.db
from agent import tools
from data import seed_data
from shop import catalog, orders

ALL_PRODUCT_IDS = [p[0] for p in seed_data.PRODUCTS]
LAPTOP_PRICES = {p[0]: p[4] for p in seed_data.PRODUCTS if p[2] == "noutbuk"}


# The fresh_db fixture (a temp copy of the seed data) lives in tests/conftest.py.


def fake_ranker(query, chroma_filter, k):
    """Pretends to be Chroma, but ignores the filter and returns EVERY product.
    So whatever is filtered out in the results was filtered by SQL — the source of truth."""
    return ALL_PRODUCT_IDS[:]


def stock_of(product_id: str) -> int:
    return catalog.get_product(product_id)["stock"]


# ======================= catalog =======================

def test_price_filter_keeps_only_cheaper_laptops():
    result = catalog.search_products("noutbuk", category="noutbuk", max_price=6_000_000,
                                     limit=10, ranker=fake_ranker)
    products = result["products"]
    assert products, "6 mln gacha noutbuklar bor bo'lishi kerak"
    assert all(p["category"] == "noutbuk" for p in products)
    assert all(p["price_uzs"] <= 6_000_000 for p in products)
    # Prices must equal the DB values exactly (never invented).
    assert all(p["price_uzs"] == LAPTOP_PRICES[p["id"]] for p in products)


def test_min_price_filter():
    products = catalog.search_products("telefon", category="telefon", min_price=10_000_000,
                                       limit=10, ranker=fake_ranker)["products"]
    assert {p["id"] for p in products} == {"PHN-003", "PHN-004"}


def test_out_of_stock_excluded_by_default():
    # LAP-003 costs 4.3 mln and has stock = 0.
    ids = [p["id"] for p in catalog.search_products(
        "arzon noutbuk", category="noutbuk", max_price=6_000_000, limit=10,
        ranker=fake_ranker)["products"]]
    assert "LAP-003" not in ids
    assert all(stock_of(pid) > 0 for pid in ids)


def test_out_of_stock_included_when_asked():
    ids = [p["id"] for p in catalog.search_products(
        "noutbuk", category="noutbuk", max_price=6_000_000, in_stock_only=False, limit=10,
        ranker=fake_ranker)["products"]]
    assert "LAP-003" in ids


def test_stock_is_read_from_sqlite_at_search_time():
    # Stock changes after indexing -> search must notice it, because stock is never cached.
    with shop.db.connect() as conn:
        conn.execute("UPDATE products SET stock = 0 WHERE id = 'LAP-008'")
    ids = [p["id"] for p in catalog.search_products(
        "yengil noutbuk", category="noutbuk", limit=10, ranker=fake_ranker)["products"]]
    assert "LAP-008" not in ids


def test_semantic_order_is_kept_and_limit_applied():
    def ranker(query, chroma_filter, k):
        return ["LAP-008", "LAP-004", "LAP-001", "LAP-002"]  # "most similar" first
    products = catalog.search_products("yengil noutbuk", limit=2, ranker=ranker)["products"]
    assert [p["id"] for p in products] == ["LAP-008", "LAP-004"]


def test_nothing_matches_returns_empty_list_not_invented_products():
    result = catalog.search_products("noutbuk", category="noutbuk", max_price=100_000,
                                     ranker=fake_ranker)
    assert result["products"] == []
    assert "message" in result


def test_chroma_filter_is_built_correctly():
    assert catalog.build_chroma_filter(None, None) is None
    assert catalog.build_chroma_filter("noutbuk", None) == {"category": "noutbuk"}
    assert catalog.build_chroma_filter("noutbuk", "Lenovo") == {"$and": [
        {"category": "noutbuk"}, {"brand": "Lenovo"}]}


def test_price_is_never_sent_to_chroma():
    seen = {}

    def ranker(query, chroma_filter, k):
        seen["filter"] = chroma_filter
        return ALL_PRODUCT_IDS[:]

    catalog.search_products("noutbuk", category="noutbuk", min_price=1, max_price=6_000_000,
                            ranker=ranker)
    assert "price_uzs" not in str(seen["filter"])


def chroma_like_ranker(query, chroma_filter, k):
    """A fake that behaves like the real Chroma collection: it applies the `where`
    filter to metadata copied at ingest time (the ORIGINAL seed prices).
    If shop code sent a price filter here, it would be checked against stale prices."""
    stale_metadata = {p[0]: {"category": p[2], "brand": p[3], "price_uzs": p[4]}
                      for p in seed_data.PRODUCTS}

    def matches(meta, cond):
        if "$and" in cond:
            return all(matches(meta, c) for c in cond["$and"])
        (field, rule), = cond.items()
        if isinstance(rule, dict):
            value = meta[field]
            return (value >= rule.get("$gte", value)) and (value <= rule.get("$lte", value))
        return meta[field] == rule

    return [pid for pid, meta in stale_metadata.items()
            if chroma_filter is None or matches(meta, chroma_filter)][:k]


def test_discounted_product_is_found_even_if_chroma_has_old_price():
    # LAP-005 costs 7 900 000 in Chroma metadata. It goes on sale in SQLite only.
    with shop.db.connect() as conn:
        conn.execute("UPDATE products SET price_uzs = 5500000 WHERE id = 'LAP-005'")

    products = catalog.search_products("noutbuk", category="noutbuk", max_price=6_000_000,
                                       limit=20, ranker=chroma_like_ranker)["products"]
    by_id = {p["id"]: p for p in products}
    assert "LAP-005" in by_id
    assert by_id["LAP-005"]["price_uzs"] == 5_500_000  # the new price, read from SQLite


def test_brand_is_case_insensitive():
    seen = {}

    def ranker(query, chroma_filter, k):
        seen["filter"] = chroma_filter
        return ALL_PRODUCT_IDS[:]

    products = catalog.search_products("noutbuk", brand="lenovo", limit=10, ranker=ranker)["products"]
    assert seen["filter"] == {"brand": "Lenovo"}  # Chroma needs the exact DB spelling
    assert products and all(p["brand"] == "Lenovo" for p in products)


def test_unknown_category_is_an_error():
    assert "error" in catalog.search_products("x", category="televizor", ranker=fake_ranker)


def test_search_failure_returns_error_instead_of_crashing():
    def broken_ranker(query, chroma_filter, k):
        raise ConnectionError("Gemini API unavailable")
    assert "error" in catalog.search_products("noutbuk", ranker=broken_ranker)


def test_get_product_parses_specs():
    product = catalog.get_product("lap-006")  # lower case works too
    assert product["price_uzs"] == 12_500_000
    assert product["specs"]["cpu"] == "Apple M2"
    assert "specs_json" not in product


def test_get_product_unknown():
    assert "error" in catalog.get_product("LAP-999")


def test_compare_products_uses_db_values():
    result = catalog.compare_products(["LAP-001", "LAP-002"])
    by_id = {p["id"]: p for p in result["products"]}
    assert by_id["LAP-001"]["price_uzs"] == 5_400_000
    assert by_id["LAP-001"]["weight_kg"] == 1.62
    assert by_id["LAP-002"]["price_uzs"] == 4_900_000
    assert by_id["LAP-002"]["weight_kg"] == 1.74


@pytest.mark.parametrize("ids", [["LAP-001"], ["LAP-001", "LAP-001"], ["A", "B", "C", "D"]])
def test_compare_needs_two_or_three_different_products(ids):
    assert "error" in catalog.compare_products(ids)


# ======================= orders: read =======================

def test_own_order_status():
    result = orders.get_order_status("C-001", "1234")
    assert result["status"] == "yetkazildi"
    assert result["delivered_at"] is not None


def test_customer_cannot_read_someone_elses_order():
    result = orders.get_order_status("C-001", "5678")  # 5678 belongs to C-002
    assert "error" in result
    assert "status" not in result
    # Same answer as for a non-existent order, so existence is not leaked.
    assert result == orders.get_order_status("C-001", "0000")


def test_logged_out_user_must_log_in():
    for result in [orders.get_order_status(None, "1234"), orders.list_my_orders(None),
                   orders.view_cart(None), orders.add_to_cart(None, "ACC-001"),
                   orders.place_order(None), orders.cancel_order(None, "1001")]:
        assert result == orders.LOGIN_REQUIRED


def test_list_my_orders_shows_only_own_orders():
    ids = {o["id"] for o in orders.list_my_orders("C-001")["orders"]}
    assert {"1234", "1001"} <= ids
    assert "5678" not in ids


def test_view_cart_total():
    cart = orders.view_cart("C-001")  # seed: ACC-001 (160 000) + HDP-003 (620 000)
    assert cart["total_uzs"] == 780_000


# ======================= orders: write =======================

def test_add_to_cart_checks_stock():
    assert "error" in orders.add_to_cart("C-002", "LAP-003")        # stock 0
    assert "error" in orders.add_to_cart("C-002", "LAP-006", 99)    # stock 3
    assert "error" in orders.add_to_cart("C-002", "LAP-006", 0)     # bad quantity
    assert "error" in orders.add_to_cart("C-002", "XXX-000")        # unknown product


def test_add_to_cart_adds_up_quantities():
    orders.add_to_cart("C-002", "LAP-006", 2)
    cart = orders.add_to_cart("C-002", "LAP-006", 1)
    assert cart["items"][0]["quantity"] == 3
    assert "error" in orders.add_to_cart("C-002", "LAP-006", 1)     # would be 4 > stock 3


def test_place_order_creates_order_decreases_stock_and_clears_cart():
    stock_before = stock_of("HDP-003")
    result = orders.place_order("C-001")

    assert result["status"] == "yangi"
    assert result["total_uzs"] == 780_000
    assert stock_of("HDP-003") == stock_before - 1
    assert orders.view_cart("C-001")["items"] == []
    assert orders.get_order_status("C-001", result["order_id"])["status"] == "yangi"


def test_place_order_with_empty_cart():
    assert "error" in orders.place_order("C-002")


def test_cancel_new_order_restores_stock():
    stock_before = stock_of("ACC-002")  # order 1001 (C-001, yangi) contains 1 x ACC-002
    result = orders.cancel_order("C-001", "1001")
    assert result["status"] == "bekor_qilindi"
    assert stock_of("ACC-002") == stock_before + 1


def test_cannot_cancel_delivered_order():
    result = orders.cancel_order("C-001", "1234")  # yetkazildi
    assert "error" in result
    assert orders.get_order_status("C-001", "1234")["status"] == "yetkazildi"


def test_cannot_cancel_someone_elses_order():
    result = orders.cancel_order("C-001", "5678")  # belongs to C-002
    assert result == orders.ORDER_NOT_FOUND
    assert orders.get_order_status("C-002", "5678")["status"] == "yolda"  # unchanged


# ======================= @tool wrappers =======================

ORDER_TOOLS = tools.ORDER_READ_TOOLS + tools.CART_TOOLS + tools.CONFIRM_REQUIRED_TOOLS


@pytest.mark.parametrize("t", ORDER_TOOLS, ids=lambda t: t.name)
def test_customer_id_is_hidden_from_llm_schema(t):
    # tool_call_schema is exactly what the LLM is shown.
    visible_args = t.tool_call_schema.model_json_schema().get("properties", {})
    assert "customer_id" not in visible_args
    assert "config" not in visible_args


def test_tool_reads_customer_from_config():
    as_c001 = {"configurable": {"customer_id": "C-001"}}
    assert tools.get_order_status.invoke({"order_id": "1234"}, config=as_c001)["status"] == "yetkazildi"
    assert "error" in tools.get_order_status.invoke({"order_id": "5678"}, config=as_c001)


def test_llm_cannot_sneak_in_another_customer_id():
    # Even if the model invents a customer_id argument, it is ignored: only config counts.
    tool_call = {"type": "tool_call", "id": "call-1", "name": "get_order_status",
                 "args": {"order_id": "5678", "customer_id": "C-002"}}
    message = tools.get_order_status.invoke(
        tool_call, config={"configurable": {"customer_id": "C-001"}})
    assert "yolda" not in message.content


def test_tool_without_login_asks_to_log_in():
    assert tools.list_my_orders.invoke({}) == orders.LOGIN_REQUIRED


def test_catalog_tool_schema_matches_spec():
    args = tools.search_products.tool_call_schema.model_json_schema()["properties"]
    assert set(args) == {"query", "category", "min_price", "max_price", "brand",
                         "in_stock_only", "limit"}
