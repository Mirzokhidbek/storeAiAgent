"""Phase 4d tests: confirm_action and execute_action (place / cancel with confirmation).

Run:  python -m pytest -v
No API key needed: the LLMs are fakes, the database is a fresh temp copy (conftest.py).
The confirmation is a fake `confirm` function that answers "y" or "n" and remembers
the summary it was shown.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.graph import build_graph, run_config
from agent.prompts import REJECTED_MESSAGES
from shop import orders
from shop.db import connect
from tests.test_graph import FakeLLM, tool_call

TABLES = ["products", "customers", "orders", "order_items", "cart_items"]


class FakeConfirm:
    """Pretends to be the customer typing y/n in the terminal."""

    def __init__(self, answer: bool):
        self.answer, self.shown = answer, []

    def __call__(self, summary: str) -> bool:
        self.shown.append(summary)
        return self.answer


@pytest.fixture
def fake(monkeypatch):
    llms = {"router": FakeLLM(), "action": FakeLLM(), "tool": FakeLLM(), "respond": FakeLLM()}
    monkeypatch.setattr(nodes, "_router_llm", lambda: llms["router"])
    monkeypatch.setattr(nodes, "_action_llm", lambda: llms["action"])
    monkeypatch.setattr(nodes, "_tool_llm", lambda group: llms["tool"])
    monkeypatch.setattr(nodes, "_respond_llm", lambda: llms["respond"])
    return llms


def db_snapshot() -> dict:
    """Every row of every table: if this is equal before and after, nothing changed."""
    with connect() as conn:
        return {t: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}")) for t in TABLES}


def ask_action(fake, request: nodes.ActionRequest, confirm=None, customer_id="C-001",
               question="Savatimdagini buyurtma qil") -> dict:
    fake["router"].replies = [nodes.RouteDecision(intent="action")]
    fake["action"].replies = [request]
    fake["respond"].replies = [AIMessage("javob")]
    return build_graph().invoke(
        {"messages": [HumanMessage(question)], "customer_id": customer_id},
        run_config("actions", confirm=confirm))


PLACE = nodes.ActionRequest(action="place_order")


def cancel(order_id: str | None) -> nodes.ActionRequest:
    return nodes.ActionRequest(action="cancel_order", order_id=order_id)


# ======================= place_order =======================

def test_no_order_without_confirmation(fake):
    """No way to ask the customer (no confirm function) = no order. Safe default."""
    before = db_snapshot()

    state = ask_action(fake, PLACE, confirm=None)

    assert db_snapshot() == before
    assert state["messages"][-1].content == REJECTED_MESSAGES["place_order"]


def test_rejected_order_leaves_db_unchanged(fake):
    before = db_snapshot()
    confirm = FakeConfirm(answer=False)

    state = ask_action(fake, PLACE, confirm=confirm)

    assert len(confirm.shown) == 1, "the customer must be asked exactly once"
    assert db_snapshot() == before
    assert state["messages"][-1].content == REJECTED_MESSAGES["place_order"]
    assert fake["respond"].calls == [], "fixed text after 'n', no LLM"


def test_approved_order_is_created_from_cart(fake):
    # Control case: proves the tests above fail for the right reason (the "n").
    cart = orders.view_cart("C-001")
    orders_before = len(orders.list_my_orders("C-001")["orders"])

    state = ask_action(fake, PLACE, confirm=FakeConfirm(answer=True))

    result = state["context"][0]["result"]
    assert result["status"] == "yangi"
    assert result["total_uzs"] == cart["total_uzs"]
    assert len(orders.list_my_orders("C-001")["orders"]) == orders_before + 1
    assert orders.view_cart("C-001")["items"] == []
    assert state["pending_action"] is None


def test_summary_shows_items_prices_total_and_delivery(fake):
    confirm = FakeConfirm(answer=False)
    summary = orders.order_summary("C-001")

    ask_action(fake, PLACE, confirm=confirm)

    text = confirm.shown[0]
    for item in summary["items"]:
        assert item["name"] in text
        assert nodes.format_uzs(item["price_uzs"]) in text
    assert nodes.format_uzs(summary["total_tashkent_uzs"]) in text
    assert "Yetkazib berish" in text and "50 000 so'm" in text


def test_delivery_fees_follow_policy():
    # yetkazib_berish.md: Tashkent free from 1 000 000, else 30 000; regions 50 000.
    assert orders.delivery_fees(999_999) == {"tashkent_uzs": 30_000, "region_uzs": 50_000}
    assert orders.delivery_fees(1_000_000) == {"tashkent_uzs": 0, "region_uzs": 50_000}


def test_empty_cart_cannot_be_ordered(fake):
    with connect() as conn:
        conn.execute("DELETE FROM cart_items WHERE customer_id = 'C-001'")
    before = db_snapshot()
    confirm = FakeConfirm(answer=True)   # even a "y" must not help

    state = ask_action(fake, PLACE, confirm=confirm)

    assert confirm.shown == [], "nothing to confirm: do not even ask"
    assert "error" in state["context"][0]["result"]
    assert db_snapshot() == before


def test_cart_changed_after_summary_is_not_ordered(fake):
    """The customer approved one summary; a different cart must not be ordered."""
    def confirm_then_change_cart(summary):
        orders.add_to_cart("C-001", "ACC-001", 1)   # cart changes while "y" is being typed
        return True
    before_orders = len(orders.list_my_orders("C-001")["orders"])

    state = ask_action(fake, PLACE, confirm=confirm_then_change_cart)

    assert "error" in state["context"][0]["result"]
    assert len(orders.list_my_orders("C-001")["orders"]) == before_orders


# ======================= cancel_order =======================

def test_cannot_cancel_other_customers_order(fake):
    before = db_snapshot()
    confirm = FakeConfirm(answer=True)

    state = ask_action(fake, cancel("5678"), confirm=confirm,   # 5678 belongs to C-002
                       question="5678 buyurtmani bekor qil")

    assert confirm.shown == [], "a foreign order must not even reach the y/n question"
    assert "error" in state["context"][0]["result"]
    assert db_snapshot() == before
    assert "yolda" not in fake["respond"].calls[0][0].content   # nothing about it leaks


def test_rejected_cancel_leaves_order_unchanged(fake):
    before = db_snapshot()
    confirm = FakeConfirm(answer=False)

    state = ask_action(fake, cancel("1001"), confirm=confirm, question="1001 ni bekor qil")

    assert "1001" in confirm.shown[0]
    assert db_snapshot() == before
    assert state["messages"][-1].content == REJECTED_MESSAGES["cancel_order"]


def test_approved_cancel_changes_status(fake):
    state = ask_action(fake, cancel("1001"), confirm=FakeConfirm(answer=True),
                       question="1001 ni bekor qil")

    assert state["context"][0]["result"]["status"] == "bekor_qilindi"
    assert orders.get_order_status("C-001", "1001")["status"] == "bekor_qilindi"


def test_cancel_without_order_number_asks_for_it(fake):
    confirm = FakeConfirm(answer=True)

    state = ask_action(fake, cancel(None), confirm=confirm, question="buyurtmamni bekor qil")

    assert confirm.shown == []
    assert "raqam" in state["context"][0]["result"]["error"]


# ======================= add_to_cart (no confirmation needed) =======================

def test_add_to_cart_runs_without_confirmation(fake):
    fake["tool"].replies = [tool_call("add_to_cart", product_id="LAP-001", quantity=1),
                            AIMessage("TAYYOR")]
    confirm = FakeConfirm(answer=False)

    state = ask_action(fake, nodes.ActionRequest(action="add_to_cart"), confirm=confirm,
                       question="LAP-001 ni savatga qo'sh")

    assert confirm.shown == []
    ids = [i["product_id"] for i in orders.view_cart("C-001")["items"]]
    assert "LAP-001" in ids
    assert state["context"][0]["tool"] == "add_to_cart"


def test_llm_cannot_reach_place_order_from_cart_tools():
    """The add_to_cart tool loop must not even see the confirm-required tools."""
    names = {t.name for t in nodes.TOOL_GROUPS["cart"]}
    assert not names & {"place_order", "cancel_order"}
