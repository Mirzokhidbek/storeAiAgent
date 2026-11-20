"""Tests for prepare_action -> confirm_action (interrupt) -> execute_action.

Run:  python -m pytest -v
No API key needed: the LLMs are fakes, the database is a fresh temp copy (conftest.py).
The customer's answer is sent the way the app sends it: invoke(Command(resume=...)).
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from agent import nodes
from agent.graph import build_graph, pending_confirmation, run_config
from agent.nodes import APPROVE, REJECT
from agent.prompts import REJECTED_MESSAGES
from shop import orders
from shop.db import connect
from tests.test_graph import FakeLLM, tool_call

TABLES = ["products", "customers", "orders", "order_items", "cart_items"]


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


class Run:
    """One customer request through the real graph: pause, then (maybe) an answer."""

    def __init__(self, fake, request, customer_id="C-001", question="Savatimdagini buyurtma qil"):
        self.agent, self.config = build_graph(), run_config("actions")
        fake["router"].replies = [nodes.RouteDecision(intent="action")]
        fake["action"].replies = [request]
        fake["respond"].replies = [AIMessage("javob")]
        self.state = self.agent.invoke(
            {"messages": [HumanMessage(question)], "customer_id": customer_id}, self.config)

    @property
    def pending(self):
        return pending_confirmation(self.agent, self.config)

    def answer(self, value: str) -> dict:
        self.state = self.agent.invoke(Command(resume=value), self.config)
        return self.state


PLACE = nodes.ActionRequest(action="place_order")


def cancel(order_id: str | None) -> nodes.ActionRequest:
    return nodes.ActionRequest(action="cancel_order", order_id=order_id)


# ======================= place_order =======================

def test_interrupt_pauses_before_anything_changes(fake):
    before = db_snapshot()

    run = Run(fake, PLACE)

    assert run.pending is not None, "the graph must stop and wait"
    assert run.agent.get_state(run.config).next == ("confirm_action",)
    assert db_snapshot() == before
    assert "__interrupt__" in run.state


def test_rejected_order_leaves_db_unchanged(fake):
    run = Run(fake, PLACE)
    before = db_snapshot()

    state = run.answer(REJECT)

    assert db_snapshot() == before
    assert state["messages"][-1].content == REJECTED_MESSAGES["place_order"]
    assert fake["respond"].calls == [], "fixed text after 'reject', no LLM"
    assert run.pending is None


def test_unknown_answer_counts_as_reject(fake):
    run = Run(fake, PLACE)
    before = db_snapshot()

    run.answer("ha balki")   # only the exact word "approve" places an order

    assert db_snapshot() == before


def test_approved_order_is_created_from_cart(fake):
    cart = orders.view_cart("C-001")
    orders_before = len(orders.list_my_orders("C-001")["orders"])
    run = Run(fake, PLACE)

    state = run.answer(APPROVE)

    result = state["context"][0]["result"]
    assert result["status"] == "yangi"
    assert result["total_uzs"] == cart["total_uzs"]
    assert len(orders.list_my_orders("C-001")["orders"]) == orders_before + 1
    assert orders.view_cart("C-001")["items"] == []
    assert state["pending_action"] is None


def test_success_message_is_fixed_text_not_llm(fake):
    """Live bug: after "approve" the LLM asked "tasdiqlaysizmi?" again although the
    order already existed. The final text is now built by code from the DB result."""
    summary = orders.order_summary("C-001")
    run = Run(fake, PLACE)

    state = run.answer(APPROVE)

    text = state["messages"][-1].content
    assert f"#{state['context'][0]['result']['order_id']}" in text
    assert nodes.format_uzs(summary["total_tashkent_uzs"]) in text
    assert "tasdiq" not in text.lower()
    assert fake["respond"].calls == []


def test_action_llm_is_not_asked_again_on_resume(fake):
    """Resume re-runs the interrupted node from its first line. The LLM lives in
    prepare_action, so it must not run again (it could pick another action)."""
    run = Run(fake, PLACE)
    run.answer(APPROVE)

    assert len(fake["action"].calls) == 1


def test_second_resume_does_not_order_twice(fake):
    run = Run(fake, PLACE)
    run.answer(APPROVE)
    count = len(orders.list_my_orders("C-001")["orders"])

    run.answer(APPROVE)   # e.g. a double click that reaches the graph

    assert len(orders.list_my_orders("C-001")["orders"]) == count


def test_summary_shows_items_prices_total_and_delivery(fake):
    summary = orders.order_summary("C-001")

    text = Run(fake, PLACE).pending.value["summary"]

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

    run = Run(fake, PLACE)

    assert run.pending is None, "nothing to confirm: do not even ask"
    assert "error" in run.state["context"][0]["result"]
    assert db_snapshot() == before


def test_cart_changed_while_paused_is_not_ordered(fake):
    """The customer approved one summary; a different cart must not be ordered."""
    run = Run(fake, PLACE)
    orders.add_to_cart("C-001", "ACC-001", 1)   # cart changes while the graph waits
    before_orders = len(orders.list_my_orders("C-001")["orders"])

    state = run.answer(APPROVE)

    assert "error" in state["context"][0]["result"]
    assert len(orders.list_my_orders("C-001")["orders"]) == before_orders


# ======================= cancel_order =======================

def test_cannot_cancel_other_customers_order(fake):
    before = db_snapshot()

    run = Run(fake, cancel("5678"), question="5678 buyurtmani bekor qil")   # C-002's order

    assert run.pending is None, "a foreign order must not even reach the question"
    assert "error" in run.state["context"][0]["result"]
    assert db_snapshot() == before
    assert "yolda" not in fake["respond"].calls[0][0].content   # nothing about it leaks


def test_rejected_cancel_leaves_order_unchanged(fake):
    run = Run(fake, cancel("1001"), question="1001 ni bekor qil")
    assert "1001" in run.pending.value["summary"]
    before = db_snapshot()

    state = run.answer(REJECT)

    assert db_snapshot() == before
    assert state["messages"][-1].content == REJECTED_MESSAGES["cancel_order"]


def test_approved_cancel_changes_status(fake):
    run = Run(fake, cancel("1001"), question="1001 ni bekor qil")

    state = run.answer(APPROVE)

    assert state["context"][0]["result"]["status"] == "bekor_qilindi"
    assert orders.get_order_status("C-001", "1001")["status"] == "bekor_qilindi"
    assert state["messages"][-1].content.startswith("✅ #1001 buyurtma bekor qilindi")


def test_cancel_without_order_number_asks_for_it(fake):
    run = Run(fake, cancel(None), question="buyurtmamni bekor qil")

    assert run.pending is None
    assert "raqam" in run.state["context"][0]["result"]["error"]


# ======================= add_to_cart (no confirmation needed) =======================

def test_add_to_cart_runs_without_confirmation(fake):
    fake["tool"].replies = [tool_call("add_to_cart", product_id="LAP-001", quantity=1),
                            AIMessage("TAYYOR")]

    run = Run(fake, nodes.ActionRequest(action="add_to_cart"), question="LAP-001 ni savatga qo'sh")

    assert run.pending is None, "add_to_cart needs no confirmation"
    ids = [i["product_id"] for i in orders.view_cart("C-001")["items"]]
    assert "LAP-001" in ids
    assert run.state["context"][0]["tool"] == "add_to_cart"


def test_llm_cannot_reach_place_order_from_cart_tools():
    """The add_to_cart tool loop must not even see the confirm-required tools."""
    names = {t.name for t in nodes.TOOL_GROUPS["cart"]}
    assert not names & {"place_order", "cancel_order"}
