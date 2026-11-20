"""Phase 5 tests: the UI's helpers and its button confirmation, without a browser.

Run:  python -m pytest -v
The graph is the real one with fake LLMs (as in test_actions.py). We call the same
make_confirm function app.py uses, so we test the pause -> click -> resume flow.
"""

import pytest
from langchain_core.exceptions import ModelInvalidRequestError, ModelRateLimitError
from langchain_core.messages import AIMessage, HumanMessage

import app
from agent import nodes
from agent.graph import build_graph, run_config
from agent.prompts import REJECTED_MESSAGES
from config import MissingSettingError
from shop import customers, orders
from tests.test_actions import db_snapshot, fake  # noqa: F401 — `fake` is a fixture


# ======================= login =======================

@pytest.mark.parametrize("typed", ["+998901234567", "998901234567", "90 123 45 67",
                                   "+998 (90) 123-45-67"])
def test_login_accepts_common_phone_formats(typed):
    assert customers.find_by_phone(typed)["id"] == "C-001"


def test_unknown_phone_is_not_logged_in():
    assert customers.find_by_phone("+998000000000") is None


# ======================= errors and sources =======================

def test_rate_limit_gets_uzbek_message():
    text = app.friendly_error(ModelRateLimitError("429 RESOURCE_EXHAUSTED"))
    assert "limit" in text and "qayta urinib" in text


def test_missing_api_key_gets_uzbek_message():
    assert "API kaliti" in app.friendly_error(MissingSettingError("GOOGLE_API_KEY"))


def test_invalid_api_key_gets_uzbek_message():
    # Real Gemini answer to a wrong key: HTTP 400 with reason API_KEY_INVALID.
    err = ModelInvalidRequestError("400 INVALID_ARGUMENT ... 'reason': 'API_KEY_INVALID'")
    assert "API kaliti" in app.friendly_error(err)


def test_unknown_error_does_not_leak_details():
    text = app.friendly_error(ValueError("secret internal detail"))
    assert "secret" not in text


def test_source_line_moves_to_caption():
    text = "14 kun ichida qaytarish mumkin.\n\nManba: qaytarish.md"
    assert app.split_sources(text, ["qaytarish.md"]) == "14 kun ichida qaytarish mumkin."


# ======================= button confirmation =======================

def start_order(fake, agent, thread: str):
    """First run: the graph must pause at confirm_action and hand back the summary."""
    fake["router"].replies = [nodes.RouteDecision(intent="action")]
    fake["action"].replies = [nodes.ActionRequest(action="place_order")]
    with pytest.raises(app.ConfirmationNeeded) as paused:
        agent.invoke({"messages": [HumanMessage("Savatimdagini buyurtma qil")],
                      "customer_id": "C-001"},
                     run_config(thread, confirm=app.make_confirm(None, None)))
    return paused.value.summary


def click(fake, agent, thread: str, summary: str, approved: bool) -> dict:
    """Second run after a click: resume the same thread (input None)."""
    # confirm_action runs again on resume, so it asks the action LLM once more.
    fake["action"].replies = [nodes.ActionRequest(action="place_order")]
    fake["respond"].replies = [AIMessage("Buyurtmangiz qabul qilindi.")]
    return agent.invoke(None, run_config(thread, confirm=app.make_confirm(approved, summary)))


def test_graph_pauses_and_nothing_is_ordered_before_click(fake):
    before = db_snapshot()

    summary = start_order(fake, build_graph(), "ui-1")

    assert "Jami" in summary
    assert db_snapshot() == before


def test_click_reject_leaves_db_unchanged(fake):
    agent = build_graph()
    summary = start_order(fake, agent, "ui-2")
    before = db_snapshot()

    state = click(fake, agent, "ui-2", summary, approved=False)

    assert db_snapshot() == before
    assert state["messages"][-1].content == REJECTED_MESSAGES["place_order"]


def test_click_approve_creates_order(fake):
    agent = build_graph()
    summary = start_order(fake, agent, "ui-3")
    count = len(orders.list_my_orders("C-001")["orders"])

    state = click(fake, agent, "ui-3", summary, approved=True)

    assert state["context"][0]["result"]["status"] == "yangi"
    assert len(orders.list_my_orders("C-001")["orders"]) == count + 1


def test_approval_of_an_old_summary_does_not_order_a_changed_cart(fake):
    """The customer clicked "Tasdiqlash" on summary A; the cart is now B -> ask again."""
    agent = build_graph()
    summary = start_order(fake, agent, "ui-4")
    orders.add_to_cart("C-001", "ACC-001", 1)
    count = len(orders.list_my_orders("C-001")["orders"])

    fake["action"].replies = [nodes.ActionRequest(action="place_order")]
    with pytest.raises(app.ConfirmationNeeded) as asked_again:
        agent.invoke(None, run_config("ui-4", confirm=app.make_confirm(True, summary)))

    assert asked_again.value.summary != summary
    assert len(orders.list_my_orders("C-001")["orders"]) == count
