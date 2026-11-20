"""UI tests: helpers, customer switching and the interrupt() buttons, without a browser.

Run:  python -m pytest -v
Streamlit's AppTest runs the real app.py and clicks/types like a user. The graph is
the real one; only the LLMs are fakes (as in test_actions.py).
"""

import pytest
import streamlit as st
from langchain_core.exceptions import ModelInvalidRequestError, ModelRateLimitError
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

import agent.graph
import app
from agent import nodes
from agent.graph import build_graph, pending_confirmation, run_config
from agent.nodes import APPROVE
from agent.prompts import REJECTED_MESSAGES
from config import MissingSettingError
from shop import customers, orders
from tests.test_actions import db_snapshot, fake  # noqa: F401 — `fake` is a fixture
from tests.test_graph import FakeLLM


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


# ======================= switching customers =======================
# Runs the real app.py with Streamlit's AppTest (no browser): it clicks and types
# like a user. Only the LLMs are fakes.

SECRET = "Mening manzilim Chilonzor 5-uy"


@pytest.fixture
def ui(fake, monkeypatch):
    chat_llm = FakeLLM()
    monkeypatch.setattr(nodes, "_chat_llm", lambda: chat_llm)
    # AppTest runs app.py as its own module, so app.get_agent() here would be a different
    # cache. Give both sides ONE graph, so the test can look into the app's memory.
    graph = build_graph()
    monkeypatch.setattr(agent.graph, "build_graph", lambda: graph)
    st.cache_resource.clear()        # forget graphs cached by earlier tests
    return open_app(), chat_llm, graph


def open_app(thread_id: str | None = None) -> AppTest:
    """A new browser tab. With thread_id it is a reload of a page with ?t=<thread_id>."""
    at = AppTest.from_file(app.__file__, default_timeout=10)
    if thread_id:
        at.query_params["t"] = thread_id
    return at.run()


def login(at, phone: str) -> None:
    at.sidebar.text_input[0].input(phone)
    at.sidebar.button[0].click().run()     # the form's "Kirish" button


def say(at, fake, chat_llm, text: str) -> None:
    fake["router"].replies.append(nodes.RouteDecision(intent="chitchat"))
    chat_llm.replies.append(AIMessage("Tushunarli!"))
    at.chat_input[0].set_value(text).run()


def thread_texts(graph, thread_id: str) -> str:
    state = graph.get_state({"configurable": {"thread_id": thread_id}})
    return " ".join(m.content for m in state.values.get("messages", []))


def test_next_customer_cannot_see_previous_chat(ui, fake):
    at, chat_llm, graph = ui
    login(at, "+998901234567")                        # C-001
    say(at, fake, chat_llm, SECRET)
    c001_thread = at.session_state["thread_id"]
    assert SECRET in thread_texts(graph, c001_thread)        # control: it really was remembered

    at.sidebar.button[0].click().run()                # "Chiqish"
    login(at, "+998912345678")                        # C-002
    say(at, fake, chat_llm, "Mening manzilim qayerda edi?")

    assert at.session_state["customer_id"] == "C-002"
    assert at.session_state["thread_id"] != c001_thread
    # 1) the graph memory of the new thread has nothing from C-001
    assert SECRET not in thread_texts(graph, at.session_state["thread_id"])
    # 2) the LLM answering C-002 was not shown C-001's message
    assert SECRET not in " ".join(m.content for m in chat_llm.calls[-1])
    # 3) the old thread is deleted from the server's memory, not just abandoned
    assert thread_texts(graph, c001_thread) == ""
    # 4) the screen shows no old messages either
    assert all(SECRET not in m["content"] for m in at.session_state["messages"])


def test_logout_clears_chat_and_memory(ui, fake):
    at, chat_llm, graph = ui
    login(at, "+998901234567")
    say(at, fake, chat_llm, SECRET)
    old_thread = at.session_state["thread_id"]

    at.sidebar.button[0].click().run()                # "Chiqish"

    assert at.session_state["customer_id"] is None
    assert at.session_state["thread_id"] != old_thread
    assert thread_texts(graph, old_thread) == ""
    assert [m["content"] for m in at.session_state["messages"]] == [app.GREETING]


# ======================= interrupt() buttons =======================

def order_until_buttons(at, fake) -> None:
    fake["router"].replies.append(nodes.RouteDecision(intent="action"))
    fake["action"].replies.append(nodes.ActionRequest(action="place_order"))
    at.chat_input[0].set_value("Savatimdagini buyurtma qil").run()


def button(at, label: str):
    matches = [b for b in at.main.button if b.label == label]
    return matches[0] if matches else None


def click(at, fake, label: str) -> None:
    fake["respond"].replies.append(AIMessage("Buyurtmangiz qabul qilindi."))
    button(at, label).click().run()


def order_count() -> int:
    return len(orders.list_my_orders("C-001")["orders"])


def paused(graph, at) -> bool:
    return pending_confirmation(graph, run_config(at.session_state["thread_id"])) is not None


def test_buttons_appear_and_db_is_unchanged_while_paused(ui, fake):
    at, _, graph = ui
    login(at, "+998901234567")
    before = db_snapshot()

    order_until_buttons(at, fake)

    assert button(at, "✅ Tasdiqlash") and button(at, "❌ Bekor qilish")
    assert paused(graph, at)
    assert at.chat_input[0].disabled, "answer with the buttons, not by typing"
    assert db_snapshot() == before


def test_approve_button_creates_order(ui, fake):
    at, _, graph = ui
    login(at, "+998901234567")
    count = order_count()
    order_until_buttons(at, fake)

    click(at, fake, "✅ Tasdiqlash")

    assert order_count() == count + 1
    assert orders.view_cart("C-001")["items"] == []
    assert not paused(graph, at) and button(at, "✅ Tasdiqlash") is None


def test_reject_button_leaves_db_unchanged(ui, fake):
    at, _, graph = ui
    login(at, "+998901234567")
    order_until_buttons(at, fake)
    before = db_snapshot()

    click(at, fake, "❌ Bekor qilish")

    assert db_snapshot() == before
    assert at.session_state["messages"][-1]["content"] == REJECTED_MESSAGES["place_order"]
    assert not paused(graph, at)


def test_reload_keeps_pending_confirmation(ui, fake):
    at, _, graph = ui
    login(at, "+998901234567")
    order_until_buttons(at, fake)
    count = order_count()

    reloaded = open_app(at.session_state["thread_id"])     # F5 in the browser

    assert reloaded.session_state["customer_id"] == "C-001", "still logged in"
    assert button(reloaded, "✅ Tasdiqlash"), "the question did not disappear"
    assert order_count() == count, "a reload must not answer the question"

    click(reloaded, fake, "✅ Tasdiqlash")
    assert order_count() == count + 1

    again = open_app(at.session_state["thread_id"])        # reload once more
    assert button(again, "✅ Tasdiqlash") is None, "answered questions do not come back"
    assert order_count() == count + 1


def test_stale_button_after_reload_does_not_order_twice(ui, fake):
    """The old tab still shows the buttons after the new tab already approved."""
    at, _, graph = ui
    login(at, "+998901234567")
    order_until_buttons(at, fake)
    count = order_count()
    click(open_app(at.session_state["thread_id"]), fake, "✅ Tasdiqlash")   # new tab

    button(at, "✅ Tasdiqlash").click().run()                            # old tab

    assert order_count() == count + 1, "exactly one order"


def test_logout_drops_pending_confirmation(ui, fake):
    at, _, graph = ui
    login(at, "+998901234567")
    order_until_buttons(at, fake)
    old_thread = at.session_state["thread_id"]
    before = db_snapshot()

    at.sidebar.button[0].click().run()                    # "Chiqish"

    reloaded = open_app(old_thread)                       # someone opens the old URL
    assert reloaded.session_state["customer_id"] is None
    assert button(reloaded, "✅ Tasdiqlash") is None
    assert db_snapshot() == before


# ======================= no double orders =======================
# A real double click sends the same button event twice: the second one arrives with the
# SAME interrupt id after the first already resumed the graph. decide() turns a click
# into session_state["decision"]; we replay exactly that to simulate the second click.

def replay_click(at, interrupt_id: str) -> None:
    at.session_state["decision"] = {"id": interrupt_id, "answer": APPROVE}
    at.run()


def pending_id(graph, at) -> str:
    return pending_confirmation(graph, run_config(at.session_state["thread_id"])).id


def test_double_click_on_approve_orders_once(ui, fake):
    at, _, graph = ui
    login(at, "+998901234567")
    order_until_buttons(at, fake)
    first_click = pending_id(graph, at)
    count = order_count()

    click(at, fake, "✅ Tasdiqlash")
    replay_click(at, first_click)            # the second click of the double click

    assert order_count() == count + 1, "exactly one order"


def test_reload_after_approve_does_not_order_again(ui, fake):
    at, _, graph = ui
    login(at, "+998901234567")
    order_until_buttons(at, fake)
    count = order_count()
    click(at, fake, "✅ Tasdiqlash")

    reloaded = open_app(at.session_state["thread_id"])     # F5 after approving
    reloaded.run()                                          # and one more re-run

    assert order_count() == count + 1
    assert button(reloaded, "✅ Tasdiqlash") is None
    assert "✅ Buyurtmangiz qabul qilindi" in reloaded.session_state["messages"][-1]["content"]


def test_old_click_cannot_approve_a_newer_question(ui, fake):
    """Order 1 is approved; order 2 now waits. A late click from order 1 must not
    approve order 2 — that is what the interrupt id check in app.resume() is for."""
    at, _, graph = ui
    login(at, "+998901234567")
    order_until_buttons(at, fake)
    old_click = pending_id(graph, at)
    click(at, fake, "✅ Tasdiqlash")
    orders.add_to_cart("C-001", "ACC-001", 1)
    order_until_buttons(at, fake)             # a new question with a new interrupt id
    count = order_count()

    replay_click(at, old_click)

    assert order_count() == count, "the old click did nothing"
    assert paused(graph, at) and button(at, "✅ Tasdiqlash"), "question 2 still waits"
