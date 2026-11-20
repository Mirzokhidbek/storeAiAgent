"""Streamlit chat UI for the TexnoBozor assistant (SPEC §10).

Run:  streamlit run app.py

How Streamlit works (important for reading this file): every click or message re-runs
this WHOLE script from top to bottom. Normal variables are lost on each re-run, so
everything that must survive (who is logged in, the chat, the conversation thread)
lives in st.session_state.

Confirmation (Phase 6a): the graph pauses itself with interrupt(). Whether a
confirmation is waiting is read from the graph's checkpointer on every re-run, not
kept in session_state. The thread_id is also in the URL (?t=...), so a page reload
finds the same paused graph again.
"""

import uuid

import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.exceptions import (ModelAuthenticationError, ModelConnectionError,
                                       ModelError, ModelPermissionDeniedError,
                                       ModelRateLimitError, ModelTimeoutError)
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import build_graph, pending_confirmation, run_config
from agent.nodes import APPROVE, REJECT, format_uzs
from config import MissingSettingError
from shop import customers, orders

GREETING = "Salom! Men TexnoBozor yordamchisiman. Sizga qanday yordam bera olaman?"
STATUS_LABELS = {"yangi": "🆕 Yangi", "tolandi": "💳 To'landi", "yolda": "🚚 Yo'lda",
                 "yetkazildi": "✅ Yetkazildi", "bekor_qilindi": "❌ Bekor qilindi"}
SIDEBAR_ORDERS = 10


# ======================= agent =======================

@st.cache_resource
def get_agent():
    """Built once per server, not on every re-run. Its MemorySaver keeps every
    conversation; thread_id keeps different browser sessions apart."""
    return build_graph()


def friendly_error(err: Exception) -> str:
    """Turn API errors into a clear Uzbek message (the Gemini free tier has limits)."""
    if isinstance(err, ModelRateLimitError):
        return ("Hozir so'rovlar juda ko'p bo'lib ketdi (Gemini limiti). "
                "Iltimos, bir daqiqadan keyin qayta urinib ko'ring.")
    # Google answers a wrong key with 400 API_KEY_INVALID, not 401, so check the reason too.
    if (isinstance(err, (MissingSettingError, ModelAuthenticationError,
                         ModelPermissionDeniedError)) or "API_KEY_INVALID" in str(err)):
        return "Gemini API kaliti noto'g'ri yoki ko'rsatilmagan. Administrator .env faylini tekshirsin."
    if isinstance(err, (ModelConnectionError, ModelTimeoutError)):
        return "Gemini bilan bog'lanib bo'lmadi. Internet aloqasini tekshirib, qayta urinib ko'ring."
    if isinstance(err, ModelError):
        return "Gemini xizmatida vaqtinchalik xatolik yuz berdi. Birozdan keyin qayta urinib ko'ring."
    return "Kechirasiz, kutilmagan xatolik yuz berdi. Iltimos, qayta urinib ko'ring."


def split_sources(text: str, sources: list[str]) -> str:
    """respond adds "Manba: ..." to the text; the UI shows it as a caption instead."""
    if sources and "\n\nManba:" in text:
        return text.rsplit("\n\nManba:", 1)[0]
    return text


def log_turn(state: dict, usage: dict) -> None:
    """Console log per request (SPEC §12): intent, tools, tokens. Never the API key."""
    tools = [c["tool"] for c in state.get("context") or []]
    tokens = sum(u.get("total_tokens", 0) for u in usage.values())
    print(f"[app] intent={state.get('intent')} tools={tools} tokens={tokens}", flush=True)


def thread_config() -> dict:
    return run_config(st.session_state.thread_id)


def run_graph(payload) -> None:
    """Run the graph: a new question (dict) or a resume after a click (Command)."""
    try:
        with st.spinner("O'ylayapman..."), get_usage_metadata_callback() as usage:
            state = get_agent().invoke(payload, thread_config())
    except Exception as err:   # noqa: BLE001 — any failure becomes a friendly message
        print(f"[app] error: {type(err).__name__}: {str(err)[:200]}", flush=True)
        add_message("assistant", friendly_error(err))
        return

    log_turn(state, usage.usage_metadata)
    if pending_confirmation(get_agent(), thread_config()):
        return   # paused by interrupt(): main() draws the buttons from the checkpointer
    sources = state.get("sources") or []
    add_message("assistant", split_sources(state["messages"][-1].content, sources), sources)


def ask(question: str) -> None:
    # customer_id comes from the login, never from the LLM.
    run_graph({"messages": [HumanMessage(question)],
               "customer_id": st.session_state.customer_id})


def resume(decision: dict) -> None:
    """Continue the paused graph with the customer's answer, at most once.

    The click is accepted only if THAT interrupt is still open. A double click, a second
    tab or a stale button after a reload finds no matching interrupt and does nothing.
    """
    pending = pending_confirmation(get_agent(), thread_config())
    if pending is None or pending.id != decision["id"]:
        return
    add_message("assistant", pending.value["summary"])   # keep what was answered in the chat
    add_message("user", "✅ Tasdiqlash" if decision["answer"] == APPROVE else "❌ Bekor qilish")
    # {interrupt id: answer}: the answer goes to exactly the interrupt the customer saw.
    run_graph(Command(resume={pending.id: decision["answer"]}))


# ======================= session state =======================

def greeting() -> list[dict]:
    return [{"role": "assistant", "content": GREETING, "sources": []}]


def init_session() -> None:
    """First run of a browser session: restore the thread from the URL, or start a new one."""
    ss = st.session_state
    if "thread_id" in ss:
        return
    ss.customer_id, ss.customer_name, ss.decision = None, None, None
    ss.messages = greeting()
    saved = st.query_params.get("t")
    if not (saved and restore_thread(saved)):
        start_thread()


def start_thread() -> None:
    st.session_state.thread_id = uuid.uuid4().hex
    st.query_params["t"] = st.session_state.thread_id   # survives a page reload


def restore_thread(thread_id: str) -> bool:
    """After a reload: who was logged in and the chat come back from the checkpointer."""
    values = get_agent().get_state(run_config(thread_id)).values
    if not values:
        return False      # unknown or deleted thread (e.g. after logout)
    ss = st.session_state
    ss.thread_id = thread_id
    customer = customers.find_by_id(values["customer_id"]) if values.get("customer_id") else None
    if customer:
        ss.customer_id, ss.customer_name = customer["id"], customer["name"]
    ss.messages += [to_display(m) for m in values.get("messages", [])]
    return True


def to_display(message) -> dict:
    if message.type == "human":
        return {"role": "user", "content": message.content, "sources": []}
    text, _, sources = message.content.partition("\n\nManba: ")
    return {"role": "assistant", "content": text, "sources": sources.split(", ") if sources else []}


def reset_conversation() -> None:
    """Called on login and logout. The next customer must not inherit anything:
    a new thread_id gives the LLM an empty memory, and the old thread is deleted
    from the checkpointer (with any paused confirmation in it)."""
    ss = st.session_state
    get_agent().checkpointer.delete_thread(ss.thread_id)
    start_thread()
    ss.messages = greeting()
    ss.decision = None


def add_message(role: str, content: str, sources: list[str] | None = None) -> None:
    st.session_state.messages.append({"role": role, "content": content, "sources": sources or []})


# ======================= sidebar =======================

def login_form() -> None:
    with st.form("login"):
        phone = st.text_input("Telefon raqam", placeholder="+998901234567")
        if st.form_submit_button("Kirish"):
            customer = customers.find_by_phone(phone)
            if customer is None:
                st.error("Bu raqam bilan mijoz topilmadi.")
                return
            st.session_state.customer_id = customer["id"]
            st.session_state.customer_name = customer["name"]
            reset_conversation()
            st.rerun()
    st.caption("Sinov uchun: +998901234567 (Aziz Karimov)")


def logout() -> None:
    st.session_state.customer_id = None
    st.session_state.customer_name = None
    reset_conversation()


def show_cart(customer_id: str) -> None:
    st.subheader("🛒 Savat")
    cart = orders.view_cart(customer_id)   # read from the DB on every re-run: always fresh
    if not cart.get("items"):
        st.caption("Savat bo'sh")
        return
    for item in cart["items"]:
        st.markdown(f"- {item['name']} — {item['quantity']} ta × {format_uzs(item['price_uzs'])}")
    st.markdown(f"**Jami: {format_uzs(cart['total_uzs'])}**")


def show_orders(customer_id: str) -> None:
    st.subheader("📦 Buyurtmalarim")
    my_orders = orders.list_my_orders(customer_id).get("orders", [])
    if not my_orders:
        st.caption("Buyurtmalar yo'q")
        return
    for order in my_orders[:SIDEBAR_ORDERS]:
        st.markdown(f"- #{order['id']} — {STATUS_LABELS.get(order['status'], order['status'])}")


def sidebar() -> None:
    with st.sidebar:
        st.title("TexnoBozor")
        if not st.session_state.customer_id:
            st.info("Buyurtmalar va savat uchun tizimga kiring.")
            login_form()
            return
        st.success(f"👤 {st.session_state.customer_name}")
        st.button("Chiqish", on_click=logout)
        show_cart(st.session_state.customer_id)
        show_orders(st.session_state.customer_id)


# ======================= chat =======================

def show_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(as_markdown(message["content"]))
            if message["sources"]:
                st.caption("Manba: " + ", ".join(message["sources"]))


def decide(interrupt_id: str, answer: str) -> None:
    """Button callback. Runs BEFORE the re-run, so the choice is ready when main() starts."""
    st.session_state.decision = {"id": interrupt_id, "answer": answer}


def as_markdown(text: str) -> str:
    return text.replace("\n", "  \n")   # two spaces = line break in Markdown


def confirmation_box(pending) -> None:
    with st.chat_message("assistant"):
        st.markdown(as_markdown(pending.value["summary"]))
        left, right = st.columns(2)
        # The interrupt id travels with the click, so it answers this question only.
        left.button("✅ Tasdiqlash", on_click=decide, args=(pending.id, APPROVE), type="primary")
        right.button("❌ Bekor qilish", on_click=decide, args=(pending.id, REJECT))


def main() -> None:
    st.set_page_config(page_title="TexnoBozor yordamchisi", page_icon="🛍️")
    init_session()
    sidebar()
    show_history()

    ss = st.session_state
    if ss.decision:
        decision, ss.decision = ss.decision, None   # use a click once
        resume(decision)
        st.rerun()

    # Source of truth: the checkpointer. A reload or re-run sees the same pause.
    pending = pending_confirmation(get_agent(), thread_config())
    if pending:
        confirmation_box(pending)

    # While a confirmation is open, the customer answers with the buttons, not by typing.
    question = st.chat_input("Savolingizni yozing...", disabled=pending is not None)
    if question:
        add_message("user", question)
        with st.chat_message("user"):   # show it now, not only after the answer arrives
            st.markdown(question)
        ask(question)
        st.rerun()             # redraw: new messages, fresh cart and orders in the sidebar


if __name__ == "__main__":   # Streamlit runs the file as __main__; tests import it safely
    main()
