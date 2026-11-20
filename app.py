"""Phase 5: Streamlit chat UI for the TexnoBozor assistant (SPEC §10).

Run:  streamlit run app.py

How Streamlit works (important for reading this file): every click or message re-runs
this WHOLE script from top to bottom. Normal variables are lost on each re-run, so
everything that must survive (who is logged in, the chat, the conversation thread)
lives in st.session_state.
"""

import uuid

import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.exceptions import (ModelAuthenticationError, ModelConnectionError,
                                       ModelError, ModelPermissionDeniedError,
                                       ModelRateLimitError, ModelTimeoutError)
from langchain_core.messages import HumanMessage

from agent.graph import build_graph, run_config
from agent.nodes import format_uzs
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


class ConfirmationNeeded(Exception):
    """Raised inside confirm_action to pause the graph until a button is clicked.

    The graph's checkpointer has saved every step before confirm_action, so after the
    click we resume the same thread and confirm_action runs again, this time with an
    answer. Phase 6 replaces this hand-made pause with LangGraph's interrupt().
    """

    def __init__(self, summary: str):
        super().__init__(summary)
        self.summary = summary


def make_confirm(decision: bool | None, shown_summary: str | None):
    """The `confirm` function the graph calls before place_order / cancel_order."""
    def confirm(summary: str) -> bool:
        # A click answers only the exact summary the customer saw. If the cart changed
        # meanwhile, the summary differs and the customer is asked again.
        if decision is not None and summary == shown_summary:
            return decision
        raise ConfirmationNeeded(summary)
    return confirm


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


def run_agent(question: str | None) -> None:
    """Send a new question, or (question=None) resume a paused confirmation."""
    ss = st.session_state
    confirm = make_confirm(ss.decision, ss.pending_summary)
    config = run_config(ss.thread_id, confirm=confirm)
    # customer_id comes from the login, never from the LLM.
    payload = (None if question is None
               else {"messages": [HumanMessage(question)], "customer_id": ss.customer_id})
    ss.pending_summary, ss.decision = None, None

    try:
        with st.spinner("O'ylayapman..."), get_usage_metadata_callback() as usage:
            state = get_agent().invoke(payload, config)
    except ConfirmationNeeded as pause:
        ss.pending_summary = pause.summary   # show the buttons on the next re-run
        return
    except Exception as err:   # noqa: BLE001 — any failure becomes a friendly message
        print(f"[app] error: {type(err).__name__}: {str(err)[:200]}", flush=True)
        add_message("assistant", friendly_error(err))
        return

    log_turn(state, usage.usage_metadata)
    sources = state.get("sources") or []
    add_message("assistant", split_sources(state["messages"][-1].content, sources), sources)


# ======================= session state =======================

def init_session() -> None:
    """Runs on every re-run, but setdefault only fills keys that do not exist yet."""
    ss = st.session_state
    ss.setdefault("thread_id", uuid.uuid4().hex)
    ss.setdefault("customer_id", None)
    ss.setdefault("customer_name", None)
    ss.setdefault("messages", [{"role": "assistant", "content": GREETING, "sources": []}])
    ss.setdefault("pending_summary", None)   # order summary waiting for a button click
    ss.setdefault("decision", None)          # True / False after the click


def reset_conversation() -> None:
    """Called on login and logout. The next customer must not inherit anything:
    a new thread_id gives the LLM an empty memory, and the old thread is deleted
    from the checkpointer, so the previous chat is gone from the server too."""
    ss = st.session_state
    old_thread = ss.get("thread_id")
    if old_thread:
        get_agent().checkpointer.delete_thread(old_thread)
    ss.thread_id = uuid.uuid4().hex
    ss.messages = [{"role": "assistant", "content": GREETING, "sources": []}]
    ss.pending_summary, ss.decision = None, None


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
            st.markdown(message["content"])
            if message["sources"]:
                st.caption("Manba: " + ", ".join(message["sources"]))


def decide(approved: bool) -> None:
    """Button callback. Runs BEFORE the re-run, so the choice is ready when the script starts."""
    ss = st.session_state
    add_message("assistant", as_markdown(ss.pending_summary))   # keep what was approved in the chat
    add_message("user", "✅ Tasdiqlash" if approved else "❌ Bekor qilish")
    ss.decision = approved


def as_markdown(text: str) -> str:
    return text.replace("\n", "  \n")   # two spaces = line break in Markdown


def confirmation_box() -> None:
    with st.chat_message("assistant"):
        st.markdown(as_markdown(st.session_state.pending_summary))
        left, right = st.columns(2)
        left.button("✅ Tasdiqlash", on_click=decide, args=(True,), type="primary")
        right.button("❌ Bekor qilish", on_click=decide, args=(False,))


def main() -> None:
    st.set_page_config(page_title="TexnoBozor yordamchisi", page_icon="🛍️")
    init_session()
    sidebar()
    show_history()

    ss = st.session_state
    if ss.pending_summary and ss.decision is not None:
        run_agent(None)        # a button was clicked: resume the paused graph
        st.rerun()
    if ss.pending_summary:
        confirmation_box()

    # While a confirmation is open, the customer answers with the buttons, not by typing.
    question = st.chat_input("Savolingizni yozing...", disabled=bool(ss.pending_summary))
    if question:
        add_message("user", question)
        with st.chat_message("user"):   # show it now, not only after the answer arrives
            st.markdown(question)
        run_agent(question)
        st.rerun()             # redraw: new messages, fresh cart and orders in the sidebar


if __name__ == "__main__":   # Streamlit runs the file as __main__; tests import it safely
    main()
