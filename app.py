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

import html
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.exceptions import (ModelAuthenticationError, ModelConnectionError,
                                       ModelError, ModelPermissionDeniedError,
                                       ModelRateLimitError, ModelTimeoutError)
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent import tools
from agent.graph import build_graph, pending_confirmation, run_config
from agent.nodes import APPROVE, REJECT, format_uzs
from config import MissingSettingError
from shop import customers, orders

GREETING = "Salom! Men TexnoBozor yordamchisiman. Sizga qanday yordam bera olaman?"
STATUS_LABELS = {"yangi": "Yangi", "tolandi": "To'landi", "yolda": "Yo'lda",
                 "yetkazildi": "Yetkazildi", "bekor_qilindi": "Bekor qilindi"}
# Ready-made first questions shown on an empty chat (one click = one question).
SUGGESTIONS = ["6 mln so'mgacha yengil noutbuk", "Mahsulotni qaytarsam bo'ladimi?",
               "Yetkazib berish qancha turadi?", "1234 buyurtmam qayerda?"]
STYLE_FILE = Path(__file__).with_name("assets") / "style.css"
SIDEBAR_ORDERS = 10


# ======================= agent =======================

@st.cache_resource
def get_agent():
    """Built once per server, not on every re-run. Its MemorySaver keeps every
    conversation; thread_id keeps different browser sessions apart."""
    # Start the MCP servers now (once per process), not on the first customer question.
    # They run on their own background event loop, so Streamlit code stays sync.
    tools.backend()
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
    st.markdown('<div class="tb-muted">Savat va buyurtmalarni ko\'rish uchun kiring.</div>',
                unsafe_allow_html=True)
    with st.form("login", border=False):
        phone = st.text_input("Telefon raqam", placeholder="+998 90 123 45 67")
        if st.form_submit_button("Kirish", type="primary", use_container_width=True):
            customer = customers.find_by_phone(phone)
            if customer is None:
                st.error("Bu raqam bilan mijoz topilmadi.")
                return
            st.session_state.customer_id = customer["id"]
            st.session_state.customer_name = customer["name"]
            reset_conversation()
            st.rerun()
    st.caption("Sinov uchun: +998901234567")


def logout() -> None:
    st.session_state.customer_id = None
    st.session_state.customer_name = None
    reset_conversation()


def row(left: str, right: str, extra_class: str = "") -> str:
    """One "name ........ value" line. Text from the DB is escaped before it goes into HTML."""
    return (f'<div class="tb-row {extra_class}"><span>{html.escape(left)}</span>'
            f'<span>{html.escape(right)}</span></div>')


def show_cart(customer_id: str) -> None:
    st.markdown('<div class="tb-section">Savat</div>', unsafe_allow_html=True)
    cart = orders.view_cart(customer_id)   # read from the DB on every re-run: always fresh
    if not cart.get("items"):
        st.markdown('<div class="tb-muted">Savat bo\'sh</div>', unsafe_allow_html=True)
        return
    lines = [row(f"{item['name']} × {item['quantity']}", format_uzs(item["subtotal_uzs"]))
             for item in cart["items"]]
    lines.append(row("Jami", format_uzs(cart["total_uzs"]), "tb-total"))
    st.markdown("".join(lines), unsafe_allow_html=True)


def show_orders(customer_id: str) -> None:
    st.markdown('<div class="tb-section">Buyurtmalar</div>', unsafe_allow_html=True)
    my_orders = orders.list_my_orders(customer_id).get("orders", [])
    if not my_orders:
        st.markdown('<div class="tb-muted">Hali buyurtma yo\'q</div>', unsafe_allow_html=True)
        return
    lines = [f'<div class="tb-row"><span>#{html.escape(order["id"])}</span>'
             f'<span class="tb-status tb-{html.escape(order["status"])}">'
             f'{html.escape(STATUS_LABELS.get(order["status"], order["status"]))}</span></div>'
             for order in my_orders[:SIDEBAR_ORDERS]]
    st.markdown("".join(lines), unsafe_allow_html=True)


def sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="tb-side-brand">TexnoBozor</div>', unsafe_allow_html=True)
        if not st.session_state.customer_id:
            login_form()
            return
        st.markdown(f'<div class="tb-muted">{html.escape(st.session_state.customer_name)}</div>',
                    unsafe_allow_html=True)
        st.button("Chiqish", on_click=logout, type="tertiary")
        show_cart(st.session_state.customer_id)
        show_orders(st.session_state.customer_id)


# ======================= chat =======================

def header() -> None:
    st.markdown('<div class="tb-header"><span class="tb-brand">TexnoBozor</span>'
                '<span class="tb-sub">elektronika do\'koni yordamchisi</span></div>',
                unsafe_allow_html=True)


def show_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(as_markdown(message["content"]))
            if message["sources"]:
                sources = html.escape(", ".join(message["sources"]))
                st.markdown(f'<span class="tb-source">Manba: {sources}</span>', unsafe_allow_html=True)


def suggestions() -> None:
    """Shown only on an empty chat, so a new customer sees what they can ask."""
    columns = st.columns(2)
    for i, text in enumerate(SUGGESTIONS):
        columns[i % 2].button(text, key=f"suggest-{i}", on_click=queue_question, args=(text,),
                              use_container_width=True)


def queue_question(text: str) -> None:
    st.session_state.queued_question = text


def decide(interrupt_id: str, answer: str) -> None:
    """Button callback. Runs BEFORE the re-run, so the choice is ready when main() starts."""
    st.session_state.decision = {"id": interrupt_id, "answer": answer}


def as_markdown(text: str) -> str:
    return text.replace("\n", "  \n")   # two spaces = line break in Markdown


def confirmation_box(pending) -> None:
    with st.container(border=True):
        st.markdown('<div class="tb-card-title">Tasdiqlaysizmi?</div>', unsafe_allow_html=True)
        st.markdown(as_markdown(pending.value["summary"]))
        left, right = st.columns(2)
        # The interrupt id travels with the click, so it answers this question only.
        left.button("✅ Tasdiqlash", on_click=decide, args=(pending.id, APPROVE), type="primary",
                    use_container_width=True)
        right.button("❌ Bekor qilish", on_click=decide, args=(pending.id, REJECT),
                     use_container_width=True)


def send(question: str) -> None:
    add_message("user", question)
    with st.chat_message("user"):   # show it now, not only after the answer arrives
        st.markdown(question)
    ask(question)
    st.rerun()                      # redraw: new messages, fresh cart and orders in the sidebar


def main() -> None:
    st.set_page_config(page_title="TexnoBozor", page_icon="🛍️")
    st.markdown(f"<style>{STYLE_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    init_session()
    sidebar()
    header()
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
    elif len(ss.messages) == 1:
        suggestions()

    if queued := ss.pop("queued_question", None):
        send(queued)

    # While a confirmation is open, the customer answers with the buttons, not by typing.
    question = st.chat_input("Savol yozing — masalan, 6 mln gacha noutbuk",
                             disabled=pending is not None)
    if question:
        send(question)


if __name__ == "__main__":   # Streamlit runs the file as __main__; tests import it safely
    main()
