"""Graph tests for Phase 4b nodes, with fake LLMs.

Run:  python -m pytest -v
No API key needed: every LLM (router, tool-calling, respond) and the policy retriever
are replaced by fakes that return scripted answers. So we test OUR code — routing,
tool execution, customer_id injection, the source line — not Gemini's mood.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent import nodes
from agent.graph import build_graph, run_config
from agent.prompts import NO_INFO


class FakeLLM:
    """Returns pre-written replies in order and remembers every call it got."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        return self.replies.pop(0)


def tool_call(name: str, **args) -> AIMessage:
    """An AI message that asks to run one tool, like Gemini would send."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}])


@pytest.fixture
def fake(monkeypatch):
    """Swap the real LLMs for fakes. Each test sets the replies it needs."""
    llms = {"router": FakeLLM(), "tool": FakeLLM(), "respond": FakeLLM()}
    monkeypatch.setattr(nodes, "_router_llm", lambda: llms["router"])
    monkeypatch.setattr(nodes, "_tool_llm", lambda group: llms["tool"])
    monkeypatch.setattr(nodes, "_respond_llm", lambda: llms["respond"])
    return llms


def ask(question: str, customer_id: str | None = "C-001") -> dict:
    agent = build_graph()
    return agent.invoke({"messages": [HumanMessage(question)], "customer_id": customer_id},
                        run_config("test"))


# ======================= policy_rag + respond =======================

def test_policy_below_threshold_skips_llm_and_says_no_info(fake, monkeypatch):
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    # The real retriever drops chunks under RAG_SCORE_THRESHOLD; nothing passed here.
    monkeypatch.setattr(nodes, "search_policies", lambda query: [])

    state = ask("Sport zali uchun chegirma bormi?")

    assert state["messages"][-1].content == NO_INFO
    assert state["sources"] == []
    assert fake["respond"].calls == [], "LLM must not be called when no chunk passed"


def test_source_line_is_added_by_code_not_llm(fake, monkeypatch):
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    monkeypatch.setattr(nodes, "search_policies", lambda query: [
        {"text": "14 kun ichida qaytarish mumkin.", "source": "qaytarish.md",
         "section": "Qaytarish", "score": 0.85},
        {"text": "Qaytarish uchun buyurtma raqami kerak.", "source": "faq.md",
         "section": "FAQ", "score": 0.80},
    ])
    # The LLM also writes a (fake) source line. Code must drop it and add the real one.
    fake["respond"].replies = [AIMessage("14 kun ichida qaytarishingiz mumkin.\nManba: internet.md")]

    state = ask("Mahsulotni qaytarsam bo'ladimi?")
    answer = state["messages"][-1].content

    assert answer.endswith("Manba: faq.md, qaytarish.md")
    assert "internet.md" not in answer
    assert state["sources"] == ["faq.md", "qaytarish.md"]


def test_no_source_line_when_llm_says_no_info(fake, monkeypatch):
    # Chunk passed the threshold, but it does not answer the question (e.g. trade-in).
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    monkeypatch.setattr(nodes, "search_policies", lambda query: [
        {"text": "Yetkazib berish 1-2 kun.", "source": "yetkazib_berish.md",
         "section": "Toshkent", "score": 0.76}])
    fake["respond"].replies = [AIMessage(NO_INFO)]

    state = ask("Trade-in qilasizlarmi?")

    assert "Manba:" not in state["messages"][-1].content
    assert state["sources"] == []


# ======================= order_support =======================

def run_order_question(fake, order_id: str) -> dict:
    fake["router"].replies = [nodes.RouteDecision(intent="order")]
    # The LLM asks for the order, then says it is done. Real tools + real (temp) DB run.
    fake["tool"].replies = [tool_call("get_order_status", order_id=order_id),
                            AIMessage("TAYYOR")]
    fake["respond"].replies = [AIMessage("javob")]
    return ask(f"{order_id} buyurtma holati?", customer_id="C-001")


def test_order_support_does_not_return_other_customers_order(fake):
    state = run_order_question(fake, "5678")   # 5678 belongs to C-002

    result = state["context"][0]["result"]
    assert state["context"][0]["tool"] == "get_order_status"
    assert "error" in result
    assert "status" not in result
    # Nothing about the foreign order reached the LLM that writes the answer.
    respond_prompt = fake["respond"].calls[0][0].content
    assert "yolda" not in respond_prompt


def test_order_support_returns_own_order(fake):
    # Control case: proves the test above fails for the right reason (ownership).
    state = run_order_question(fake, "1234")   # 1234 belongs to C-001

    result = state["context"][0]["result"]
    assert result["status"] == "yetkazildi"


def test_tool_messages_stay_out_of_chat_history(fake):
    state = run_order_question(fake, "1234")

    # History keeps only the user's question and the final answer, not tool calls.
    assert [m.type for m in state["messages"]] == ["human", "ai"]
