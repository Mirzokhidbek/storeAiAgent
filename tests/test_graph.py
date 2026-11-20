"""Graph tests for Phase 4b nodes, with fake LLMs.

Run:  python -m pytest -v
No API key needed: every LLM (router, tool-calling, respond) and the policy retriever
are replaced by fakes that return scripted answers. So we test OUR code — routing,
tool execution, customer_id injection, the source line — not Gemini's mood.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import config
from agent import nodes
from agent.graph import build_graph, run_config
from agent.prompts import HANDOFF_MESSAGE, NO_INFO


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
    llms = {"router": FakeLLM(), "tool": FakeLLM(), "respond": FakeLLM(), "rewrite": FakeLLM()}
    monkeypatch.setattr(nodes, "_router_llm", lambda: llms["router"])
    monkeypatch.setattr(nodes, "_tool_llm", lambda group: llms["tool"])
    monkeypatch.setattr(nodes, "_respond_llm", lambda: llms["respond"])
    monkeypatch.setattr(nodes, "_rewrite_llm", lambda: llms["rewrite"])
    # Pin the SPEC values, so a different .env cannot change what the tests expect.
    monkeypatch.setattr(config, "RAG_SCORE_THRESHOLD", 0.75)
    monkeypatch.setattr(config, "RAG_LOW_THRESHOLD", 0.68)
    monkeypatch.setattr(config, "MAX_RETRIEVAL_RETRIES", 2)
    return llms


def chunk(score: float, source: str = "yetkazib_berish.md") -> dict:
    return {"text": "Viloyatlarga 3-5 kun, 50 000 so'm.", "source": source,
            "section": "Viloyatlar", "score": score}


class FakeRetriever:
    """Pretends to be search_policies: returns a fixed score for each query.
    Unknown queries score 0.70 (the doubtful zone). Remembers every query it got."""

    def __init__(self, scores: dict[str, float], default: float = 0.70):
        self.scores, self.default, self.queries = scores, default, []

    def __call__(self, query, threshold=None):
        self.queries.append(query)
        score = self.scores.get(query, self.default)
        return [chunk(score)] if score >= (threshold or 0) else []


def ask(question: str, customer_id: str | None = "C-001") -> dict:
    agent = build_graph()
    return agent.invoke({"messages": [HumanMessage(question)], "customer_id": customer_id},
                        run_config("test"))


# ======================= policy_rag + respond =======================

def test_policy_below_threshold_skips_llm_and_says_no_info(fake, monkeypatch):
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    # The real retriever drops chunks under RAG_SCORE_THRESHOLD; nothing passed here.
    monkeypatch.setattr(nodes, "search_policies", lambda query, threshold=None: [])

    state = ask("Sport zali uchun chegirma bormi?")

    assert state["messages"][-1].content == NO_INFO
    assert state["sources"] == []
    assert fake["respond"].calls == [], "LLM must not be called when no chunk passed"


def test_source_line_is_added_by_code_not_llm(fake, monkeypatch):
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    monkeypatch.setattr(nodes, "search_policies", lambda query, threshold=None: [
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
    monkeypatch.setattr(nodes, "search_policies", lambda query, threshold=None: [
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


# ======================= Phase 4c: three zones, rewrite loop, handoff =======================
# Zones (config): confident >= 0.75, doubtful >= 0.68, below = no answer.

def test_three_zones(fake):
    assert nodes.retrieval_zone(0.80) == "confident"
    assert nodes.retrieval_zone(0.75) == "confident"
    assert nodes.retrieval_zone(0.70) == "doubtful"
    assert nodes.retrieval_zone(0.68) == "doubtful"
    assert nodes.retrieval_zone(0.60) == "no_answer"


def test_doubtful_question_is_rewritten_then_answered(fake, monkeypatch):
    question = "Samarqandga olib borasizlarmi?"
    rewritten = "Viloyatlarga yetkazib berish muddati va narxi"
    search = FakeRetriever({question: 0.71, rewritten: 0.86})
    monkeypatch.setattr(nodes, "search_policies", search)
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    fake["rewrite"].replies = [AIMessage(rewritten)]
    fake["respond"].replies = [AIMessage("Viloyatlarga 3-5 kunda yetkazamiz.")]

    state = ask(question)

    assert search.queries == [question, rewritten]
    assert state["retries"] == 1
    assert state["messages"][-1].content.endswith("Manba: yetkazib_berish.md")


def test_clearly_unanswerable_skips_rewrite_and_llm(fake, monkeypatch):
    search = FakeRetriever({}, default=0.60)   # e.g. "Ob-havo qanday?" = 0.646
    monkeypatch.setattr(nodes, "search_policies", search)
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]

    state = ask("Ob-havo qanday?")

    assert state["messages"][-1].content == NO_INFO
    assert len(search.queries) == 1, "no second search"
    assert fake["rewrite"].calls == [] and fake["respond"].calls == []


def test_still_doubtful_after_max_retries_goes_to_handoff(fake, monkeypatch):
    search = FakeRetriever({}, default=0.70)   # every attempt stays doubtful
    monkeypatch.setattr(nodes, "search_policies", search)
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    fake["rewrite"].replies = [AIMessage("1-urinish"), AIMessage("2-urinish")]

    state = ask("Chegirmalar bormi?")

    # MAX_RETRIEVAL_RETRIES = 2: first search + 2 rewrites = 3 searches, then stop.
    assert len(fake["rewrite"].calls) == config.MAX_RETRIEVAL_RETRIES
    assert len(search.queries) == config.MAX_RETRIEVAL_RETRIES + 1
    assert state["messages"][-1].content == HANDOFF_MESSAGE
    assert fake["respond"].calls == []


def test_rewrite_that_drifts_off_topic_does_not_mean_no_answer(fake, monkeypatch):
    # Live case: "Chegirmalar bormi?" 0.733 -> rewrite 0.730 -> rewrite drifted to 0.0.
    search = FakeRetriever({"Chegirmalar bormi?": 0.73, "1-urinish": 0.73, "2-urinish": 0.0})
    monkeypatch.setattr(nodes, "search_policies", search)
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    fake["rewrite"].replies = [AIMessage("1-urinish"), AIMessage("2-urinish")]

    state = ask("Chegirmalar bormi?")

    assert state["messages"][-1].content == HANDOFF_MESSAGE


def test_rewrite_sees_chat_history(fake, monkeypatch):
    """ "uning kafolati" can only be rewritten if the earlier answer is visible."""
    rewritten = "Lenovo IdeaPad noutbukining kafolati"
    search = FakeRetriever({rewritten: 0.88})
    monkeypatch.setattr(nodes, "search_policies", search)
    agent = build_graph()

    # Turn 1: a product question; its answer names the laptop.
    fake["router"].replies = [nodes.RouteDecision(intent="product")]
    fake["tool"].replies = [AIMessage("TAYYOR")]
    fake["respond"].replies = [AIMessage("Lenovo IdeaPad Slim 3 — 5 400 000 so'm.")]
    agent.invoke({"messages": [HumanMessage("Lenovo noutbuk bormi?")], "customer_id": "C-001"},
                 run_config("memory"))

    # Turn 2: a follow-up with a pronoun.
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    fake["rewrite"].replies = [AIMessage(rewritten)]
    fake["respond"].replies = [AIMessage("Noutbuklarga 12 oy kafolat beriladi.")]
    agent.invoke({"messages": [HumanMessage("uning kafolati-chi?")], "customer_id": "C-001"},
                 run_config("memory"))

    seen = " ".join(m.content for m in fake["rewrite"].calls[0])
    assert "Lenovo IdeaPad Slim 3" in seen, "rewrite LLM must get the earlier answer"
    assert search.queries[-1] == rewritten


def test_new_question_starts_fresh_retry_state(fake, monkeypatch):
    """A rewritten query or retry count must not leak into the next question."""
    rewritten = "Viloyatlarga yetkazib berish"
    search = FakeRetriever({"Samarqandga?": 0.70, rewritten: 0.86, "Kafolat necha oy?": 0.87})
    monkeypatch.setattr(nodes, "search_policies", search)
    agent = build_graph()

    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    fake["rewrite"].replies = [AIMessage(rewritten)]
    fake["respond"].replies = [AIMessage("3-5 kun.")]
    agent.invoke({"messages": [HumanMessage("Samarqandga?")]}, run_config("fresh"))

    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    fake["respond"].replies = [AIMessage("12 oy.")]
    state = agent.invoke({"messages": [HumanMessage("Kafolat necha oy?")]}, run_config("fresh"))

    assert search.queries[-1] == "Kafolat necha oy?"
    assert state["retries"] == 0
    assert [c["args"]["query"] for c in state["context"]] == ["Kafolat necha oy?"]


def test_handoff_intent_gives_fixed_message(fake):
    fake["router"].replies = [nodes.RouteDecision(intent="handoff")]

    state = ask("Operator bilan gaplashmoqchiman")

    assert state["messages"][-1].content == HANDOFF_MESSAGE
    assert fake["respond"].calls == [] and fake["tool"].calls == []


def test_rewrite_prompt_asks_for_uzbek(fake, monkeypatch):
    """Live eval: a Russian question was rewritten in Russian, scored lower each time
    and ended in handoff. The documents are Uzbek, so rewrites must be Uzbek Latin."""
    search = FakeRetriever({"Mahsulotni qaytarish mumkinmi?": 0.88})
    monkeypatch.setattr(nodes, "search_policies", search)
    fake["router"].replies = [nodes.RouteDecision(intent="policy")]
    fake["rewrite"].replies = [AIMessage("Mahsulotni qaytarish mumkinmi?")]
    fake["respond"].replies = [AIMessage("14 kun ichida qaytarish mumkin.")]

    ask("Можно ли вернуть товар?")

    system_prompt = fake["rewrite"].calls[0][0].content
    assert "o'zbek tilida, lotin yozuvida" in system_prompt
    assert search.queries[-1] == "Mahsulotni qaytarish mumkinmi?"
