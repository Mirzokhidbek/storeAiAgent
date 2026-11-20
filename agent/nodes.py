"""Graph nodes. Each node takes the state and returns only the fields it changes."""

import json
from functools import lru_cache

from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage, SystemMessage,
                                     ToolMessage, trim_messages)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

import config
from agent.llm import get_chat_model
from agent.prompts import (CHITCHAT_PROMPT, HANDOFF_MESSAGE, NO_INFO, NOT_READY,
                           ORDER_SUPPORT_PROMPT, PRODUCT_SEARCH_PROMPT, RESPOND_PROMPT,
                           REWRITE_PROMPT, ROUTER_PROMPT)
from agent.state import Intent, ShopState
from agent.tools import CATALOG_TOOLS, ORDER_READ_TOOLS
from rag.retriever import search_policies

# How much history each LLM call sees. Protects the context window (and the quota).
ROUTER_HISTORY = 6     # a few messages are enough to resolve "uning narxi qancha?"
ANSWER_HISTORY = 20
# Max LLM <-> tool round trips in one node (e.g. search, then get_product). Caps cost.
MAX_TOOL_ROUNDS = 3
# Tools each node may use. place_order / cancel_order are not here on purpose:
# the LLM in these nodes cannot even see them.
TOOL_GROUPS = {"catalog": CATALOG_TOOLS, "order": ORDER_READ_TOOLS}


class RouteDecision(BaseModel):
    """Structured output schema: the model must fill exactly this field, nothing else."""

    intent: Intent = Field(description="Mijozning oxirgi xabari turi")


@lru_cache
def _router_llm():
    # temperature=0 + low thinking: classification should be fast and repeatable.
    # ("minimal" would be even cheaper, but gemini-3.8-flash rejects it with a 400 error.)
    llm = get_chat_model(temperature=0, thinking_level="low")
    return llm.with_structured_output(RouteDecision)


@lru_cache
def _chat_llm():
    return get_chat_model()


@lru_cache
def _respond_llm():
    # Hidden thinking tokens count against LLM_MAX_TOKENS. With full thinking, long answers
    # got cut off mid-sentence. The data is already collected, so low thinking is enough.
    return get_chat_model(thinking_level="low")


@lru_cache
def _rewrite_llm():
    # temperature=0: a rewrite should be a faithful rephrasing, not a creative one.
    return get_chat_model(temperature=0, thinking_level="low")


@lru_cache
def _tool_llm(group: str):
    # Picking tools is a simple task, so low thinking is enough.
    return get_chat_model(thinking_level="low").bind_tools(TOOL_GROUPS[group])


def recent_messages(messages: list[BaseMessage], count: int) -> list[BaseMessage]:
    """Keep the last `count` messages, always starting on a user message."""
    return trim_messages(
        messages,
        max_tokens=count,
        token_counter=len,      # len counts messages, not tokens: simple and predictable
        strategy="last",
        start_on="human",
    )


def router(state: ShopState) -> dict:
    history = recent_messages(state["messages"], ROUTER_HISTORY)
    decision: RouteDecision = _router_llm().invoke([SystemMessage(ROUTER_PROMPT), *history])
    # A new question starts with a clean per-turn scratchpad.
    return {"intent": decision.intent, "context": [], "sources": [], "retries": 0,
            "search_query": None}


def chitchat(state: ShopState) -> dict:
    history = recent_messages(state["messages"], ANSWER_HISTORY)
    reply = _chat_llm().invoke([SystemMessage(CHITCHAT_PROMPT), *history])
    return {"messages": [AIMessage(reply.text.strip())]}


def _last_question(state: ShopState) -> str:
    human = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    return human[-1].content if human else ""


def run_tool_loop(state: ShopState, config: RunnableConfig, group: str, prompt: str) -> list[dict]:
    """Let the LLM call tools until it has enough data; return every call as context.

    The tool messages stay inside this function. They are not added to state["messages"],
    so the chat history (and every later prompt) stays short.
    """
    by_name = {t.name: t for t in TOOL_GROUPS[group]}
    # customer_id comes from graph state, never from the LLM (see agent/tools.py).
    tool_config = {**config, "configurable": {**config.get("configurable", {}),
                                              "customer_id": state.get("customer_id")}}
    messages = [SystemMessage(prompt), *recent_messages(state["messages"], ROUTER_HISTORY)]
    context = []

    for _ in range(MAX_TOOL_ROUNDS):
        ai = _tool_llm(group).invoke(messages, config)
        if not ai.tool_calls:
            break  # the model said it is done
        messages.append(ai)
        for call in ai.tool_calls:
            tool = by_name.get(call["name"])
            result = (tool.invoke(call["args"], tool_config) if tool
                      else {"error": f"Noma'lum tool: {call['name']}"})
            context.append({"tool": call["name"], "args": call["args"], "result": result})
            messages.append(ToolMessage(json.dumps(result, ensure_ascii=False),
                                        tool_call_id=call["id"]))
    return context


def product_search(state: ShopState, config: RunnableConfig) -> dict:
    return {"context": run_tool_loop(state, config, "catalog", PRODUCT_SEARCH_PROMPT)}


def order_support(state: ShopState, config: RunnableConfig) -> dict:
    return {"context": run_tool_loop(state, config, "order", ORDER_SUPPORT_PROMPT)}


def retrieval_zone(top_score: float) -> str:
    """Three zones by the best chunk's score. Thresholds are calibrated in SPEC.md §6."""
    if top_score >= config.RAG_SCORE_THRESHOLD:
        return "confident"
    if top_score >= config.RAG_LOW_THRESHOLD:
        return "doubtful"
    return "no_answer"


def policy_rag(state: ShopState) -> dict:
    """Retrieval step of RAG: search, then decide which zone the result falls into."""
    query = state.get("search_query") or _last_question(state)
    # Ask for everything above the LOW line, so we can see "doubtful" scores too.
    chunks = search_policies(query, threshold=config.RAG_LOW_THRESHOLD)
    top_score = max((c["score"] for c in chunks), default=0.0)
    attempt = {
        "tool": "search_policies",
        "args": {"query": query},
        # Only confident chunks may be used for the answer (and as sources).
        "result": [c for c in chunks if c["score"] >= config.RAG_SCORE_THRESHOLD],
        "zone": retrieval_zone(top_score),
        "top_score": top_score,
    }
    # Keep earlier attempts too: the terminal shows the whole retry history.
    return {"context": [*state.get("context", []), attempt]}


def route_after_policy(state: ShopState) -> str:
    """Where to go after a search. This is also where the retry loop stops."""
    zone = state["context"][-1]["zone"]
    retries = state.get("retries", 0)
    if zone == "confident":
        return "respond"
    if zone == "no_answer" and retries == 0:
        return "respond"   # the user's own words are far from every document: "ma'lumotim yo'q"
    # Doubtful, or a rewrite that drifted off-topic. A bad rewrite does not prove the
    # documents have no answer, so it counts as one more failed attempt.
    if retries < config.MAX_RETRIEVAL_RETRIES:
        return "rewrite_query"
    return "handoff"       # rewrote MAX times and still unsure: a human should answer


def rewrite_query(state: ShopState) -> dict:
    """Rephrase the question for a better search, using the chat history."""
    tried = [item["args"]["query"] for item in state["context"]
             if item["tool"] == "search_policies"]
    system = SystemMessage(REWRITE_PROMPT + "\n\nOldin sinalgan so'rovlar:\n"
                           + "\n".join(f"- {q}" for q in tried))
    history = recent_messages(state["messages"], ROUTER_HISTORY)
    reply = _rewrite_llm().invoke([system, *history]).text.strip()
    new_query = reply.splitlines()[0].strip(' "\'') if reply else tried[-1]
    return {"search_query": new_query, "retries": state.get("retries", 0) + 1}


def handoff(state: ShopState) -> dict:
    """Polite fixed message. Fixed text on purpose: nothing for the LLM to get wrong."""
    return {"messages": [AIMessage(HANDOFF_MESSAGE)], "sources": []}


def _policy_chunks(context: list[dict]) -> list[dict]:
    return [c for item in context if item["tool"] == "search_policies" for c in item["result"]]


def format_context(context: list[dict]) -> str:
    """Delimiters mark where trusted instructions end and plain data begins (SPEC §12)."""
    parts = [f'<natija tool="{item["tool"]}">\n'
             f'{json.dumps(item["result"], ensure_ascii=False, indent=1)}\n</natija>'
             for item in context
             if item["result"] != []]   # failed search attempts add nothing useful
    return "<malumotlar>\n" + "\n".join(parts) + "\n</malumotlar>"


def _strip_source_lines(text: str) -> str:
    # Sources are code's job. If the model wrote one anyway, drop it: it may be invented.
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("Manba:")).strip()


def respond(state: ShopState) -> dict:
    """Generation step: write the final Uzbek answer from context, then add sources by code."""
    context = state.get("context") or []
    chunks = _policy_chunks(context)

    if state["intent"] == "policy" and not chunks:
        # Nothing passed the threshold: answer without the LLM, so it cannot invent anything.
        return {"messages": [AIMessage(NO_INFO)], "sources": []}

    history = recent_messages(state["messages"], ANSWER_HISTORY)
    # One system message: Gemini expects the system prompt only at the start.
    system = SystemMessage(f"{RESPOND_PROMPT}\n\nOxirgi savol uchun ma'lumotlar:\n{format_context(context)}")
    reply = _strip_source_lines(_respond_llm().invoke([system, *history]).text)

    sources = [] if NO_INFO in reply else sorted({c["source"] for c in chunks})
    if sources:
        reply += f"\n\nManba: {', '.join(sources)}"
    return {"messages": [AIMessage(reply)], "sources": sources}


def not_ready(state: ShopState) -> dict:
    """Temporary stand-in for nodes that later sub-phases will build."""
    return {"messages": [AIMessage(NOT_READY)]}
