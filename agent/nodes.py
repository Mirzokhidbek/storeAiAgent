"""Graph nodes. Each node takes the state and returns only the fields it changes."""

from functools import lru_cache

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, trim_messages
from pydantic import BaseModel, Field

from agent.llm import get_chat_model
from agent.prompts import CHITCHAT_PROMPT, NOT_READY, ROUTER_PROMPT
from agent.state import Intent, ShopState

# How much history each LLM call sees. Protects the context window (and the quota).
ROUTER_HISTORY = 6     # a few messages are enough to resolve "uning narxi qancha?"
ANSWER_HISTORY = 20


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
    return {"intent": decision.intent, "context": [], "sources": [], "retries": 0}


def chitchat(state: ShopState) -> dict:
    history = recent_messages(state["messages"], ANSWER_HISTORY)
    reply = _chat_llm().invoke([SystemMessage(CHITCHAT_PROMPT), *history])
    return {"messages": [AIMessage(reply.text.strip())]}


def not_ready(state: ShopState) -> dict:
    """Temporary stand-in for nodes that later sub-phases will build."""
    return {"messages": [AIMessage(NOT_READY)]}
