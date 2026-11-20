"""Builds the LangGraph agent: START -> router -> one node per intent -> END.

Phase 4a: only `chitchat` is real. The other intent nodes return a "not ready" stub
and will be replaced in the next sub-phases.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.state import ShopState

# Stops runaway loops (e.g. retrieval retries later). Each node visit counts as one step.
RECURSION_LIMIT = 15

# intent -> node name. Names match SPEC.md §8 so later phases just swap the function.
INTENT_TO_NODE = {
    "chitchat": "chitchat",
    "product": "product_search",
    "policy": "policy_rag",
    "order": "order_support",
    "action": "confirm_action",
    "handoff": "handoff",
}
STUB_NODES = ["product_search", "policy_rag", "order_support", "confirm_action", "handoff"]


def route_by_intent(state: ShopState) -> str:
    return INTENT_TO_NODE[state["intent"]]


def build_graph():
    graph = StateGraph(ShopState)
    graph.add_node("router", nodes.router)
    graph.add_node("chitchat", nodes.chitchat)
    for name in STUB_NODES:
        graph.add_node(name, nodes.not_ready)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_by_intent, list(INTENT_TO_NODE.values()))
    for name in INTENT_TO_NODE.values():
        graph.add_edge(name, END)

    # The checkpointer saves the state after every step, keyed by thread_id.
    # That is what gives the agent memory between turns.
    return graph.compile(checkpointer=MemorySaver())


def run_config(thread_id: str) -> dict:
    """Same thread_id = same conversation (same saved state)."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
