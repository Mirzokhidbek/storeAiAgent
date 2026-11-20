"""Builds the LangGraph agent.

    START -> router -> product_search | order_support -> respond -> END
                    -> policy_rag -> respond | rewrite_query | handoff
                         rewrite_query -> policy_rag            (the retry loop)
                    -> chitchat -> END
                    -> handoff -> END
                    -> confirm_action -> execute_action -> respond -> END
                                                        -> END      ("n": fixed message)
                                      -> respond -> END  (empty cart, foreign order...)

Data-collecting nodes only fill state["context"]; `respond` writes the one final answer.
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
# These nodes collect data, then hand over to `respond` for the final answer.
DATA_NODES = {"product_search": nodes.product_search, "order_support": nodes.order_support}


def route_by_intent(state: ShopState) -> str:
    return INTENT_TO_NODE[state["intent"]]


def build_graph():
    graph = StateGraph(ShopState)
    graph.add_node("router", nodes.router)
    graph.add_node("chitchat", nodes.chitchat)
    graph.add_node("respond", nodes.respond)
    graph.add_node("policy_rag", nodes.policy_rag)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("handoff", nodes.handoff)
    for name, node in DATA_NODES.items():
        graph.add_node(name, node)
    graph.add_node("confirm_action", nodes.confirm_action)
    graph.add_node("execute_action", nodes.execute_action)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_by_intent, list(INTENT_TO_NODE.values()))
    for name in DATA_NODES:
        graph.add_edge(name, "respond")
    # Policy retry loop: policy_rag -> rewrite_query -> policy_rag ... until confident,
    # clearly unanswerable, or MAX_RETRIEVAL_RETRIES rewrites (then handoff).
    graph.add_conditional_edges("policy_rag", nodes.route_after_policy,
                                ["respond", "rewrite_query", "handoff"])
    graph.add_edge("rewrite_query", "policy_rag")
    # Actions: confirm (summary + y/n) -> execute only if approved.
    graph.add_conditional_edges("confirm_action", nodes.route_after_confirm,
                                ["execute_action", "respond"])
    graph.add_conditional_edges("execute_action", nodes.route_after_execute,
                                {"respond": "respond", "end": END})
    for name in ["chitchat", "respond", "handoff"]:
        graph.add_edge(name, END)

    # The checkpointer saves the state after every step, keyed by thread_id.
    # That is what gives the agent memory between turns.
    return graph.compile(checkpointer=MemorySaver())


def run_config(thread_id: str, confirm=None) -> dict:
    """Same thread_id = same conversation (same saved state).

    confirm: function(summary_text) -> bool that asks the customer to approve an order
    or a cancellation. Without it every such action is refused (safe default).
    """
    configurable = {"thread_id": thread_id}
    if confirm:
        configurable["confirm"] = confirm
    return {"configurable": configurable, "recursion_limit": RECURSION_LIMIT}
