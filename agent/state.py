"""The graph's shared memory: every node reads from and writes to this dict."""

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

Intent = Literal["product", "policy", "order", "action", "chitchat", "handoff"]


class ShopState(TypedDict):
    # add_messages is a reducer: new messages are appended, not overwritten.
    messages: Annotated[list, add_messages]
    customer_id: str | None          # set by the UI/terminal, never by the LLM
    intent: Intent | None            # chosen by the router for the latest message
    context: list[dict]              # retrieved chunks / tool results
    sources: list[str]               # shown under the answer
    retries: int                     # retrieval retry counter
    search_query: str | None         # rewritten policy question (None = use the user's words)
    pending_action: dict | None      # action awaiting confirmation
