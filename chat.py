"""Phase 4: chat with the agent in the terminal as test customer C-001.

Usage:
    python chat.py          # empty line or "chiqish" to quit
"""

import uuid

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage

from agent.graph import build_graph, run_config

TEST_CUSTOMER_ID = "C-001"


def ask_yes_no(summary: str) -> bool:
    """Phase 4 confirmation: the customer types y/n. Phase 6 replaces this with interrupt()."""
    print(f"\n{summary}")
    answer = input("Tasdiqlaysizmi? (y/n): ").strip().lower()
    return answer in ("y", "ha")


def print_debug(state: dict, usage_by_model: dict) -> None:
    """Shown for learning: what the graph did behind the answer."""
    tools = [describe_call(c) for c in state.get("context") or []]
    print(f"   [intent: {state['intent']}]")
    print(f"   [tool'lar: {', '.join(tools) or "yo'q"}]")
    if state.get("retries"):
        print(f"   [qayta yozishlar: {state['retries']}]")
    tokens_in = sum(u.get("input_tokens", 0) for u in usage_by_model.values())
    tokens_out = sum(u.get("output_tokens", 0) for u in usage_by_model.values())
    print(f"   [tokenlar: kirish {tokens_in}, chiqish {tokens_out}, jami {tokens_in + tokens_out}]")


def describe_call(call: dict) -> str:
    text = f"{call['tool']}({format_args(call['args'])})"
    if "zone" in call:   # policy search: show its score and zone, to see the retry loop
        text += f" -> {call['top_score']:.3f} {call['zone']}"
    return text


def format_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def main() -> None:
    agent = build_graph()
    # One thread_id per terminal session: every turn continues the same conversation.
    config = run_config(thread_id=f"terminal-{uuid.uuid4().hex[:8]}", confirm=ask_yes_no)

    print(f"TexnoBozor yordamchisi (mijoz: {TEST_CUSTOMER_ID}). Chiqish uchun Enter.\n")
    while True:
        question = input("Siz: ").strip()
        if not question or question.lower() == "chiqish":
            break

        # Only the new message is sent; the checkpointer adds the earlier history itself.
        # customer_id comes from code, never from the LLM.
        # The callback adds up token usage of every LLM call in this turn
        # (router + tool-calling + respond), not just the last one.
        with get_usage_metadata_callback() as usage:
            state = agent.invoke(
                {"messages": [HumanMessage(question)], "customer_id": TEST_CUSTOMER_ID},
                config,
            )
        print(f"Yordamchi: {state['messages'][-1].content}")
        print_debug(state, usage.usage_metadata)
        print()

    print("Xayr!")


if __name__ == "__main__":
    main()
