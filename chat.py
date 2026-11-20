"""Phase 4: chat with the agent in the terminal as test customer C-001.

Usage:
    python chat.py          # empty line or "chiqish" to quit
"""

import uuid

from langchain_core.messages import HumanMessage

from agent.graph import build_graph, run_config

TEST_CUSTOMER_ID = "C-001"


def main() -> None:
    agent = build_graph()
    # One thread_id per terminal session: every turn continues the same conversation.
    config = run_config(thread_id=f"terminal-{uuid.uuid4().hex[:8]}")

    print(f"TexnoBozor yordamchisi (mijoz: {TEST_CUSTOMER_ID}). Chiqish uchun Enter.\n")
    while True:
        question = input("Siz: ").strip()
        if not question or question.lower() == "chiqish":
            break

        # Only the new message is sent; the checkpointer adds the earlier history itself.
        # customer_id comes from code, never from the LLM.
        state = agent.invoke(
            {"messages": [HumanMessage(question)], "customer_id": TEST_CUSTOMER_ID},
            config,
        )
        print(f"   [intent: {state['intent']}]")
        print(f"Yordamchi: {state['messages'][-1].content}\n")

    print("Xayr!")


if __name__ == "__main__":
    main()
