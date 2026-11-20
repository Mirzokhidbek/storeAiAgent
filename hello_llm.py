"""Phase 0: send one question to Gemini and print the answer and token usage."""

from agent.llm import get_chat_model

QUESTION = "Salom! O'zbekiston poytaxti qaysi shahar? Bir-ikki gapda javob ber."


def main() -> None:
    llm = get_chat_model()
    response = llm.invoke(QUESTION)

    print("Savol:", QUESTION)
    print("Javob:", response.text)

    # usage_metadata shows how much of the context window and quota we spent.
    usage = response.usage_metadata or {}
    print("\nTokenlar:")
    print("  kirish (input): ", usage.get("input_tokens"))
    print("  chiqish (output):", usage.get("output_tokens"))
    print("  jami (total):   ", usage.get("total_tokens"))


if __name__ == "__main__":
    main()
