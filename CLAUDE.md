# TexnoBozor AI Assistant

A learning project: an AI shopping assistant for a fictional Uzbek electronics store ("TexnoBozor").
Stack: Python, Gemini (via LangChain), LangGraph, Chroma (RAG), SQLite, MCP, Streamlit.

**The full specification is in `SPEC.md`. Read the relevant section before starting any phase.**

## How to work in this repo

- Build strictly **phase by phase** (SPEC.md §14). Do only the phase the user asks for.
- At the end of each phase: run the phase's acceptance checks, then **stop** and report.
- Do not start the next phase until the user confirms.
- Keep changes small and focused. Do not refactor unrelated code.
- If something in SPEC.md is ambiguous or seems wrong, ask instead of guessing.

## The user is learning

The user is a student learning AI engineering (LLM, embeddings, RAG, LangChain, LangGraph, MCP).
After each phase, write a short summary **in Uzbek** (simple words) covering:
1. What was built and which files changed
2. Which concept from the course it uses (e.g. "bu RAG'ning Retrieval qismi")
3. How the user can run and test it themselves

Keep code readable for a learner: clear names, short functions, brief comments on *why*, not *what*.

## Library APIs change fast

LangChain, LangGraph, `langchain-google-genai`, `mcp` and `langchain-mcp-adapters` change often.
Before writing code that uses them, check the currently installed version and its docs.
Do not rely on memorised APIs. Pin versions in `requirements.txt` once they work.

## Hard rules (never break)

- Prices, stock and order status come **only** from SQLite via tools. Never from the LLM or RAG.
- `customer_id` is injected from the session/graph state. The LLM must never choose or change it.
- `place_order` and `cancel_order` require explicit user confirmation (LangGraph `interrupt()`).
- Secrets live in `.env` only. `.env` must be in `.gitignore`. Never print API keys.
- User-facing text (assistant replies, UI labels) is in **Uzbek (Latin script)**. Code, identifiers and comments are in English.

## Commands

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python hello_llm.py             # Phase 0: one Gemini call, prints answer + token usage
python data/seed_data.py        # create shop.db with fake data
python rag/ingest.py            # load policies + product descriptions into Chroma
python ask_policy.py "Mahsulotni qaytarsam bo'ladimi?"   # Phase 2: policy Q&A from the terminal
python -m pytest -v             # Phase 3: unit tests for catalog/order tools (no API key needed)
python -m tests.run_eval        # run evaluation questions
streamlit run app.py            # start the UI
```

Update this section if commands change.
