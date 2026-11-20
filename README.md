# TexnoBozor AI Assistant

A chat assistant for a fictional Uzbek electronics store. It recommends products, answers
store-policy questions, shows order status, adds to cart, and places or cancels orders only
after the customer confirms. Replies are in Uzbek.

Learning project: Gemini · LangChain · LangGraph · RAG (Chroma) · SQLite · MCP · Streamlit.
**Status: Phase 7 of 7 complete.** Full spec: [SPEC.md](SPEC.md).

## How it works

```
app.py (Streamlit) / chat.py (terminal)
  └─ LangGraph agent (agent/)
       ├─ RAG → Chroma (policy docs, product descriptions)
       └─ MCP servers (mcp_servers/, stdio) → shop/ → SQLite
```

| Part | Where | What it does |
|---|---|---|
| Router | `agent/nodes.py` | Gemini labels each message: product, policy, order, action, chitchat or handoff |
| Product search | `agent/nodes.py` + `catalog_server` | Gemini calls catalog tools. Prices and stock are read from SQLite |
| Policy RAG | `rag/retriever.py` | Three zones by score: ≥0.75 answer with source, 0.68–0.75 rewrite and retry, <0.68 "no info" |
| Orders | `orders_server` | Status, cart, order history. Ownership is checked in SQL |
| Confirmation | `prepare_action` → `confirm_action` → `execute_action` | Builds the summary, pauses with `interrupt()`, runs `place_order` / `cancel_order` only after approval |
| Memory | `MemorySaver` | Keeps each conversation by `thread_id`. Logout deletes the thread |
| MCP client | `agent/mcp_client.py` | Keeps both MCP sessions open on one background event loop |
| UI | `app.py` | Phone login, cart and orders sidebar, sources under answers, ✅/❌ buttons |

**Safety rules**
- Prices, stock and order status come only from the database. The LLM never makes them up.
- The LLM never sees or picks `customer_id`. The app adds it from the logged-in session.
- Orders are placed or cancelled only after the customer clicks "Tasdiqlash".

## Run it

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `GOOGLE_API_KEY`, `GEMINI_MODEL`, `EMBEDDING_MODEL`
4. `python data/seed_data.py` creates the fake store database
5. `python rag/ingest.py` loads the docs into Chroma
6. `streamlit run app.py` and log in with a test phone:
   - `+998901234567` (C-001): has a cart, order 1234 (delivered) and order 1001 (cancellable)
   - `+998912345678` (C-002): has order 5678, which C-001 cannot see

Other commands: `python chat.py` (terminal chat), `python ask_policy.py "Kafolat necha oy?"` (RAG only),
`python -m pytest -v` (tests, no API key needed), `python -m tests.run_eval` (eval with real Gemini).

## Test results

**150 tests pass.** They use fake LLMs and a temporary copy of the database, so they need no API key and cost nothing.

| File | Tests | What it checks |
|---|---|---|
| `test_tools.py` | 41 | Price filter, out-of-stock items, C-001 cannot read order 5678, cancel rules, stock updates |
| `test_graph.py` | 16 | Routing, the RAG zones and rewrite loop, handoff, sources added by code, memory |
| `test_actions.py` | 17 | No order while paused, approve and reject, empty cart, someone else's order, cart changed while paused, no double resume |
| `test_app.py` | 21 | Streamlit via AppTest: login, Uzbek error messages, buttons, page reload, double click, logout clears memory |
| `test_mcp.py` | 3 | Real MCP servers return exactly what the in-process tools return, and the LLM never sees `customer_id` |
| `test_eval.py` | 52 | The evaluator itself: case file format, and each check catches bad answers and passes good ones |

**Live checks with Gemini:**
- Terminal and browser: every intent works, and follow-ups like "uning narxi qancha?" use memory.
- Same 7 questions with and without MCP gave identical intents, tool calls, tool results and order summaries.

## Evaluation (Phase 7)

`python -m tests.run_eval` sends the 30 questions in `tests/eval_questions.json` to the real agent (real Gemini,
MCP tools) and checks intent, required and forbidden phrases, sources, confirmation and DB changes.
It prints the pass rate, total tokens and an estimated cost. Each case runs on a fresh seed database,
so `data/shop.db` is never touched (checked by hash). Confirmations are always rejected.

Covered: every category in SPEC §15, plus Cyrillic Uzbek, Russian and questions without apostrophes.

**Latest full run: 29/30 passed**, about 72k tokens (~$0.04, estimated) and 8.6 minutes.
Failed cases were re-run 2–3 times to separate random failures from real ones.

Fixed during Phase 7 (each re-checked live afterwards):
- Russian questions were rewritten in Russian and missed the Uzbek docs. Rewrites are now always in Uzbek Latin (passed 3/3).
- `product-none`: the agent offered laptops above the 100 000 so'm budget without saying they cost more. The respond prompt now requires saying so (passed 3/3).
- `cyrillic-order`: the agent sometimes wrote "yetkazib berilgan" instead of "yetkazildi". The meaning is correct, so the order cases now accept both (passed 2/2).

The full 30-question run has not been repeated since these last two fixes.

## Known limits

- Memory lives in RAM, so restarting the server clears chats and pending confirmations.
- Login is phone number only (no real authentication).
- In the browser, sending a new message before the reply arrives can hide the earlier reply. The action itself still runs.
- The Gemini free tier is rate-limited. The app then shows a clear message in Uzbek.
