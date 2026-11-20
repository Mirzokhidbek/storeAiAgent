"""Phase 7: run every case in eval_questions.json through the REAL agent (real Gemini).

Run:
    python -m tests.run_eval                       # all cases, MCP tools, 6 s between cases
    python -m tests.run_eval --only order-own,memory-followup
    python -m tests.run_eval --backend local --delay 10

Safety:
- every case gets its own fresh seed database in a temp folder (SHOP_DB_PATH for the
  MCP servers); data/shop.db is never used, and its hash is checked at the end
- a confirmation (interrupt) is always answered "reject", so no order is ever placed

Cost: token usage of every LLM call is summed. The price per 1M tokens comes from
EVAL_PRICE_INPUT_PER_1M_USD / EVAL_PRICE_OUTPUT_PER_1M_USD in .env (an estimate: check
the current price of your GEMINI_MODEL in Google AI Studio). Embedding calls made by
RAG and product search report no usage, so they are not included.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.exceptions import ModelRateLimitError
from langchain_core.messages import HumanMessage
from langgraph.types import Command

import config
import shop.db
from agent import tools
from agent.graph import build_graph, pending_confirmation, run_config
from agent.nodes import REJECT
from tests.conftest import build_seed_db

CASES_FILE = Path(__file__).with_name("eval_questions.json")
REPORT_FILE = Path(__file__).resolve().parent.parent / "eval_report.json"
TABLES = ["products", "customers", "orders", "order_items", "cart_items"]
PRICE_IN = float(os.getenv("EVAL_PRICE_INPUT_PER_1M_USD", "0.30"))
PRICE_OUT = float(os.getenv("EVAL_PRICE_OUTPUT_PER_1M_USD", "2.50"))
# Policy numbers that may appear next to "so'm" besides product prices (SPEC §5).
POLICY_AMOUNTS = {30_000, 50_000, 1_000_000, 2_000_000}
# Words that tell the customer an offered product is above their budget.
OVER_BUDGET_WORDS = ["qimmat", "byudjet", "budjet", "yuqori", "oshadi", "ortiq", "ko'proq",
                     "kattaroq", "baland"]
RATE_LIMIT_WAIT = 60   # seconds to wait after a 429 before retrying the case
MAX_ATTEMPTS = 3


# ======================= text helpers =======================

APOSTROPHES = "’‘ʻʼ`´"


def normalize(text: str) -> str:
    """Make comparisons robust to formatting, not to meaning:
    o’ / oʻ / o' -> o', lowercase, and "5 400 000" / "5,400,000" -> "5400000"."""
    for ch in APOSTROPHES:
        text = text.replace(ch, "'")
    text = text.replace(" ", " ").replace(" ", " ").lower()
    return re.sub(r"(?<=\d)[ ,.](?=\d{3}(?!\d))", "", text)


def as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def contains(answer: str, expected) -> bool:
    """expected is a string, or a list meaning "any of these"."""
    options = expected if isinstance(expected, list) else [expected]
    return any(normalize(option) in normalize(answer) for option in options)


def amounts_in(answer: str) -> set[int]:
    """Every number written as a price ("5 400 000 so'm") in the answer."""
    return {int(n) for n in re.findall(r"(\d{4,})\s*so'm", normalize(answer))}


# ======================= database helpers =======================

def snapshot(db_file: Path) -> dict:
    conn = shop.db.get_connection(db_file)
    rows = {t: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}")) for t in TABLES}
    conn.close()
    return rows


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "yo'q"


@contextmanager
def tools_on(db_file: Path, backend: str):
    """Point the agent's tools at this case's own database."""
    if backend == "local":
        old_path, shop.db.DB_PATH = shop.db.DB_PATH, db_file
        box = tools.LocalBackend()
    else:
        from agent.mcp_client import MCPToolbox
        box = MCPToolbox(extra_env={"SHOP_DB_PATH": str(db_file)})
    original = tools.backend
    tools.backend = lambda: box
    try:
        yield
    finally:
        tools.backend = original
        if backend == "local":
            shop.db.DB_PATH = old_path
        else:
            box.close()


# ======================= running one case =======================

def ask(agent, cfg, text: str, customer_id):
    return agent.invoke({"messages": [HumanMessage(text)], "customer_id": customer_id}, cfg)


def run_case(case: dict, backend: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        db_file = build_seed_db(Path(tmp) / "shop.db")
        with tools_on(db_file, backend), get_usage_metadata_callback() as usage:
            agent, cfg = build_graph(), run_config(f"eval-{case['id']}")
            for turn in case.get("setup_turns", []):       # e.g. for "uning narxi qancha?"
                ask(agent, cfg, turn, case["customer_id"])
            before = snapshot(db_file)
            state = ask(agent, cfg, case["question"], case["customer_id"])
            pending = pending_confirmation(agent, cfg)
            summary = pending.value["summary"] if pending else None
            if pending:
                state = agent.invoke(Command(resume=REJECT), cfg)   # never place an order
            after = snapshot(db_file)
        cart = after["cart_items"]

    answer = state["messages"][-1].content
    return {
        "intent": state.get("intent"),
        "answer": answer,
        "summary": summary,          # what the customer would have been asked to approve
        "sources": state.get("sources") or [],
        "context": state.get("context") or [],
        "db_changed": before != after,
        "cart": cart,
        "allowed_amounts": allowed_amounts(case, before, state.get("context") or []),
        "usage": {"in": sum(u.get("input_tokens", 0) for u in usage.usage_metadata.values()),
                  "out": sum(u.get("output_tokens", 0) for u in usage.usage_metadata.values())},
    }


def allowed_amounts(case: dict, db: dict, context: list[dict]) -> set[int]:
    """Amounts the answer may name: product prices, policy amounts, the customer's own
    numbers ("1 000 so'mga ber") and the price limits the search used ("6 mln gacha").

    Limit: because the customer's own number is allowed (a refusal may repeat it),
    prices_from_db alone cannot catch "Mayli, 1 000 so'mga beramiz". Cases about price
    manipulation therefore also list the real price in must_contain and agreement
    phrases in must_not_contain."""
    said = " ".join([case["question"], *case.get("setup_turns", [])])
    limits = {c["args"][key] for c in context if c["tool"] == "search_products"
              for key in ("min_price", "max_price") if c["args"].get(key)}
    return ({row[4] for row in db["products"]} | POLICY_AMOUNTS | amounts_in(said) | limits
            | price_differences(context))


def price_differences(context: list[dict]) -> set[int]:
    """"HP 500 000 so'mga arzonroq": the difference between two prices the tools returned
    in THIS answer. Only these products, not the whole catalog, so the check stays strict."""
    prices = set()
    for call in context:
        result = call["result"] if isinstance(call["result"], dict) else {}
        products = result.get("products", [result] if "price_uzs" in result else [])
        prices |= {p["price_uzs"] for p in products if isinstance(p, dict) and "price_uzs" in p}
    return {abs(a - b) for a in prices for b in prices if a != b}


def run_with_retry(case: dict, backend: str) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return run_case(case, backend)
        except ModelRateLimitError:
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"      Gemini limiti: {RATE_LIMIT_WAIT} s kutilyapti "
                  f"({attempt}/{MAX_ATTEMPTS - 1})...", flush=True)
            time.sleep(RATE_LIMIT_WAIT)


# ======================= checks =======================

def search_results(run: dict) -> list[list[dict]]:
    return [c["result"].get("products", []) for c in run["context"]
            if c["tool"] == "search_products" and isinstance(c["result"], dict)]


def run_check(check: str, case: dict, run: dict) -> str | None:
    """One named check from the case's "checks" list. Returns a problem or None."""
    name, _, value = check.partition(":")
    products = [p for result in search_results(run) for p in result]
    if name == "no_tools":
        return None if not run["context"] else f"tool chaqirilmasligi kerak edi: {[c['tool'] for c in run['context']]}"
    if name == "search_called":
        return None if search_results(run) else "search_products chaqirilmadi"
    if name == "search_empty_under":
        # Only the searches limited to this budget must be empty. A later search for
        # alternatives (without a price limit) is allowed: offering real options is fine.
        limited = [c["result"].get("products", []) for c in run["context"]
                   if c["tool"] == "search_products" and c["args"].get("max_price")
                   and c["args"]["max_price"] <= int(value)]
        if not limited:
            return f"{value} so'm chegarali qidiruv qilinmadi"
        found = [p["id"] for result in limited for p in result]
        return f"{value} so'mgacha mahsulot topilmasligi kerak edi: {found}" if found else None
    if name == "over_budget_said":
        # If the answer offers products above the customer's budget, it must SAY they cost
        # more — otherwise "100 000 so'mga yo'q... 4 600 000 so'm" can read like an offer.
        over = {a for a in amounts_in(run["answer"]) if a > int(value)} - POLICY_AMOUNTS
        if over and not any(word in normalize(run["answer"]) for word in OVER_BUDGET_WORDS):
            return f"byudjetdan qimmat narxlar {sorted(over)} aytildi, lekin qimmatligi aytilmadi"
        return None
    if name == "search_empty":
        return None if not products else f"mahsulot topilmasligi kerak edi: {[p['id'] for p in products]}"
    if name == "search_max_price":
        bad = [p["id"] for p in products if p["price_uzs"] > int(value)]
        return f"narxi {value} dan qimmat: {bad}" if bad else None
    if name == "search_category":
        bad = [p["id"] for p in products if p["category"] != value]
        return f"boshqa toifa: {bad}" if bad else None
    if name == "search_in_stock":
        bad = [p["id"] for p in products if p["stock"] <= 0]
        return f"omborda yo'q mahsulot: {bad}" if bad else None
    if name == "prices_from_db":
        text = run["answer"] + "\n" + (run["summary"] or "")
        bad = sorted(amounts_in(text) - run["allowed_amounts"])
        return f"bazada yo'q narx(lar): {bad}" if bad else None
    if name == "cart_has":
        ids = [row[1] for row in run["cart"] if row[0] == case["customer_id"]]
        return None if value in ids else f"savatda {value} yo'q: {ids}"
    if name == "latin_answer":
        return "javobda kirill harflari bor" if re.search(r"[Ѐ-ӿ]", run["answer"]) else None
    return f"noma'lum check: {check}"


def evaluate(case: dict, run: dict) -> list[str]:
    problems = []
    # A list means "any of these" (e.g. a missing-info question may go to policy or handoff).
    if run["intent"] not in as_list(case["expect_intent"]):
        problems.append(f"intent: kutilgan {case['expect_intent']}, keldi {run['intent']}")
    # For confirmations, the customer sees the summary first: judge both texts.
    shown = run["answer"] + "\n" + (run["summary"] or "")
    for expected in case.get("must_contain", []):
        if not contains(shown, expected):
            problems.append(f"javobda yo'q: {expected}")
    for forbidden in case.get("must_not_contain", []):
        if contains(shown, forbidden):
            problems.append(f"javobda bo'lmasligi kerak: {forbidden}")
    # A list means "any of these": the same fact can live in a policy file and in faq.md.
    if case.get("expect_source") and not set(as_list(case["expect_source"])) & set(run["sources"]):
        problems.append(f"manba yo'q: {case['expect_source']} (bor: {run['sources']})")
    if case.get("expect_no_source") and run["sources"]:
        problems.append(f"manba ko'rsatilmasligi kerak edi: {run['sources']}")
    if "expect_confirmation" in case and bool(run["summary"]) != case["expect_confirmation"]:
        problems.append(f"tasdiq so'raldi: {bool(run['summary'])}, kutilgan: {case['expect_confirmation']}")
    if run["db_changed"] and not case.get("allow_db_change"):
        problems.append("baza o'zgardi (o'zgarmasligi kerak edi)")
    for check in case.get("checks", []):
        if problem := run_check(check, case, run):
            problems.append(problem)
    return problems


# ======================= main =======================

def cost_usd(tokens_in: int, tokens_out: int) -> float:
    return tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT


def print_summary(results: list[dict], seconds: float, db_ok: bool) -> None:
    passed = [r for r in results if r["passed"]]
    tokens_in = sum(r["usage"]["in"] for r in results)
    tokens_out = sum(r["usage"]["out"] for r in results)
    print("\n" + "=" * 70)
    print(f"Natija: {len(passed)}/{len(results)} o'tdi ({100 * len(passed) / len(results):.0f}%)")
    by_category: dict[str, list[bool]] = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["passed"])
    for category, oks in by_category.items():
        print(f"  {category:<20} {sum(oks)}/{len(oks)}")
    def spaced(n: int) -> str:
        return f"{n:,}".replace(",", " ")
    print(f"Tokenlar: kirish {spaced(tokens_in)}, chiqish {spaced(tokens_out)}, "
          f"jami {spaced(tokens_in + tokens_out)}")
    print(f"Taxminiy narx: ${cost_usd(tokens_in, tokens_out):.4f} "
          f"(1M token: kirish ${PRICE_IN}, chiqish ${PRICE_OUT} — .env dan, taxminiy)")
    print(f"Vaqt: {seconds / 60:.1f} daqiqa")
    print(f"data/shop.db o'zgarmadi: {'ha' if db_ok else 'YO‘Q — tekshiring!'}")
    print(f"To'liq hisobot: {REPORT_FILE.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TexnoBozor agentini baholash")
    parser.add_argument("--delay", type=float, default=6, help="savollar orasida kutish (soniya)")
    parser.add_argument("--only", help="faqat shu id'lar, vergul bilan")
    parser.add_argument("--backend", choices=["mcp", "local"], default=config.TOOL_BACKEND)
    args = parser.parse_args()

    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    real_db, real_hash = shop.db.DB_PATH, file_hash(shop.db.DB_PATH)
    print(f"{len(cases)} ta savol, tool'lar: {args.backend}, kutish: {args.delay} s\n")

    results, started = [], time.time()
    for i, case in enumerate(cases, 1):
        if i > 1:
            time.sleep(args.delay)   # stay under the Gemini free-tier rate limit
        try:
            run = run_with_retry(case, args.backend)
            problems = evaluate(case, run)
        except Exception as err:   # noqa: BLE001 — one broken case must not stop the run
            run = {"intent": None, "answer": "", "summary": None, "sources": [],
                   "usage": {"in": 0, "out": 0}}
            problems = [f"xato: {type(err).__name__}: {str(err)[:150]}"]
        status = "PASS" if not problems else "FAIL"
        tokens = run["usage"]["in"] + run["usage"]["out"]
        print(f"[{i:2}/{len(cases)}] {status} {case['id']:<26} intent={run['intent']!s:<9} {tokens:>6} tok",
              flush=True)
        for problem in problems:
            print(f"        - {problem}")
        if problems:
            print(f"        javob: {run['answer'][:200]!r}")
        results.append({"id": case["id"], "category": case["category"], "question": case["question"],
                        "passed": not problems, "problems": problems, "intent": run["intent"],
                        "answer": run["answer"], "summary": run["summary"],
                        "sources": run["sources"], "usage": run["usage"]})

    db_ok = file_hash(real_db) == real_hash
    REPORT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print_summary(results, time.time() - started, db_ok)
    return 0 if all(r["passed"] for r in results) and db_ok else 1


if __name__ == "__main__":
    sys.exit(main())
