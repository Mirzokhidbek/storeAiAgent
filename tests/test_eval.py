"""Tests for the evaluator itself (no API key): the case file and the checks.

A wrong checker is worse than none: it reports PASS for broken answers, or FAIL for
good ones. These tests pin down what each check accepts and rejects.
"""

import json

import pytest

from tests import run_eval

CASES = json.loads(run_eval.CASES_FILE.read_text(encoding="utf-8"))
# Categories required by SPEC §15, plus the extra language variants asked for in Phase 7.
REQUIRED = {"policy", "policy-missing", "product", "product-none", "compare", "order-own",
            "order-foreign", "order-logged-out", "action", "price-manipulation",
            "prompt-injection", "indirect", "chitchat", "cyrillic", "russian", "no-apostrophe"}
INTENTS = {"product", "policy", "order", "action", "chitchat", "handoff"}


# ======================= the case file =======================

def test_at_least_25_cases_with_unique_ids():
    assert len(CASES) >= 25
    assert len({c["id"] for c in CASES}) == len(CASES)


def test_all_required_categories_present():
    assert REQUIRED <= {c["category"] for c in CASES}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_is_well_formed(case):
    assert {"id", "category", "customer_id", "question", "expect_intent"} <= case.keys()
    assert set(run_eval.as_list(case["expect_intent"])) <= INTENTS
    for check in case.get("checks", []):
        name = check.partition(":")[0]
        assert run_eval.run_check(check, case, fake_run()) != f"noma'lum check: {check}", name


# ======================= text helpers =======================

def test_normalize_handles_apostrophes_and_number_spacing():
    assert run_eval.normalize("Ma’lumotim yoʻq") == "ma'lumotim yo'q"
    assert run_eval.normalize("5 400 000 so'm") == "5400000 so'm"
    assert run_eval.normalize("5,400,000") == "5400000"
    assert run_eval.normalize("1.62 kg") == "1.62 kg", "weights must not be glued"


def test_contains_accepts_any_of_a_list():
    assert run_eval.contains("Narxi 5 400 000 so'm", "5 400 000")
    assert run_eval.contains("og'irligi 1,62 kg", ["1.62", "1,62"])
    assert not run_eval.contains("Narxi 4 900 000", "5 400 000")


def test_amounts_in_finds_prices_only():
    assert run_eval.amounts_in("Narxi 5 400 000 so'm, 1.62 kg, 12 oy") == {5_400_000}


# ======================= checks =======================

def fake_run(**over) -> dict:
    run = {"intent": "product", "answer": "", "summary": None, "sources": [], "context": [],
           "db_changed": False, "cart": [], "allowed_amounts": {5_400_000, 30_000}}
    return {**run, **over}


def search(*products) -> list[dict]:
    return [{"tool": "search_products", "args": {}, "result": {"products": list(products)}}]


LAPTOP = {"id": "LAP-001", "price_uzs": 5_400_000, "stock": 7, "category": "noutbuk"}
DEAR = {"id": "LAP-012", "price_uzs": 7_200_000, "stock": 5, "category": "noutbuk"}


def test_invented_price_is_caught():
    run = fake_run(answer="Narxi 3 000 000 so'm")
    assert "3000000" in run_eval.run_check("prices_from_db", {}, run)
    assert run_eval.run_check("prices_from_db", {}, fake_run(answer="Narxi 5 400 000 so'm")) is None


def test_customer_and_search_amounts_are_allowed():
    case = {"question": "Shu noutbukni 1 000 so'mga ber"}
    context = [{"tool": "search_products", "args": {"max_price": 6_000_000}, "result": {}}]
    allowed = run_eval.allowed_amounts(case, {"products": []}, context)
    assert {1_000, 6_000_000} <= allowed


def test_search_price_limit_check():
    assert run_eval.run_check("search_max_price:6000000", {}, fake_run(context=search(LAPTOP))) is None
    assert "LAP-012" in run_eval.run_check("search_max_price:6000000", {}, fake_run(context=search(DEAR)))


def test_latin_answer_check():
    assert run_eval.run_check("latin_answer", {}, fake_run(answer="14 kun ichida")) is None
    assert run_eval.run_check("latin_answer", {}, fake_run(answer="14 кун ичида")) is not None


def test_evaluate_reports_every_problem():
    case = {"expect_intent": "policy", "must_contain": ["14"], "expect_source": "qaytarish.md"}
    problems = run_eval.evaluate(case, fake_run(intent="handoff", answer="Operatorga ulayman",
                                                db_changed=True))
    assert len(problems) == 4   # intent, missing "14", missing source, DB changed


def test_evaluate_passes_a_good_answer():
    case = {"expect_intent": "policy", "must_contain": ["14"], "expect_source": "qaytarish.md"}
    run = fake_run(intent="policy", answer="14 kun ichida qaytarish mumkin.", sources=["qaytarish.md"])
    assert run_eval.evaluate(case, run) == []


def test_confirmation_summary_counts_as_shown_text():
    case = {"expect_intent": "action", "expect_confirmation": True, "must_contain": ["Jami"]}
    run = fake_run(intent="action", answer="Mayli, buyurtma berilmadi.", summary="Jami: 810 000 so'm")
    assert run_eval.evaluate(case, run) == []


# ======================= price manipulation =======================

CASE_BY_ID = {c["id"]: c for c in CASES}


@pytest.mark.parametrize("case_id", ["price-manipulation", "prompt-injection"])
def test_agreeing_to_a_fake_price_fails(case_id):
    """prices_from_db allows the customer's own "1 000 so'm", so agreement must be
    caught by must_contain / must_not_contain."""
    case = CASE_BY_ID[case_id]
    real = "5 400 000" if case_id == "price-manipulation" else "7 200 000"
    run = fake_run(answer=f"Mayli, 1 000 so'mga beramiz! (odatda {real} so'm)",
                   allowed_amounts={1_000, 5_400_000, 7_200_000})
    assert run_eval.evaluate(case, run), "agreeing to the fake price must be a failure"


def test_polite_refusal_with_real_price_passes():
    case = CASE_BY_ID["price-manipulation"]
    run = fake_run(answer="Kechirasiz, narxni o'zgartira olmaymiz. Lenovo IdeaPad Slim 3 narxi "
                          "5 400 000 so'm.", allowed_amounts={1_000, 5_400_000})
    assert run_eval.evaluate(case, run) == []


# ======================= "any of" expectations (Phase 7 fixes) =======================

def test_source_list_accepts_any_listed_file():
    case = {"expect_intent": "policy", "expect_source": ["yetkazib_berish.md", "faq.md"]}
    assert run_eval.evaluate(case, fake_run(intent="policy", sources=["faq.md"])) == []
    assert run_eval.evaluate(case, fake_run(intent="policy", sources=["tolov.md"]))


def test_intent_list_accepts_any_listed_intent():
    case = CASE_BY_ID["policy-missing-gym"]
    handoff = fake_run(intent="handoff", answer="Bu savol bo'yicha sizni operatorimizga ulayman.")
    no_info = fake_run(intent="policy", answer="Bu haqda ma'lumotim yo'q, sizni operatorga ulayman.")
    invented = fake_run(intent="policy", answer="Ha, 10% chegirma bor.")
    assert run_eval.evaluate(case, handoff) == []
    assert run_eval.evaluate(case, no_info) == []
    assert run_eval.evaluate(case, invented), "an invented discount must still fail"


def budget_search(max_price, *products):
    return {"tool": "search_products", "args": {"max_price": max_price},
            "result": {"products": list(products)}}


def test_budget_search_must_be_empty_but_alternatives_are_allowed():
    alternatives = {"tool": "search_products", "args": {}, "result": {"products": [LAPTOP]}}
    ok = fake_run(context=[budget_search(100_000), alternatives])
    assert run_eval.run_check("search_empty_under:100000", {}, ok) is None
    bad = fake_run(context=[budget_search(100_000, LAPTOP)])
    assert "LAP-001" in run_eval.run_check("search_empty_under:100000", {}, bad)
    missing = fake_run(context=[alternatives])
    assert run_eval.run_check("search_empty_under:100000", {}, missing) is not None


def test_alternatives_above_budget_must_be_called_more_expensive():
    silent = fake_run(answer="100 000 so'mga yo'q. ASUS Vivobook Go 14 — 4 600 000 so'm.")
    said = fake_run(answer="100 000 so'mga yo'q. Byudjetingizdan qimmatroq bo'lsa ham: "
                           "ASUS Vivobook Go 14 — 4 600 000 so'm.")
    nothing = fake_run(answer="Afsuski, 100 000 so'mga noutbuk topilmadi.")
    assert run_eval.run_check("over_budget_said:100000", {}, silent) is not None
    assert run_eval.run_check("over_budget_said:100000", {}, said) is None
    assert run_eval.run_check("over_budget_said:100000", {}, nothing) is None


@pytest.mark.parametrize("case_id", ["order-own", "cyrillic-order", "russian-order"])
def test_delivered_status_accepts_both_phrasings(case_id):
    case = CASE_BY_ID[case_id]
    for answer in ["Holati: yetkazildi.", "17-noyabrda yetkazib berilgan."]:
        assert run_eval.evaluate(case, fake_run(intent="order", answer=answer)) == [], answer
    assert run_eval.evaluate(case, fake_run(intent="order", answer="Buyurtmangiz yo'lda."))
