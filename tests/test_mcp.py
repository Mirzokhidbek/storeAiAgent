"""Phase 6b: the MCP servers must give exactly the same answers as the in-process tools.

Run:  python -m pytest -v tests/test_mcp.py
No API key needed. Each test starts the two real MCP servers (subprocesses, stdio) on
their own fresh copy of the seed database (SHOP_DB_PATH), and runs the same calls
in-process on another fresh copy. Same seed + same calls -> results must be equal.
(search_products is left out: its semantic step needs Gemini embeddings.)
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

import config
import shop.db
from agent import nodes, tools
from agent.graph import build_graph, pending_confirmation, run_config
from agent.mcp_client import MCPToolbox
from agent.nodes import APPROVE
from tests.conftest import build_seed_db
from tests.test_actions import TABLES, fake  # noqa: F401 — `fake` is a fixture

# The same requests a customer session would make, reads and writes mixed.
CALLS = [
    ("catalog", "get_product", {"product_id": "LAP-001"}),
    ("catalog", "get_product", {"product_id": "nope-1"}),
    ("catalog", "compare_products", {"product_ids": ["LAP-001", "LAP-002"]}),
    ("orders", "get_order_status", {"customer_id": "C-001", "order_id": "1234"}),
    ("orders", "get_order_status", {"customer_id": "C-001", "order_id": "5678"}),  # foreign
    ("orders", "get_order_status", {"customer_id": None, "order_id": "1234"}),     # logged out
    ("orders", "list_my_orders", {"customer_id": "C-001"}),
    ("orders", "view_cart", {"customer_id": "C-001"}),
    ("orders", "add_to_cart", {"customer_id": "C-001", "product_id": "LAP-001", "quantity": 1}),
    ("orders", "order_summary", {"customer_id": "C-001"}),
    ("orders", "place_order", {"customer_id": "C-001"}),
    ("orders", "place_order", {"customer_id": "C-001"}),                          # empty cart
    ("orders", "cancel_preview", {"customer_id": "C-001", "order_id": "1001"}),
    ("orders", "cancel_order", {"customer_id": "C-001", "order_id": "1001"}),
    ("orders", "cancel_order", {"customer_id": "C-001", "order_id": "5678"}),      # foreign
]


@pytest.fixture
def mcp(tmp_path):
    """Real MCP servers working on their own seed database."""
    db_file = build_seed_db(tmp_path / "mcp_shop.db")
    toolbox = MCPToolbox(extra_env={"SHOP_DB_PATH": str(db_file)})
    yield toolbox, db_file
    toolbox.close()


def snapshot(db_file) -> dict:
    conn = shop.db.get_connection(db_file)
    rows = {t: sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}")) for t in TABLES}
    conn.close()
    return rows


def test_mcp_and_local_give_identical_results(mcp, fresh_db):
    toolbox, mcp_db = mcp
    local = tools.LocalBackend()   # works on fresh_db (conftest)

    for server, name, args in CALLS:
        assert toolbox.call(server, name, args) == local.call(server, name, args), name

    # The writes changed both databases in exactly the same way.
    assert snapshot(mcp_db) == snapshot(fresh_db)


def test_llm_sees_no_customer_id_although_server_needs_it(mcp):
    toolbox, _ = mcp
    server_schema = toolbox._tools["orders"]["get_order_status"].args_schema
    assert "customer_id" in server_schema["properties"]          # server side: yes

    for tool in nodes.TOOL_GROUPS["order"] + nodes.TOOL_GROUPS["cart"]:
        llm_schema = tool.tool_call_schema.model_json_schema()
        assert "customer_id" not in llm_schema["properties"], tool.name   # LLM side: no


def test_order_through_graph_and_mcp_needs_approval(mcp, fake, monkeypatch):
    """Whole path: graph -> interrupt -> approve -> execute_action -> MCP -> SQLite."""
    toolbox, mcp_db = mcp
    monkeypatch.setattr(config, "TOOL_BACKEND", "mcp")
    monkeypatch.setattr(tools, "backend", lambda: toolbox)
    agent, cfg = build_graph(), run_config("mcp-order")
    fake["router"].replies = [nodes.RouteDecision(intent="action")]
    fake["action"].replies = [nodes.ActionRequest(action="place_order")]
    before = snapshot(mcp_db)

    agent.invoke({"messages": [HumanMessage("Savatimdagini buyurtma qil")],
                  "customer_id": "C-001"}, cfg)
    assert pending_confirmation(agent, cfg) is not None
    assert snapshot(mcp_db) == before, "nothing changes in the MCP database while paused"

    state = agent.invoke(Command(resume=APPROVE), cfg)

    order_id = state["context"][0]["result"]["order_id"]
    assert any(row[0] == order_id for row in snapshot(mcp_db)["orders"])
    assert snapshot(mcp_db)["cart_items"] == [r for r in before["cart_items"] if r[0] != "C-001"]
