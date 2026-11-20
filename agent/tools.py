"""LangChain @tool wrappers: what the LLM sees. The work happens behind a "backend".

The LLM reads each tool's name, docstring and argument schema to decide what to call.
Those schemas are defined HERE, so they stayed exactly the same when the work moved
to MCP servers in Phase 6b.

Backends (config.TOOL_BACKEND):
  "mcp"   (default) every call goes to mcp_servers/*.py over MCP (agent/mcp_client.py)
  "local" calls shop/ directly in this process, like Phase 3 (unit tests use this)
Both run the same shop/ functions, so results are identical.

How customer_id stays hidden from the LLM:
  Order tools take a `config: RunnableConfig` parameter. LangChain removes it from the
  schema the LLM sees and fills it in automatically when the tool runs. The app puts the
  logged-in customer there:  tool.invoke(args, config={"configurable": {"customer_id": "C-001"}})
  So the LLM can only pass order_id / product_id — it has no way to pick a customer.
"""

from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

import config
from shop import catalog, orders


class LocalBackend:
    """Phase 3 path: call shop/ in this process. Same interface as MCPToolbox."""

    MODULES = {"catalog": catalog, "orders": orders}

    def call(self, server: str, tool_name: str, args: dict) -> dict:
        return getattr(self.MODULES[server], tool_name)(**args)


@lru_cache
def backend():
    """Created once per process: the MCP servers start on the first use."""
    if config.TOOL_BACKEND == "local":
        return LocalBackend()
    from agent.mcp_client import MCPToolbox   # imported here: local mode needs no MCP
    return MCPToolbox()


def _catalog(tool_name: str, **args) -> dict:
    return backend().call("catalog", tool_name, args)


def _orders(tool_name: str, config: RunnableConfig | None, **args) -> dict:
    # customer_id is added HERE from the runtime config, never taken from the LLM.
    return backend().call("orders", tool_name, {"customer_id": _customer_id(config), **args})


def _customer_id(config: RunnableConfig) -> str | None:
    """Read the logged-in customer from the runtime config (set by the app, not the LLM)."""
    return (config or {}).get("configurable", {}).get("customer_id")


# ---------- catalog tools ----------

@tool
def search_products(query: str, category: str | None = None, min_price: int | None = None,
                    max_price: int | None = None, brand: str | None = None,
                    in_stock_only: bool = True, limit: int = 3) -> dict:
    """Find products for the customer's request and return them best match first.

    Use for recommendations and "do you have ...?" questions.
    query: what the customer wants in their own words, e.g. "talaba uchun yengil noutbuk".
    category: one of "noutbuk", "telefon", "quloqchin", "aksessuar".
    min_price / max_price: price limits in so'm (e.g. "6 mln gacha" -> max_price=6000000).
    brand: e.g. "Lenovo", "Samsung".
    Prices and stock in the result come from the store database and are exact.
    """
    return _catalog("search_products", query=query, category=category, min_price=min_price,
                    max_price=max_price, brand=brand, in_stock_only=in_stock_only, limit=limit)


@tool
def get_product(product_id: str) -> dict:
    """Get full details of one product (price, stock, weight, specs) by its code, e.g. "LAP-001"."""
    return _catalog("get_product", product_id=product_id)


@tool
def compare_products(product_ids: list[str]) -> dict:
    """Compare 2 or 3 products side by side by their codes, e.g. ["LAP-001", "LAP-002"]."""
    return _catalog("compare_products", product_ids=product_ids)


# ---------- order tools: read ----------

@tool
def get_order_status(order_id: str, config: RunnableConfig) -> dict:
    """Get the status and dates of one of the logged-in customer's orders by order number."""
    return _orders("get_order_status", config, order_id=order_id)


@tool
def list_my_orders(config: RunnableConfig) -> dict:
    """List the logged-in customer's orders, newest first."""
    return _orders("list_my_orders", config)


@tool
def view_cart(config: RunnableConfig) -> dict:
    """Show the logged-in customer's cart: items, quantities, current prices and total."""
    return _orders("view_cart", config)


# ---------- order tools: write ----------

@tool
def add_to_cart(product_id: str, config: RunnableConfig, quantity: int = 1) -> dict:
    """Add a product to the logged-in customer's cart (checks stock). Returns the updated cart."""
    return _orders("add_to_cart", config, product_id=product_id, quantity=quantity)


@tool
def place_order(config: RunnableConfig) -> dict:
    """Place an order from everything in the cart. Only after the customer has confirmed."""
    return _orders("place_order", config)


@tool
def cancel_order(order_id: str, config: RunnableConfig) -> dict:
    """Cancel one of the customer's orders (only 'yangi' or 'tolandi'). Only after confirmation."""
    return _orders("cancel_order", config, order_id=order_id)


# ---------- read-only helpers for the graph (NOT LLM tools) ----------
# prepare_action / execute_action use these to build and re-check the confirmation
# summary. They go through the same backend, so prices come from the tool layer too.

def order_summary(customer_id: str | None) -> dict:
    return backend().call("orders", "order_summary", {"customer_id": customer_id})


def cancel_preview(customer_id: str | None, order_id: str) -> dict:
    return backend().call("orders", "cancel_preview",
                          {"customer_id": customer_id, "order_id": order_id})


# ---------- groups used by the agent (Phase 4) ----------

CATALOG_TOOLS = [search_products, get_product, compare_products]
ORDER_READ_TOOLS = [get_order_status, list_my_orders, view_cart]
CART_TOOLS = [add_to_cart]
# Never bind these to the LLM directly. They run only from the graph's execute_action
# node, after interrupt() has received the customer's explicit approval.
CONFIRM_REQUIRED_TOOLS = [place_order, cancel_order]
