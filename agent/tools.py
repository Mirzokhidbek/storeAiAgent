"""LangChain @tool wrappers around shop/catalog.py and shop/orders.py.

The LLM reads each tool's name, docstring and argument schema to decide what to call.

How customer_id stays hidden from the LLM:
  Order tools take a `config: RunnableConfig` parameter. LangChain removes it from the
  schema the LLM sees and fills it in automatically when the tool runs. The app puts the
  logged-in customer there:  tool.invoke(args, config={"configurable": {"customer_id": "C-001"}})
  So the LLM can only pass order_id / product_id — it has no way to pick a customer.
"""

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from shop import catalog, orders


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
    return catalog.search_products(query, category, min_price, max_price, brand,
                                   in_stock_only, limit)


@tool
def get_product(product_id: str) -> dict:
    """Get full details of one product (price, stock, weight, specs) by its code, e.g. "LAP-001"."""
    return catalog.get_product(product_id)


@tool
def compare_products(product_ids: list[str]) -> dict:
    """Compare 2 or 3 products side by side by their codes, e.g. ["LAP-001", "LAP-002"]."""
    return catalog.compare_products(product_ids)


# ---------- order tools: read ----------

@tool
def get_order_status(order_id: str, config: RunnableConfig) -> dict:
    """Get the status and dates of one of the logged-in customer's orders by order number."""
    return orders.get_order_status(_customer_id(config), order_id)


@tool
def list_my_orders(config: RunnableConfig) -> dict:
    """List the logged-in customer's orders, newest first."""
    return orders.list_my_orders(_customer_id(config))


@tool
def view_cart(config: RunnableConfig) -> dict:
    """Show the logged-in customer's cart: items, quantities, current prices and total."""
    return orders.view_cart(_customer_id(config))


# ---------- order tools: write ----------

@tool
def add_to_cart(product_id: str, config: RunnableConfig, quantity: int = 1) -> dict:
    """Add a product to the logged-in customer's cart (checks stock). Returns the updated cart."""
    return orders.add_to_cart(_customer_id(config), product_id, quantity)


@tool
def place_order(config: RunnableConfig) -> dict:
    """Place an order from everything in the cart. Only after the customer has confirmed."""
    return orders.place_order(_customer_id(config))


@tool
def cancel_order(order_id: str, config: RunnableConfig) -> dict:
    """Cancel one of the customer's orders (only 'yangi' or 'tolandi'). Only after confirmation."""
    return orders.cancel_order(_customer_id(config), order_id)


# ---------- groups used by the agent (Phase 4) ----------

CATALOG_TOOLS = [search_products, get_product, compare_products]
ORDER_READ_TOOLS = [get_order_status, list_my_orders, view_cart]
CART_TOOLS = [add_to_cart]
# Never bind these to the LLM directly. They run only from the graph's execute_action
# node, after interrupt() has received the customer's explicit approval.
CONFIRM_REQUIRED_TOOLS = [place_order, cancel_order]
