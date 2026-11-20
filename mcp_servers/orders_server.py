"""MCP server with the order tools (SPEC §11). Transport: stdio.

Only a thin wrapper around shop/orders.py. Here customer_id IS a normal argument:
this is server-side code and its only client is our agent. The agent never shows
these schemas to the LLM; it fills customer_id from the logged-in session itself
(see agent/tools.py), so the LLM still cannot choose whose orders are touched.

place_order / cancel_order are listed here like any tool, but the agent calls them
only from the graph's execute_action node, after interrupt() got "approve".

Run by hand (it waits for MCP messages on stdin):  python -m mcp_servers.orders_server
"""

from mcp.server.fastmcp import FastMCP

from shop import orders

# WARNING: hide the per-request INFO lines (they go to stderr, but flood the terminal).
mcp = FastMCP("texnobozor-orders", log_level="WARNING")


@mcp.tool()
def get_order_status(customer_id: str | None, order_id: str) -> dict:
    """Status and dates of one of this customer's orders (ownership is checked)."""
    return orders.get_order_status(customer_id, order_id)


@mcp.tool()
def list_my_orders(customer_id: str | None) -> dict:
    """This customer's orders, newest first."""
    return orders.list_my_orders(customer_id)


@mcp.tool()
def view_cart(customer_id: str | None) -> dict:
    """This customer's cart: items, quantities, current prices and total."""
    return orders.view_cart(customer_id)


@mcp.tool()
def add_to_cart(customer_id: str | None, product_id: str, quantity: int = 1) -> dict:
    """Add a product to this customer's cart (checks stock). Returns the updated cart."""
    return orders.add_to_cart(customer_id, product_id, quantity)


@mcp.tool()
def order_summary(customer_id: str | None) -> dict:
    """Read-only preview of place_order: items, subtotal, delivery and totals."""
    return orders.order_summary(customer_id)


@mcp.tool()
def cancel_preview(customer_id: str | None, order_id: str) -> dict:
    """Read-only check for cancel_order: the order, or the error cancel_order would give."""
    return orders.cancel_preview(customer_id, order_id)


@mcp.tool()
def place_order(customer_id: str | None) -> dict:
    """Create an order from the cart. HIGH RISK: only after the customer approved."""
    return orders.place_order(customer_id)


@mcp.tool()
def cancel_order(customer_id: str | None, order_id: str) -> dict:
    """Cancel this customer's order if its status allows. HIGH RISK: only after approval."""
    return orders.cancel_order(customer_id, order_id)


if __name__ == "__main__":
    # stdout carries the MCP protocol, so nothing else may print to it.
    mcp.run(transport="stdio")
