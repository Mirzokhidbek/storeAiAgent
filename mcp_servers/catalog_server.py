"""MCP server with the catalog tools (SPEC §11). Transport: stdio.

Only a thin wrapper: every tool calls shop/catalog.py, the same logic the Phase 3
@tool version used. The agent starts this file as a subprocess and talks to it over
stdin/stdout (see agent/mcp_client.py).

Run by hand (it waits for MCP messages on stdin):  python -m mcp_servers.catalog_server
"""

from mcp.server.fastmcp import FastMCP

from shop import catalog

# WARNING: hide the per-request INFO lines (they go to stderr, but flood the terminal).
mcp = FastMCP("texnobozor-catalog", log_level="WARNING")


@mcp.tool()
def search_products(query: str, category: str | None = None, min_price: int | None = None,
                    max_price: int | None = None, brand: str | None = None,
                    in_stock_only: bool = True, limit: int = 3) -> dict:
    """Find products for the customer's request, best match first.
    Prices and stock come from the store database and are exact."""
    return catalog.search_products(query, category, min_price, max_price, brand,
                                   in_stock_only, limit)


@mcp.tool()
def get_product(product_id: str) -> dict:
    """Full details of one product (price, stock, weight, specs) by its code, e.g. "LAP-001"."""
    return catalog.get_product(product_id)


@mcp.tool()
def compare_products(product_ids: list[str]) -> dict:
    """Compare 2 or 3 products side by side by their codes."""
    return catalog.compare_products(product_ids)


if __name__ == "__main__":
    # stdout carries the MCP protocol, so nothing else may print to it.
    mcp.run(transport="stdio")
