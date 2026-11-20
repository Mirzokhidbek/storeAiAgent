"""Order logic: order status, cart, place and cancel orders.

Every function takes `customer_id` as a normal argument. That is fine here: this is
server-side code. The LLM never sees this argument — agent/tools.py fills it in from
the session (Phase 3) and the MCP wrapper will do the same (Phase 6).

Every function returns a JSON-serialisable dict; errors are {"error": "<Uzbek text>"}.
"""

import sqlite3
from datetime import date

from shop.db import connect

CANCELLABLE_STATUSES = ("yangi", "tolandi")

# Delivery prices from data/policies/yetkazib_berish.md (SPEC §5). Code computes them,
# so the confirmation summary never depends on the LLM reading a document correctly.
TASHKENT_FREE_FROM_UZS = 1_000_000
TASHKENT_DELIVERY_UZS = 30_000
REGION_DELIVERY_UZS = 50_000
LOGIN_REQUIRED = {"error": "Buyurtmalar bilan ishlash uchun avval tizimga kiring (chap paneldagi telefon raqam orqali)."}
# Same message for "does not exist" and "belongs to someone else":
# otherwise a user could probe which order numbers exist.
ORDER_NOT_FOUND = {"error": "Sizning hisobingizda bunday raqamli buyurtma topilmadi."}


# ---------- helpers ----------

def _find_own_order(conn: sqlite3.Connection, customer_id: str, order_id: str) -> sqlite3.Row | None:
    """Ownership check lives in the SQL itself: WHERE id = ? AND customer_id = ?"""
    return conn.execute(
        "SELECT * FROM orders WHERE id = ? AND customer_id = ?",
        (order_id.strip(), customer_id),
    ).fetchone()


def _order_items(conn: sqlite3.Connection, order_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT oi.product_id, p.name, oi.quantity, oi.price_uzs
           FROM order_items oi JOIN products p ON p.id = oi.product_id
           WHERE oi.order_id = ?""",
        (order_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _cart(conn: sqlite3.Connection, customer_id: str) -> dict:
    """Cart with CURRENT prices from the products table (prices may change)."""
    rows = conn.execute(
        """SELECT c.product_id, p.name, c.quantity, p.price_uzs, p.stock
           FROM cart_items c JOIN products p ON p.id = c.product_id
           WHERE c.customer_id = ? ORDER BY p.name""",
        (customer_id,),
    ).fetchall()
    items = [
        {"product_id": r["product_id"], "name": r["name"], "quantity": r["quantity"],
         "price_uzs": r["price_uzs"], "subtotal_uzs": r["price_uzs"] * r["quantity"],
         "stock": r["stock"]}
        for r in rows
    ]
    return {"items": items, "total_uzs": sum(i["subtotal_uzs"] for i in items)}


def _next_order_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT MAX(CAST(id AS INTEGER)) AS max_id FROM orders").fetchone()
    return str((row["max_id"] or 1000) + 1)


# ---------- read ----------

def get_order_status(customer_id: str | None, order_id: str) -> dict:
    if not customer_id:
        return LOGIN_REQUIRED
    with connect() as conn:
        order = _find_own_order(conn, customer_id, order_id)
        if order is None:
            return ORDER_NOT_FOUND
        return {
            "order_id": order["id"],
            "status": order["status"],
            "created_at": order["created_at"],
            "delivered_at": order["delivered_at"],
            "items": _order_items(conn, order["id"]),
        }


def list_my_orders(customer_id: str | None) -> dict:
    if not customer_id:
        return LOGIN_REQUIRED
    with connect() as conn:
        rows = conn.execute(
            """SELECT id, status, created_at, delivered_at FROM orders
               WHERE customer_id = ? ORDER BY created_at DESC, CAST(id AS INTEGER) DESC""",
            (customer_id,),
        ).fetchall()
    return {"orders": [dict(r) for r in rows]}


def view_cart(customer_id: str | None) -> dict:
    if not customer_id:
        return LOGIN_REQUIRED
    with connect() as conn:
        return _cart(conn, customer_id)


# ---------- write ----------

def delivery_fees(subtotal_uzs: int) -> dict:
    """Delivery price for Tashkent and for the regions (customers have no address yet)."""
    tashkent = 0 if subtotal_uzs >= TASHKENT_FREE_FROM_UZS else TASHKENT_DELIVERY_UZS
    return {"tashkent_uzs": tashkent, "region_uzs": REGION_DELIVERY_UZS}


def order_summary(customer_id: str | None) -> dict:
    """What place_order would create right now: cart items, subtotal, delivery, totals.
    Read-only. The agent shows this to the customer before asking for confirmation."""
    if not customer_id:
        return LOGIN_REQUIRED
    with connect() as conn:
        cart = _cart(conn, customer_id)
    if not cart["items"]:
        return {"error": "Savatingiz bo'sh. Avval mahsulot qo'shing."}
    short = [i["name"] for i in cart["items"] if i["quantity"] > i["stock"]]
    if short:
        return {"error": f"Zaxirada yetarli emas: {', '.join(short)}. Savatni yangilang."}

    fees = delivery_fees(cart["total_uzs"])
    return {"items": cart["items"], "subtotal_uzs": cart["total_uzs"], "delivery": fees,
            "total_tashkent_uzs": cart["total_uzs"] + fees["tashkent_uzs"],
            "total_region_uzs": cart["total_uzs"] + fees["region_uzs"]}


def add_to_cart(customer_id: str | None, product_id: str, quantity: int = 1) -> dict:
    if not customer_id:
        return LOGIN_REQUIRED
    if quantity < 1:
        return {"error": "Miqdor kamida 1 bo'lishi kerak."}

    product_id = product_id.strip().upper()
    with connect() as conn:
        product = conn.execute(
            "SELECT name, stock FROM products WHERE id = ?", (product_id,)
        ).fetchone()
        if product is None:
            return {"error": f"'{product_id}' kodli mahsulot topilmadi."}

        in_cart = conn.execute(
            "SELECT quantity FROM cart_items WHERE customer_id = ? AND product_id = ?",
            (customer_id, product_id),
        ).fetchone()
        new_quantity = quantity + (in_cart["quantity"] if in_cart else 0)
        if new_quantity > product["stock"]:
            return {"error": f"Kechirasiz, '{product['name']}' zaxirada yetarli emas "
                             f"(mavjud: {product['stock']} ta)."}

        # UPSERT: insert a new row, or update quantity if the product is already in the cart.
        conn.execute(
            """INSERT INTO cart_items (customer_id, product_id, quantity) VALUES (?, ?, ?)
               ON CONFLICT (customer_id, product_id) DO UPDATE SET quantity = excluded.quantity""",
            (customer_id, product_id, new_quantity),
        )
        return _cart(conn, customer_id)


def place_order(customer_id: str | None) -> dict:
    """Create an order from the cart. HIGH RISK: the agent must get user confirmation first."""
    if not customer_id:
        return LOGIN_REQUIRED

    # One transaction: if anything fails midway, stock, order and cart all stay unchanged.
    with connect() as conn:
        cart = _cart(conn, customer_id)
        if not cart["items"]:
            return {"error": "Savatingiz bo'sh. Avval mahsulot qo'shing."}

        short = [i["name"] for i in cart["items"] if i["quantity"] > i["stock"]]
        if short:
            return {"error": f"Zaxirada yetarli emas: {', '.join(short)}. Savatni yangilang."}

        order_id = _next_order_id(conn)
        conn.execute(
            "INSERT INTO orders (id, customer_id, status, created_at) VALUES (?, ?, 'yangi', ?)",
            (order_id, customer_id, date.today().isoformat()),
        )
        for item in cart["items"]:
            # Price is copied into order_items, so later price changes do not affect this order.
            conn.execute(
                "INSERT INTO order_items VALUES (?, ?, ?, ?)",
                (order_id, item["product_id"], item["quantity"], item["price_uzs"]),
            )
            conn.execute(
                "UPDATE products SET stock = stock - ? WHERE id = ?",
                (item["quantity"], item["product_id"]),
            )
        conn.execute("DELETE FROM cart_items WHERE customer_id = ?", (customer_id,))

    return {"order_id": order_id, "status": "yangi", "items": cart["items"],
            "total_uzs": cart["total_uzs"]}


def _not_cancellable(status: str) -> dict:
    return {"error": f"Bu buyurtmani bekor qilib bo'lmaydi: holati '{status}'. "
                     "Faqat 'yangi' yoki 'tolandi' holatidagi buyurtmalar bekor qilinadi."}


def cancel_preview(customer_id: str | None, order_id: str) -> dict:
    """Read-only: the order cancel_order would cancel, or the error it would return.
    Lets the agent refuse a foreign/delivered order BEFORE asking for confirmation."""
    order = get_order_status(customer_id, order_id)
    if "error" not in order and order["status"] not in CANCELLABLE_STATUSES:
        return _not_cancellable(order["status"])
    return order


def cancel_order(customer_id: str | None, order_id: str) -> dict:
    """Cancel own order if status allows. HIGH RISK: the agent must get user confirmation first."""
    if not customer_id:
        return LOGIN_REQUIRED

    with connect() as conn:
        order = _find_own_order(conn, customer_id, order_id)
        if order is None:
            return ORDER_NOT_FOUND
        if order["status"] not in CANCELLABLE_STATUSES:
            return _not_cancellable(order["status"])

        conn.execute("UPDATE orders SET status = 'bekor_qilindi' WHERE id = ?", (order["id"],))
        # Return the items to stock, mirroring what place_order took away.
        for item in _order_items(conn, order["id"]):
            conn.execute(
                "UPDATE products SET stock = stock + ? WHERE id = ?",
                (item["quantity"], item["product_id"]),
            )

    return {"order_id": order["id"], "status": "bekor_qilindi"}
