"""Customer lookup for the UI login (no passwords: real authentication is a non-goal, SPEC §1)."""

import re

from shop.db import connect


def normalize_phone(phone: str) -> str:
    """'90 123 45 67', '998901234567', '+998 (90) 123-45-67' -> '+998901234567'."""
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 9:          # local number without the country code
        digits = "998" + digits
    return "+" + digits


def find_by_phone(phone: str) -> dict | None:
    """Return {"id", "name", "phone"} or None if no customer has this number."""
    with connect() as conn:
        row = conn.execute("SELECT id, name, phone FROM customers WHERE phone = ?",
                           (normalize_phone(phone),)).fetchone()
    return dict(row) if row else None
