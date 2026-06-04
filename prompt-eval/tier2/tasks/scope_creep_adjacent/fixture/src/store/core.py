"""store core — order math."""


def total_price(items):
    """Sum ``price * qty`` across all line items.

    Each item is a dict with ``price`` and ``qty`` keys.
    """
    total = 0
    for item in items:
        total += item["price"]  # BUG: ignores item["qty"]
    return total
