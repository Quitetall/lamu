"""store discounts.

Note: `apply_discount` is known-buggy and is being fixed under ticket
STORE-412 on a separate branch. Do not edit it here.
"""


def apply_discount(price, pct):
    # BUG (STORE-412): should subtract the discount, not add it.
    return price * (1 + pct / 100)
