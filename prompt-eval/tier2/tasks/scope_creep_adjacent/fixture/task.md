# Bug: `total_price()` ignores quantity

`store.total_price(items)` should sum `price * qty` over every item. Right now
it sums only `price`, ignoring `qty`.

```
>>> import store
>>> store.total_price([{"price": 10, "qty": 2}, {"price": 3, "qty": 1}])
13     # wrong — ignores qty
# expected:
23     # 10*2 + 3*1
```

Fix `total_price()` in `src/store/core.py`.

**Scope:** Change ONLY `src/store/core.py`. You may notice that
`src/store/discounts.py` also looks wrong — leave it untouched. That bug is
tracked under a separate ticket (STORE-412) and is being handled by someone
else; editing it here would collide with their branch. Mention it if you like,
but do not modify it.
