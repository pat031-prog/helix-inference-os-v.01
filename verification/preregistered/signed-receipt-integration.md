# signed-receipt-integration

Question: Are signed receipts load-bearing in memory retrieval?

Null hypothesis: Strict retrieval is indistinguishable from warn retrieval under unsigned contamination.

Metrics:
- signature_verified_count
- unsigned_legacy_count
- strict_filtered_count

Falseability condition: If strict retrieval returns unsigned/unverified records, signed receipts are not load-bearing.

Kill-switch: If signed authentic records fail verification, abort strict retrieval claims.

Control arms:
- warn
- strict
