# byzantine-shared-hmem

Question: Can shared hmem distinguish valid signed inserts from authentic branch truth under Byzantine writers?

Null hypothesis: A valid signature is sufficient to make a memory authentic branch truth.

Metrics:
- outsider_rejection_rate
- signed_false_insert_quarantine_rate
- authentic_branch_precision
- authentic_branch_recall

Falseability condition: If compromised-key or legitimate-adversarial inserts become authentic branch truth, the signature claim is overbroad.

Kill-switch: If outsider unsigned inserts are accepted, abort signed-hmem claims.

Control arms:
- outsider_invalid
- compromised_key
- legitimate_adversarial_writer
- honest_writer
