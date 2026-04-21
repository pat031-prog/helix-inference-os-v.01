# hybrid-rerank-live-ab

Question: Does dense rerank reduce contamination, and does receipt adjudication still govern authenticity?

Null hypothesis: Dense rerank and receipt adjudication have equal contamination under poisoned retrieval.

Metrics:
- precision_at_1
- fake_memory_contamination_rate
- context_overhead_ms

Falseability condition: If receipt_adjudicated has higher contamination than dense rerank, lineage adjudication is not load-bearing.

Kill-switch: If authentic signed memory is not retrievable, abort rerank comparison.

Control arms:
- bm25_only
- bm25_dense_rerank
- receipt_adjudicated
