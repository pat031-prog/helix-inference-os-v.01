# hybrid-rerank-ab

Question: Does hybrid rerank reduce contamination, and does receipt adjudication still dominate authenticity?

Null hypothesis: Dense rerank alone is equivalent to receipt adjudication under contaminated retrieval.

Metrics:
- precision_at_1
- fake_memory_contamination_rate
- task_score
- context_overhead_ms

Falseability condition: If receipt_adjudicated contamination exceeds dense rerank contamination, lineage enforcement is not load-bearing.

Kill-switch: If bm25_only retrieves no authentic candidates, abort rerank comparison.

Control arms:
- bm25_only
- bm25_dense_rerank
- receipt_adjudicated
