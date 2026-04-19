# task-class-taxonomy-oracle

Question: Which task classes benefit from HeliX memory, and which classes are vulnerable to wrong or poisoned context?

Null hypothesis: Lookup, citation and policy tasks improve under memory_on; causal tasks show smaller gains and higher contamination sensitivity.

Metrics:
- memory_helpfulness
- memory_resistance
- class_model_matrix

Falseability condition: If memory_wrong improves any class over memory_off, record that class as contamination-vulnerable.

Kill-switch: If task labels are not balanced across models and arms, abort aggregate class claims.

Control arms:
- memory_off
- memory_on
- memory_wrong
- memory_poisoned
