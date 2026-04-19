# prompt-selection-stress-test

Question: Do phase-transition metaphors persist across prompt templates or come from unconstrained prompt selection?

Null hypothesis: Inter-model metaphor signature collapses under the mechanical prompt to less than 20% of the unconstrained gap.

Metrics:
- metaphor_density
- phase_keyword_frequency
- inter_model_signature_slope

Falseability condition: If the mechanical prompt preserves the inter-model signature, the phenomenon survives prompt-selection controls.

Kill-switch: If prompt templates do not share the exact same ledger digest, abort the comparison.

Control arms:
- mechanical
- neutral
- exploratory
- unconstrained
