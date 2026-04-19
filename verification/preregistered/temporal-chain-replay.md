# temporal-chain-replay

Question: Does cryptographic chain verification preserve operational semantics under temporal reorderings?

Null hypothesis: Chains still verify under replay transformations, but agent temporal reasoning degrades under shuffled order.

Metrics:
- chain_still_verifies
- agent_behavior_delta
- temporal_reasoning_quality

Falseability condition: If shuffled and original produce indistinguishable temporal reasoning, semantic order sensitivity is not observed.

Kill-switch: If replay transforms alter node hashes unintentionally, abort the suite.

Control arms:
- original
- shuffled
- compressed
- expanded
