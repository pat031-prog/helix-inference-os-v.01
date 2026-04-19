# blinded-self-archaeology-ab

Question: Do models infer ledger structure independently of visible agent_id labels?

Null hypothesis: Blinded provider_self_mention_rate stays below 5%, while blinded structural_inference_score remains at least 80% of the revealed arm.

Metrics:
- provider_self_mention_rate
- structural_inference_score
- attribution_accuracy

Falseability condition: If blinded structural inference collapses or attribution accuracy is at chance, self-archaeology is label-driven rather than structural.

Kill-switch: If revealed context accidentally leaks provider labels into the blinded arm, abort the test.

Control arms:
- revealed
- blinded
