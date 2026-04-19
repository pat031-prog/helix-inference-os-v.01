# retrieval-adversarial-robustness

Question: How much plausible poisoned memory can the retrieval layer tolerate before task performance degrades below baseline quality?

Null hypothesis: Task score remains at least 80% of baseline through 25% adversarial noise.

Metrics:
- precision_at_k
- task_score_degradation
- noise_tolerance_threshold

Falseability condition: If task score drops below 80% before 25% noise, publish the lower tolerance threshold.

Kill-switch: If fake memories are not retrieved in the high-density arm, the adversarial injection failed and the test is invalid.

Control arms:
- noise_0
- noise_10
- noise_25
- noise_50
- noise_75
