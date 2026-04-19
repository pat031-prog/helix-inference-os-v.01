# lineage-forgery-gauntlet

Question: Does lineage forgery detection survive adversaries who know the schema?

Null hypothesis: HeliX detects signature-aware forgeries with recall >= 0.95 and FPR <= 0.01.

Metrics:
- forgery_detection_rate
- precision
- recall
- f1
- false_positive_rate
- detection_latency_ms

Falseability condition: If signature-aware recall < 0.95 or FPR > 0.01, publish the bound as a failure.

Kill-switch: If naive forgery escapes, abort downstream forgery claims.

Control arms:
- legitimate
- naive
- schema-aware
- hash-aware
- signature-aware
