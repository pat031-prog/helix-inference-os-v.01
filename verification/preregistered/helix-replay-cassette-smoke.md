# helix-replay-cassette-smoke

Question: Can a sanitized cassette replay deterministically without storing full prompts/outputs?

Null hypothesis: Cassette replay cannot detect deterministic decision drift without full outputs.

Metrics:
- decision_count
- decision_drift_count
- cassette_digest

Falseability condition: If an altered expected decision produces no drift, replay is falsified.

Kill-switch: If cassette stores full prompts or secrets by default, abort replay publication.

Control arms:
- verify-only
- cassette
- diff
