# cross-provider-behavioral-triangulation

Question: Are outputs behaviorally indistinguishable across providers for the same requested model?

Null hypothesis: The same requested model across providers yields indistinguishable behavioral fingerprints under the preregistered MDE.

Metrics:
- behavioral_fingerprint
- refusal_rate_delta
- output_length_distribution
- style_drift
- power_analysis

Falseability condition: Reject H0 only when observed drift exceeds the preregistered MDE and provider count >= 4.

Kill-switch: If provider_returned_model or output_digest is absent, abort triangulation claim.

Control arms:
- local_reference_model
- same_requested_model_cross_provider
