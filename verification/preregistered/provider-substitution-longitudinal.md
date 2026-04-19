# provider-substitution-longitudinal

Question: Are provider model substitutions stable or noisy over time?

Null hypothesis: For each provider/requested model, actual_model is consistent on >= 90% of days.

Metrics:
- requested_actual_matrix
- substitution_churn_rate
- output_digest_stability

Falseability condition: If consistency < 90%, substitution is dynamic rather than fixed aliasing.

Kill-switch: If requested/actual model fields are absent, abort the longitudinal claim.

Control arms:
- temperature_0_same_prompt
- provider_served_actual_model
