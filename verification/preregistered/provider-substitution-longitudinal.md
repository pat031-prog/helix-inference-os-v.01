# provider-substitution-longitudinal

Question: Are provider model substitutions stable or noisy over time?

Null hypothesis: For each provider/requested model, provider_returned_model is consistent on >= 90% of days.

Metrics:
- requested_provider_returned_matrix
- substitution_churn_rate
- output_digest_stability

Falseability condition: If consistency < 90%, provider-returned model mapping is dynamic rather than fixed aliasing.

Kill-switch: If requested/provider_returned model fields are absent, abort the longitudinal claim.

Control arms:
- temperature_0_same_prompt
- provider_served_actual_model
