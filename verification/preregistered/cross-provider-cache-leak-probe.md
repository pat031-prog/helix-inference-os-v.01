# cross-provider-cache-leak-probe

Question: When two providers serve the same actual model at temperature zero, are outputs byte-identical across providers?

Null hypothesis: Cross-provider output digests match at least 95% of the time when actual_model and prompt_digest match.

Metrics:
- cross_provider_output_equivalence_rate
- byte_level_difference_count
- actual_model_match_count

Falseability condition: If digests diverge under matched actual_model and prompt_digest, report provider-observable model drift.

Kill-switch: If actual_model is absent from provider responses, abort model-equivalence claims.

Control arms:
- provider_a
- provider_b
- provider_c
