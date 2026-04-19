# zero-day-osint-ground-truth-backtest

Question: Would the OSINT oracle have alerted before known CVEs under a strict temporal cutoff?

Null hypothesis: Median lead time >= 24h, precision >= 0.4 and recall >= 0.3.

Metrics:
- lead_time_hours
- precision
- recall
- false_alert_rate

Falseability condition: If thresholds fail, downgrade product claims instead of hiding failure.

Kill-switch: If cutoff windows include post-CVE evidence, abort the backtest.

Control arms:
- no_oracle
- fixture_cutoff_48h
- oracle_alerts
