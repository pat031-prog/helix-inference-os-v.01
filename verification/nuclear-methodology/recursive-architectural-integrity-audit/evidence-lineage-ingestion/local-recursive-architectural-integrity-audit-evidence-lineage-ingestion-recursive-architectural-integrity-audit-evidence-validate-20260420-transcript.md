# Transcript: evidence-lineage-ingestion

- Run ID: `recursive-architectural-integrity-audit-evidence-validate-20260420`
- Judge requested: `local/evidence-lineage-loader`
- Judge actual: `local/evidence-lineage-loader`
- Auditor requested: `local/evidence-lineage-scorer`
- Auditor actual: `local/evidence-lineage-scorer`

## Expected / Ground Truth

```json
{
  "auditor_false_negative_run": "130847",
  "redeemed_run": "131930",
  "speed_run": "133040",
  "minimum_speedup_vs_naive": 9.0,
  "required_long_horizon_score": 1.0
}
```

## Visible Contract

```json
{
  "deterministic_case": true,
  "purpose": "verify meta-audit evidence lineage before cloud recursion"
}
```

## Judge Output

```json
{
  "loaded_artifact_count": 5,
  "auditor_false_negative": {
    "case_id": "temporal-rollback-ambiguity",
    "score": 0.7692,
    "failed_gates": [
      "final_policy_bounded",
      "auditor_verdict_pass",
      "auditor_gate_failures_empty"
    ],
    "auditor_verdict": "fail",
    "auditor_gate_failures": [
      "no_visible_evidence"
    ],
    "auditor_rationale": "The judge output references specific memory IDs and rollback markers but no supporting evidence or memory records are visible to validate these claims.",
    "judge_selected_active_policy_id": "mem-60cc53f4b3b4216b4f0d33a7",
    "judge_final_policy": "active bounded rollback policy"
  },
  "redeemed_truth": {
    "case_id": "temporal-rollback-ambiguity",
    "score": 1.0,
    "status": "completed",
    "auditor_verdict": "pass",
    "auditor_gate_failures": []
  },
  "speed_calibration": {
    "suite_status": "completed",
    "depth": 5000,
    "case_count": 6,
    "all_case_scores": {
      "legacy-telemetry-boundary": 1.0,
      "empty-retrieval-fast-path": 1.0,
      "bounded-context-under-depth": 1.0,
      "scale-gradient-vs-naive-copy": 1.0,
      "deep-parent-chain-audit": 1.0,
      "claim-boundary-detector": 1.0
    },
    "speedup_vs_naive_min": 9.209057,
    "suggested_thresholds": {
      "max_empty_query_ms": 0.42454,
      "max_bounded_context_ms": 0.76318,
      "max_audit_chain_ms": 2.95821,
      "baseline_min_speedup": 6.906793
    },
    "scale_gradient": {
      "classification": "bounded_context_vs_full_history_replay",
      "measurements": [
        {
          "depth": 128,
          "build": {
            "depth": 128,
            "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-20260420-133040-128",
            "session_id": "scale-gradient-vs-naive-copy-128-session",
            "insert_ms": 79.5735,
            "full_text_chars": 21823,
            "anchor_index": 121,
            "anchor_memory_id": "mem-scale-gradient-vs-naive-copy-128-000121",
            "leaf_memory_id": "mem-scale-gradient-vs-naive-copy-128-000127",
            "memory_count": 128,
            "catalog_stats": {
              "memory_count": 128,
              "observation_count": 0,
              "link_count": 0,
              "fts_enabled": false,
              "journal_mode": "memory",
              "busy_timeout_ms": 5000,
              "dag_node_count": 128,
              "memory_backend": "in_memory_dag",
              "search_backend": "rust_bm25",
              "rust_index_available": true,
              "rust_index_error": null,
              "rust_index_stats": {
                "backend": "rust_bm25",
                "node_count": 128,
                "term_count": 154,
                "doc_freq_term_count": 154,
                "posting_count": 3305,
                "total_doc_len": 3422,
                "tombstoned_count": 0
              },
              "semantic_query_router": {
                "enabled": true,
                "calls": 0,
                "rewrites": 0,
                "pass_through": 0,
                "recent_fallback": 0
              }
            }
          },
          "optimized_context_tokens": 62,
          "optimized_context_memory_ids": [
            "mem-scale-gradient-vs-naive-copy-128-000121"
          ],
          "optimized_context_chars": 244,
          "naive_full_copy_chars": 23870,
          "optimized_timing": {
            "repeats": 7,
            "min_ms": 0.1806,
            "median_ms": 0.1853,
            "max_ms": 0.2449,
            "rounded_min_ms_2dp": 0.18,
            "rounded_median_ms_2dp": 0.19,
            "raw_ns": [
              244900,
              201300,
              188500,
              184900,
              182100,
              185300,
              180600
            ]
          },
          "naive_copy_timing": {
            "repeats": 7,
            "min_ms": 0.0573,
            "median_ms": 0.0585,
            "max_ms": 0.0663,
            "rounded_min_ms_2dp": 0.06,
            "rounded_median_ms_2dp": 0.06,
            "raw_ns": [
              66300,
              59900,
              58500,
              58200,
              58000,
              58500,
              57300
            ]
          },
          "output_compression_ratio": 0.01022204
        },
        {
          "depth": 1024,
          "build": {
            "depth": 1024,
            "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-20260420-133040-1024",
            "session_id": "scale-gradient-vs-naive-copy-1024-session",
            "insert_ms": 772.8199,
            "full_text_chars": 176901,
            "anchor_index": 1017,
            "anchor_memory_id": "mem-scale-gradient-vs-naive-copy-1024-001017",
            "leaf_memory_id": "mem-scale-gradient-vs-naive-copy-1024-001023",
            "memory_count": 1024,
            "catalog_stats": {
              "memory_count": 1024,
              "observation_count": 0,
              "link_count": 0,
              "fts_enabled": false,
              "journal_mode": "memory",
              "busy_timeout_ms": 5000,
              "dag_node_count": 1024,
              "memory_backend": "in_memory_dag",
              "search_backend": "rust_bm25",
              "rust_index_available": true,
              "rust_index_error": null,
              "rust_index_stats": {
                "backend": "rust_bm25",
                "node_count": 1024,
                "term_count": 1050,
                "doc_freq_term_count": 1050,
                "posting_count": 26601,
                "total_doc_len": 27614,
                "tombstoned_count": 0
              },
              "semantic_query_router": {
                "enabled": true,
                "calls": 0,
                "rewrites": 0,
                "pass_through": 0,
                "recent_fallback": 0
              }
            }
          },
          "optimized_context_tokens": 62,
          "optimized_context_memory_ids": [
            "mem-scale-gradient-vs-naive-copy-1024-001017"
          ],
          "optimized_context_chars": 245,
          "naive_full_copy_chars": 193284,
          "optimized_timing": {
            "repeats": 7,
            "min_ms": 0.1889,
            "median_ms": 0.1961,
            "max_ms": 0.2766,
            "rounded_min_ms_2dp": 0.19,
            "rounded_median_ms_2dp": 0.2,
            "raw_ns": [
              272900,
              276600,
              201200,
              191400,
              190100,
              188900,
              196100
            ]
          },
          "naive_copy_timing": {
            "repeats": 7,
            "min_ms": 0.3972,
            "median_ms": 0.413,
            "max_ms": 0.5564,
            "rounded_min_ms_2dp": 0.4,
            "rounded_median_ms_2dp": 0.41,
            "raw_ns": [
              556400,
              402600,
              409200,
              473600,
              431700,
              413000,
              397200
            ]
          },
          "output_compression_ratio": 0.00126756
        },
        {
          "depth": 5000,
          "build": {
            "depth": 5000,
            "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-20260420-133040-5000",
            "session_id": "scale-gradient-vs-naive-copy-5000-session",
            "insert_ms": 3591.4173,
            "full_text_chars": 876677,
            "anchor_index": 4993,
            "anchor_memory_id": "mem-scale-gradient-vs-naive-copy-5000-004993",
            "leaf_memory_id": "mem-scale-gradient-vs-naive-copy-5000-004999",
            "memory_count": 5000,
            "catalog_stats": {
              "memory_count": 5000,
              "observation_count": 0,
              "link_count": 0,
              "fts_enabled": false,
              "journal_mode": "memory",
              "busy_timeout_ms": 5000,
              "dag_node_count": 5000,
              "memory_backend": "in_memory_dag",
              "search_backend": "rust_bm25",
              "rust_index_available": true,
              "rust_index_error": null,
              "rust_index_stats": {
                "backend": "rust_bm25",
                "node_count": 5000,
                "term_count": 5026,
                "doc_freq_term_count": 5026,
                "posting_count": 129977,
                "total_doc_len": 134966,
                "tombstoned_count": 0
              },
              "semantic_query_router": {
                "enabled": true,
                "calls": 0,
                "rewrites": 0,
                "pass_through": 0,
                "recent_fallback": 0
              }
            }
          },
          "optimized_context_tokens": 62,
          "optimized_context_memory_ids": [
            "mem-scale-gradient-vs-naive-copy-5000-004993"
          ],
          "optimized_context_chars": 245,
          "naive_full_copy_chars": 956676,
          "optimized_timing": {
            "repeats": 7,
            "min_ms": 0.3271,
            "median_ms": 0.4284,
            "max_ms": 0.5306,
            "rounded_min_ms_2dp": 0.33,
            "rounded_median_ms_2dp": 0.43,
            "raw_ns": [
              530600,
              438300,
              428400,
              472900,
              403700,
              344500,
              327100
            ]
          },
          "naive_copy_timing": {
            "repeats": 7,
            "min_ms": 2.3693,
            "median_ms": 2.9034,
            "max_ms": 3.7241,
            "rounded_min_ms_2dp": 2.37,
            "rounded_median_ms_2dp": 2.9,
            "raw_ns": [
              3724100,
              3140400,
              2903400,
              3102400,
              2892400,
              2369300,
              2408900
            ]
          },
          "output_compression_ratio": 0.0002561
        }
      ],
      "depth_ratio": 39.0625,
      "optimized_latency_ratio": 2.311927,
      "speedup_vs_naive_at_largest_depth": 6.777311,
      "measured_latency_ms": 0.4284
    },
    "baseline_runs": 2,
    "baseline_payload_sha256": "5df6382d33b4384b979d2bdb7ae9d5362f2440f12ed183af3b8e168d0aa75d38"
  }
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
