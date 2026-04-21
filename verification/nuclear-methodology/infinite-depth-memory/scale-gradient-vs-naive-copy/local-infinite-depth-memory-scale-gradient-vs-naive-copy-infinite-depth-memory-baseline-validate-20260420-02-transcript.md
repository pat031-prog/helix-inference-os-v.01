# Transcript: scale-gradient-vs-naive-copy

- Run ID: `infinite-depth-memory-baseline-validate-20260420-02`
- Judge requested: `local/deterministic-measurer`
- Judge actual: `local/deterministic-measurer`
- Auditor requested: `local/deterministic-scorer`
- Auditor actual: `local/deterministic-scorer`

## Expected / Ground Truth

```json
{
  "largest_depth": 256,
  "max_bounded_context_ms": 1000.0,
  "max_output_compression_ratio": 0.05,
  "baseline_min_speedup": 0.0
}
```

## Visible Contract

```json
{
  "deterministic_suite": true,
  "case": "scale-gradient-vs-naive-copy",
  "protocol": {
    "null_hypothesis": "Bounded context construction has the same output amplification as naive full-history copy.",
    "alternative_hypothesis": "Bounded context emits a compressed working set and avoids full-history text replay."
  }
}
```

## Judge Output

```json
{
  "classification": "bounded_context_vs_full_history_replay",
  "measurements": [
    {
      "depth": 32,
      "build": {
        "depth": 32,
        "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-baseline-validate-20260420-02-32",
        "session_id": "scale-gradient-vs-naive-copy-32-session",
        "insert_ms": 12.182,
        "full_text_chars": 5421,
        "anchor_index": 25,
        "anchor_memory_id": "mem-scale-gradient-vs-naive-copy-32-000025",
        "leaf_memory_id": "mem-scale-gradient-vs-naive-copy-32-000031",
        "memory_count": 32,
        "catalog_stats": {
          "memory_count": 32,
          "observation_count": 0,
          "link_count": 0,
          "fts_enabled": false,
          "journal_mode": "memory",
          "busy_timeout_ms": 5000,
          "dag_node_count": 32,
          "memory_backend": "in_memory_dag",
          "search_backend": "rust_bm25",
          "rust_index_available": true,
          "rust_index_error": null,
          "rust_index_stats": {
            "backend": "rust_bm25",
            "node_count": 32,
            "term_count": 58,
            "doc_freq_term_count": 58,
            "posting_count": 809,
            "total_doc_len": 830,
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
      "optimized_context_tokens": 61,
      "optimized_context_memory_ids": [
        "mem-scale-gradient-vs-naive-copy-32-000025"
      ],
      "optimized_context_chars": 243,
      "naive_full_copy_chars": 5932,
      "optimized_timing": {
        "repeats": 2,
        "min_ms": 0.1214,
        "median_ms": 0.12415,
        "max_ms": 0.1269,
        "rounded_min_ms_2dp": 0.12,
        "rounded_median_ms_2dp": 0.12,
        "raw_ns": [
          126900,
          121400
        ]
      },
      "naive_copy_timing": {
        "repeats": 2,
        "min_ms": 0.0128,
        "median_ms": 0.01305,
        "max_ms": 0.0133,
        "rounded_min_ms_2dp": 0.01,
        "rounded_median_ms_2dp": 0.01,
        "raw_ns": [
          13300,
          12800
        ]
      },
      "output_compression_ratio": 0.04096426
    },
    {
      "depth": 128,
      "build": {
        "depth": 128,
        "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-baseline-validate-20260420-02-128",
        "session_id": "scale-gradient-vs-naive-copy-128-session",
        "insert_ms": 45.7087,
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
        "repeats": 2,
        "min_ms": 0.212,
        "median_ms": 0.2133,
        "max_ms": 0.2146,
        "rounded_min_ms_2dp": 0.21,
        "rounded_median_ms_2dp": 0.21,
        "raw_ns": [
          212000,
          214600
        ]
      },
      "naive_copy_timing": {
        "repeats": 2,
        "min_ms": 0.0399,
        "median_ms": 0.04355,
        "max_ms": 0.0472,
        "rounded_min_ms_2dp": 0.04,
        "rounded_median_ms_2dp": 0.04,
        "raw_ns": [
          47200,
          39900
        ]
      },
      "output_compression_ratio": 0.01022204
    },
    {
      "depth": 256,
      "build": {
        "depth": 256,
        "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-baseline-validate-20260420-02-256",
        "session_id": "scale-gradient-vs-naive-copy-256-session",
        "insert_ms": 103.8506,
        "full_text_chars": 43967,
        "anchor_index": 249,
        "anchor_memory_id": "mem-scale-gradient-vs-naive-copy-256-000249",
        "leaf_memory_id": "mem-scale-gradient-vs-naive-copy-256-000255",
        "memory_count": 256,
        "catalog_stats": {
          "memory_count": 256,
          "observation_count": 0,
          "link_count": 0,
          "fts_enabled": false,
          "journal_mode": "memory",
          "busy_timeout_ms": 5000,
          "dag_node_count": 256,
          "memory_backend": "in_memory_dag",
          "search_backend": "rust_bm25",
          "rust_index_available": true,
          "rust_index_error": null,
          "rust_index_stats": {
            "backend": "rust_bm25",
            "node_count": 256,
            "term_count": 282,
            "doc_freq_term_count": 282,
            "posting_count": 6633,
            "total_doc_len": 6878,
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
        "mem-scale-gradient-vs-naive-copy-256-000249"
      ],
      "optimized_context_chars": 244,
      "naive_full_copy_chars": 48062,
      "optimized_timing": {
        "repeats": 2,
        "min_ms": 0.1295,
        "median_ms": 0.1333,
        "max_ms": 0.1371,
        "rounded_min_ms_2dp": 0.13,
        "rounded_median_ms_2dp": 0.13,
        "raw_ns": [
          137100,
          129500
        ]
      },
      "naive_copy_timing": {
        "repeats": 2,
        "min_ms": 0.0761,
        "median_ms": 0.0778,
        "max_ms": 0.0795,
        "rounded_min_ms_2dp": 0.08,
        "rounded_median_ms_2dp": 0.08,
        "raw_ns": [
          79500,
          76100
        ]
      },
      "output_compression_ratio": 0.00507678
    }
  ],
  "depth_ratio": 8.0,
  "optimized_latency_ratio": 1.073701,
  "speedup_vs_naive_at_largest_depth": 0.583646,
  "measured_latency_ms": 0.1333
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
