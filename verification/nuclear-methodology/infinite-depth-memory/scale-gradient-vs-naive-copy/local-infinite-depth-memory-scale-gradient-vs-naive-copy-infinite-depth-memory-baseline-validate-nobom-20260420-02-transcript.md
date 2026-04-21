# Transcript: scale-gradient-vs-naive-copy

- Run ID: `infinite-depth-memory-baseline-validate-nobom-20260420-02`
- Judge requested: `local/deterministic-measurer`
- Judge actual: `local/deterministic-measurer`
- Auditor requested: `local/deterministic-scorer`
- Auditor actual: `local/deterministic-scorer`

## Expected / Ground Truth

```json
{
  "largest_depth": 128,
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
      "depth": 16,
      "build": {
        "depth": 16,
        "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-baseline-validate-nobom-20260420-02-16",
        "session_id": "scale-gradient-vs-naive-copy-16-session",
        "insert_ms": 6.7075,
        "full_text_chars": 2703,
        "anchor_index": 9,
        "anchor_memory_id": "mem-scale-gradient-vs-naive-copy-16-000009",
        "leaf_memory_id": "mem-scale-gradient-vs-naive-copy-16-000015",
        "memory_count": 16,
        "catalog_stats": {
          "memory_count": 16,
          "observation_count": 0,
          "link_count": 0,
          "fts_enabled": false,
          "journal_mode": "memory",
          "busy_timeout_ms": 5000,
          "dag_node_count": 16,
          "memory_backend": "in_memory_dag",
          "search_backend": "rust_bm25",
          "rust_index_available": true,
          "rust_index_error": null,
          "rust_index_stats": {
            "backend": "rust_bm25",
            "node_count": 16,
            "term_count": 42,
            "doc_freq_term_count": 42,
            "posting_count": 394,
            "total_doc_len": 400,
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
        "mem-scale-gradient-vs-naive-copy-16-000009"
      ],
      "optimized_context_chars": 242,
      "naive_full_copy_chars": 2958,
      "optimized_timing": {
        "repeats": 1,
        "min_ms": 0.1337,
        "median_ms": 0.1337,
        "max_ms": 0.1337,
        "rounded_min_ms_2dp": 0.13,
        "rounded_median_ms_2dp": 0.13,
        "raw_ns": [
          133700
        ]
      },
      "naive_copy_timing": {
        "repeats": 1,
        "min_ms": 0.0085,
        "median_ms": 0.0085,
        "max_ms": 0.0085,
        "rounded_min_ms_2dp": 0.01,
        "rounded_median_ms_2dp": 0.01,
        "raw_ns": [
          8500
        ]
      },
      "output_compression_ratio": 0.08181204
    },
    {
      "depth": 64,
      "build": {
        "depth": 64,
        "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-baseline-validate-nobom-20260420-02-64",
        "session_id": "scale-gradient-vs-naive-copy-64-session",
        "insert_ms": 22.3544,
        "full_text_chars": 10861,
        "anchor_index": 57,
        "anchor_memory_id": "mem-scale-gradient-vs-naive-copy-64-000057",
        "leaf_memory_id": "mem-scale-gradient-vs-naive-copy-64-000063",
        "memory_count": 64,
        "catalog_stats": {
          "memory_count": 64,
          "observation_count": 0,
          "link_count": 0,
          "fts_enabled": false,
          "journal_mode": "memory",
          "busy_timeout_ms": 5000,
          "dag_node_count": 64,
          "memory_backend": "in_memory_dag",
          "search_backend": "rust_bm25",
          "rust_index_available": true,
          "rust_index_error": null,
          "rust_index_stats": {
            "backend": "rust_bm25",
            "node_count": 64,
            "term_count": 90,
            "doc_freq_term_count": 90,
            "posting_count": 1641,
            "total_doc_len": 1694,
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
        "mem-scale-gradient-vs-naive-copy-64-000057"
      ],
      "optimized_context_chars": 243,
      "naive_full_copy_chars": 11884,
      "optimized_timing": {
        "repeats": 1,
        "min_ms": 0.1338,
        "median_ms": 0.1338,
        "max_ms": 0.1338,
        "rounded_min_ms_2dp": 0.13,
        "rounded_median_ms_2dp": 0.13,
        "raw_ns": [
          133800
        ]
      },
      "naive_copy_timing": {
        "repeats": 1,
        "min_ms": 0.0472,
        "median_ms": 0.0472,
        "max_ms": 0.0472,
        "rounded_min_ms_2dp": 0.05,
        "rounded_median_ms_2dp": 0.05,
        "raw_ns": [
          47200
        ]
      },
      "output_compression_ratio": 0.02044766
    },
    {
      "depth": 128,
      "build": {
        "depth": 128,
        "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-baseline-validate-nobom-20260420-02-128",
        "session_id": "scale-gradient-vs-naive-copy-128-session",
        "insert_ms": 47.559,
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
        "repeats": 1,
        "min_ms": 0.134,
        "median_ms": 0.134,
        "max_ms": 0.134,
        "rounded_min_ms_2dp": 0.13,
        "rounded_median_ms_2dp": 0.13,
        "raw_ns": [
          134000
        ]
      },
      "naive_copy_timing": {
        "repeats": 1,
        "min_ms": 0.0431,
        "median_ms": 0.0431,
        "max_ms": 0.0431,
        "rounded_min_ms_2dp": 0.04,
        "rounded_median_ms_2dp": 0.04,
        "raw_ns": [
          43100
        ]
      },
      "output_compression_ratio": 0.01022204
    }
  ],
  "depth_ratio": 8.0,
  "optimized_latency_ratio": 1.002244,
  "speedup_vs_naive_at_largest_depth": 0.321642,
  "measured_latency_ms": 0.134
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
