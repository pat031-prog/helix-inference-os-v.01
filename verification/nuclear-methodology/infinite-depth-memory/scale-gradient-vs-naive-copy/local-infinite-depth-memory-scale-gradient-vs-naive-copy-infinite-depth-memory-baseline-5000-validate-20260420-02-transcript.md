# Transcript: scale-gradient-vs-naive-copy

- Run ID: `infinite-depth-memory-baseline-5000-validate-20260420-02`
- Judge requested: `local/deterministic-measurer`
- Judge actual: `local/deterministic-measurer`
- Auditor requested: `local/deterministic-scorer`
- Auditor actual: `local/deterministic-scorer`

## Expected / Ground Truth

```json
{
  "largest_depth": 5000,
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
      "depth": 128,
      "build": {
        "depth": 128,
        "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-baseline-5000-validate-20260420-02-128",
        "session_id": "scale-gradient-vs-naive-copy-128-session",
        "insert_ms": 60.3143,
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
        "repeats": 3,
        "min_ms": 0.1557,
        "median_ms": 0.1665,
        "max_ms": 0.1925,
        "rounded_min_ms_2dp": 0.16,
        "rounded_median_ms_2dp": 0.17,
        "raw_ns": [
          155700,
          166500,
          192500
        ]
      },
      "naive_copy_timing": {
        "repeats": 3,
        "min_ms": 0.0599,
        "median_ms": 0.0654,
        "max_ms": 0.0667,
        "rounded_min_ms_2dp": 0.06,
        "rounded_median_ms_2dp": 0.07,
        "raw_ns": [
          65400,
          59900,
          66700
        ]
      },
      "output_compression_ratio": 0.01022204
    },
    {
      "depth": 1024,
      "build": {
        "depth": 1024,
        "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-baseline-5000-validate-20260420-02-1024",
        "session_id": "scale-gradient-vs-naive-copy-1024-session",
        "insert_ms": 470.6264,
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
        "repeats": 3,
        "min_ms": 0.1435,
        "median_ms": 0.1488,
        "max_ms": 0.1654,
        "rounded_min_ms_2dp": 0.14,
        "rounded_median_ms_2dp": 0.15,
        "raw_ns": [
          165400,
          148800,
          143500
        ]
      },
      "naive_copy_timing": {
        "repeats": 3,
        "min_ms": 0.2919,
        "median_ms": 0.2965,
        "max_ms": 0.3016,
        "rounded_min_ms_2dp": 0.29,
        "rounded_median_ms_2dp": 0.3,
        "raw_ns": [
          301600,
          296500,
          291900
        ]
      },
      "output_compression_ratio": 0.00126756
    },
    {
      "depth": 5000,
      "build": {
        "depth": 5000,
        "project": "scale-gradient-vs-naive-copy-infinite-depth-memory-baseline-5000-validate-20260420-02-5000",
        "session_id": "scale-gradient-vs-naive-copy-5000-session",
        "insert_ms": 2511.2191,
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
        "repeats": 3,
        "min_ms": 0.1862,
        "median_ms": 0.1899,
        "max_ms": 0.195,
        "rounded_min_ms_2dp": 0.19,
        "rounded_median_ms_2dp": 0.19,
        "raw_ns": [
          195000,
          189900,
          186200
        ]
      },
      "naive_copy_timing": {
        "repeats": 3,
        "min_ms": 1.7217,
        "median_ms": 1.7488,
        "max_ms": 1.967,
        "rounded_min_ms_2dp": 1.72,
        "rounded_median_ms_2dp": 1.75,
        "raw_ns": [
          1967000,
          1748800,
          1721700
        ]
      },
      "output_compression_ratio": 0.0002561
    }
  ],
  "depth_ratio": 39.0625,
  "optimized_latency_ratio": 1.140541,
  "speedup_vs_naive_at_largest_depth": 9.209057,
  "measured_latency_ms": 0.1899
}
```

## Auditor Output

```json
{
  "verdict": "pass",
  "gate_failures": []
}
```
