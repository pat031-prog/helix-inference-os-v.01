from helix_kv import (
    AdaptiveKVPolicy,
    KVConfig,
    TransformersCompressedKVCache,
    build_adaptive_config,
    build_asymmetric_config,
    build_transformers_variant_set,
    run_transformers_kv_benchmark,
)


def test_public_api_exposes_adaptive_and_asymmetric_builders() -> None:
    adaptive = build_adaptive_config(calibration_tokens=32, hot_window=4, topk=8)
    asymmetric = build_asymmetric_config(key_mode="turbo-4bit", value_mode="turbo-int8-hadamard")

    assert isinstance(adaptive, KVConfig)
    assert adaptive.mode == "adaptive"
    assert isinstance(asymmetric, KVConfig)
    assert asymmetric.key_mode == "turbo-4bit"
    assert asymmetric.value_mode == "turbo-int8-hadamard"


def test_adaptive_policy_mode_histogram_is_publicly_usable() -> None:
    policy = AdaptiveKVPolicy()
    policy.record_mode("turbo-4bit")
    policy.record_mode("turbo-int8-hadamard")
    policy.record_mode("turbo-int8-hadamard")

    assert policy.mode_histogram() == {"turbo-4bit": 1, "turbo-int8-hadamard": 2}


def test_public_api_exposes_transformers_benchmark_symbols() -> None:
    assert TransformersCompressedKVCache is not None
    assert callable(build_transformers_variant_set)
    assert callable(run_transformers_kv_benchmark)
    assert [variant["name"] for variant in build_transformers_variant_set("community")][-1] == "helix-optimal"
