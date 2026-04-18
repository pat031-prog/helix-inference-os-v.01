from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from helix_kv.benchmark import (
    build_transformers_variant_set,
    published_benchmark_context,
    run_adaptive_policy_benchmark,
    run_transformers_kv_benchmark,
)
from helix_proto.api import HelixRuntime, serve_api
from helix_proto.assistants import configure_assistant, list_assistants
from helix_proto.format import create_store, store_stats, streaming_matvec, verify_store
from helix_proto.hf import (
    benchmark_gpt2_generation_suite,
    benchmark_gpt2_generation_cache,
    benchmark_gpt2_kv_mode_matrix,
    export_huggingface_model,
    export_local_npz,
    GPT2StreamingEngine,
    gpt2_generate_greedy,
    gpt2_generate_sample,
    gpt2_resume_generation,
    infer_bert_mlm_logits,
    infer_gpt2_causal_lm_logits,
    infer_one_layer_bert_mlm_logits,
    infer_zero_layer_bert_mlm,
    infer_zero_layer_bert_mlm_logits,
)
from helix_proto.model_bench import (
    DEFAULT_MODEL_REFS,
    benchmark_models,
    compare_benchmark_reports,
    load_benchmark_report,
    save_benchmark_comparison,
    save_benchmark_report,
)
from helix_proto.text import save_toy_tokenizer
from helix_proto.tool_call_bench import benchmark_tool_calling, save_tool_call_report
from helix_proto.workspace import slugify, workspace_root


def _json_ready(value):
    if isinstance(value, dict):
        cleaned = dict(value)
        last_logits = cleaned.pop("last_logits", None)
        if isinstance(last_logits, np.ndarray):
            cleaned["last_logits_shape"] = list(last_logits.shape)
        return {key: _json_ready(item) for key, item in cleaned.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _print_json(payload) -> None:
    print(json.dumps(_json_ready(payload), indent=2))


def _default_session_id(alias: str) -> str:
    return f"{slugify(alias)}-{int(time.time())}"


def _prepare_alias_for_model_ref(model_ref: str) -> str:
    path = Path(model_ref)
    if path.suffix.lower() == ".gguf":
        return slugify(path.stem.replace("-q4_k_m", "-q4").replace("_q4_k_m", "-q4"))
    if path.exists():
        return slugify(path.name)
    return slugify(model_ref.split("/")[-1])


def _cmd_convert(args: argparse.Namespace) -> int:
    matrix = np.load(args.input)
    create_store(matrix, args.output, block_rows=args.block_rows, compression_level=args.level)
    stats = store_stats(args.output)
    print(f"created store at {args.output}")
    print(
        f"shape={stats['rows']}x{stats['cols']} blocks={stats['blocks']} "
        f"ratio={stats['compression_ratio']:.2f}x"
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    results = verify_store(args.store)
    print(f"verified {len(results)} blocks")
    return 0


def _cmd_matvec(args: argparse.Namespace) -> int:
    vector = np.load(args.vector)
    result = streaming_matvec(args.store, vector)
    if args.output:
        np.save(args.output, result)
        print(f"saved output to {args.output}")
    else:
        print(result)
    return 0


def _make_demo_matrix(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.15, size=(rows, cols)).astype(np.float32)
    mask = rng.random((rows, cols)) > 0.78
    base[mask] = 0.0
    return base


def _run_benchmark(rows: int, cols: int, block_rows: int, seed: int) -> dict[str, float]:
    matrix = _make_demo_matrix(rows, cols, seed)
    vector = np.random.default_rng(seed + 1).normal(0.0, 0.15, size=(cols,)).astype(np.float32)

    workdir = Path(tempfile.mkdtemp(prefix="helix-proto-"))
    try:
        store_dir = workdir / "matrix.cdna"
        dense_start = time.perf_counter()
        dense = matrix @ vector
        dense_time = time.perf_counter() - dense_start

        create_store(matrix, store_dir, block_rows=block_rows)
        stream_start = time.perf_counter()
        streamed = streaming_matvec(store_dir, vector)
        stream_time = time.perf_counter() - stream_start

        stats = store_stats(store_dir)
        max_abs_err = float(np.max(np.abs(dense - streamed)))
        return {
            "dense_time_s": dense_time,
            "stream_time_s": stream_time,
            "compression_ratio": stats["compression_ratio"],
            "raw_mb": stats["raw_bytes"] / (1024 * 1024),
            "compressed_mb": stats["compressed_bytes"] / (1024 * 1024),
            "max_abs_err": max_abs_err,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    result = _run_benchmark(args.rows, args.cols, args.block_rows, args.seed)
    for key, value in result.items():
        if key.endswith("_s"):
            print(f"{key}={value:.6f}")
        elif key.endswith("_mb"):
            print(f"{key}={value:.2f}")
        else:
            print(f"{key}={value:.6f}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("demo-output").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    matrix = _make_demo_matrix(args.rows, args.cols, args.seed)
    vector = np.random.default_rng(args.seed + 1).normal(0.0, 0.15, size=(args.cols,)).astype(
        np.float32
    )
    np.save(workdir / "matrix.npy", matrix)
    np.save(workdir / "vector.npy", vector)

    create_store(matrix, workdir / "matrix.cdna", block_rows=args.block_rows)
    verify_store(workdir / "matrix.cdna")

    dense = matrix @ vector
    streamed = streaming_matvec(workdir / "matrix.cdna", vector)
    max_abs_err = float(np.max(np.abs(dense - streamed)))
    stats = store_stats(workdir / "matrix.cdna")

    print(f"demo written to {workdir}")
    print(f"compression_ratio={stats['compression_ratio']:.2f}x")
    print(f"max_abs_err={max_abs_err:.8f}")
    return 0


def _cmd_convert_hf(args: argparse.Namespace) -> int:
    manifest = export_huggingface_model(
        args.model_ref,
        args.output,
        block_rows=args.block_rows,
        compression_level=args.level,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"exported {len(manifest['exported'])} tensors to {args.output}")
    print(f"skipped {len(manifest['skipped'])} tensors")
    print(json.dumps({"manifest": str(Path(args.output) / 'manifest.json')}, indent=2))
    return 0


def _cmd_convert_npz(args: argparse.Namespace) -> int:
    manifest = export_local_npz(args.input, args.output, block_rows=args.block_rows)
    print(f"exported {len(manifest['exported'])} tensors to {args.output}")
    print(f"skipped {len(manifest['skipped'])} tensors")
    return 0


def _cmd_build_tiny_bert(args: argparse.Namespace) -> int:
    from transformers import BertConfig, BertForMaskedLM

    config = BertConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=max(1, args.hidden_size // 8),
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
    )
    model = BertForMaskedLM(config)
    model.save_pretrained(args.output)
    print(f"saved tiny bert with {args.num_hidden_layers} layers to {args.output}")
    return 0


def _cmd_build_tiny_gpt2(args: argparse.Namespace) -> int:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=args.vocab_size,
        n_embd=args.hidden_size,
        n_layer=args.num_layers,
        n_head=args.num_heads,
        n_positions=args.max_position_embeddings,
        n_ctx=args.max_position_embeddings,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(args.output)
    tokenizer_info = save_toy_tokenizer(args.output, vocab_size=args.vocab_size)
    print(f"saved tiny gpt2 with {args.num_layers} layers to {args.output}")
    print(f"saved toy tokenizer to {tokenizer_info['tokenizer_dir']}")
    return 0


def _cmd_demo_bert_block(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForMaskedLM

    workdir = Path(args.output).resolve() if args.output else Path("bert-block-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-bert-1layer"
    export_dir = workdir / "export"
    _cmd_build_tiny_bert(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            type_vocab_size=args.type_vocab_size,
            num_hidden_layers=1,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    model = AutoModelForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    input_ids = torch.tensor([args.token_ids], dtype=torch.long)
    token_type_ids = torch.tensor([[0] * len(args.token_ids)], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
    hf_logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_one_layer_bert_mlm_logits(export_dir, token_ids=args.token_ids)
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))
    print(f"demo workspace written to {workdir}")
    print(f"seq_len={len(args.token_ids)}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_top_indices_pos0={np.argsort(our_logits[0])[-args.top_k:][::-1].tolist()}")
    print(f"hf_top_indices_pos0={np.argsort(hf_logits[0])[-args.top_k:][::-1].tolist()}")
    return 0


def _cmd_demo_bert_stack(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForMaskedLM

    workdir = Path(args.output).resolve() if args.output else Path("bert-stack-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-bert-stack"
    export_dir = workdir / "export"
    _cmd_build_tiny_bert(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            type_vocab_size=args.type_vocab_size,
            num_hidden_layers=args.num_hidden_layers,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    model = AutoModelForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    input_ids = torch.tensor([args.token_ids], dtype=torch.long)
    token_type_ids = torch.tensor([[0] * len(args.token_ids)], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)
    hf_logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_bert_mlm_logits(export_dir, token_ids=args.token_ids)
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))
    print(f"demo workspace written to {workdir}")
    print(f"layers={args.num_hidden_layers}")
    print(f"seq_len={len(args.token_ids)}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_top_indices_last={np.argsort(our_logits[-1])[-args.top_k:][::-1].tolist()}")
    print(f"hf_top_indices_last={np.argsort(hf_logits[-1])[-args.top_k:][::-1].tolist()}")
    return 0


def _cmd_demo_hf_infer(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForMaskedLM

    workdir = Path(args.output).resolve() if args.output else Path("hf-infer-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-bert"
    export_dir = workdir / "export"
    _cmd_build_tiny_bert(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            type_vocab_size=args.type_vocab_size,
            num_hidden_layers=0,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    model = AutoModelForMaskedLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor([[args.token_id]], dtype=torch.long),
            token_type_ids=torch.tensor([[args.token_type_id]], dtype=torch.long),
        )
    hf_logits = outputs.logits[0, 0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_zero_layer_bert_mlm_logits(
        export_dir,
        token_id=args.token_id,
        token_type_id=args.token_type_id,
    )
    top = infer_zero_layer_bert_mlm(
        export_dir,
        token_id=args.token_id,
        token_type_id=args.token_type_id,
        top_k=args.top_k,
    )
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))
    print(f"demo workspace written to {workdir}")
    print(f"exported_tensors={len(json.loads((export_dir / 'manifest.json').read_text(encoding='utf-8'))['exported'])}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_top_indices={top['top_indices']}")
    print(f"hf_top_indices={np.argsort(hf_logits)[-args.top_k:][::-1].tolist()}")
    return 0


def _cmd_demo_gpt_causal(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    workdir = Path(args.output).resolve() if args.output else Path("gpt-causal-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    input_ids = torch.tensor([args.token_ids], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    hf_logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_gpt2_causal_lm_logits(export_dir, token_ids=args.token_ids)
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))
    print(f"demo workspace written to {workdir}")
    print(f"layers={args.num_layers}")
    print(f"seq_len={len(args.token_ids)}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_top_indices_last={np.argsort(our_logits[-1])[-args.top_k:][::-1].tolist()}")
    print(f"hf_top_indices_last={np.argsort(hf_logits[-1])[-args.top_k:][::-1].tolist()}")
    return 0


def _cmd_demo_gpt_generate(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    workdir = Path(args.output).resolve() if args.output else Path("gpt-generate-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    ours = gpt2_generate_greedy(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        kv_cache_precision=args.kv_cache_precision,
        kv_quant_seed=args.kv_quant_seed,
        kv_rotation_mode=args.kv_rotation_mode,
        kv_hot_window=args.kv_hot_window,
    )

    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    with torch.no_grad():
        hf_output = model.generate(
            input_ids=torch.tensor([args.prompt_ids], dtype=torch.long),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=1,
        )
    hf_ids = hf_output[0].detach().cpu().tolist()

    print(f"demo workspace written to {workdir}")
    print(f"prompt_ids={args.prompt_ids}")
    print(f"our_generated_ids={ours['generated_ids']}")
    print(f"hf_generated_ids={hf_ids}")
    print(f"cache_lengths={ours['cache_lengths']}")
    print(f"runtime_cache={ours['runtime_cache']}")
    print(f"step_times_ms={ours['step_times_ms']}")
    print(f"avg_step_ms={ours['avg_step_ms']:.3f}")
    print(f"total_time_s={ours['total_time_s']:.6f}")
    print(
        f"rss_mb_before={ours['rss_before_mb']:.2f} "
        f"rss_mb_after={ours['rss_after_mb']:.2f} "
        f"rss_mb_peak={ours['rss_peak_mb']:.2f}"
    )
    print(f"match={ours['generated_ids'] == hf_ids}")
    return 0


def _cmd_demo_gpt_sample(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-sample-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    sampled = gpt2_generate_sample(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
        kv_cache_precision=args.kv_cache_precision,
        kv_key_precision=args.kv_key_precision,
        kv_value_precision=args.kv_value_precision,
        kv_quant_seed=args.kv_quant_seed,
        kv_rotation_mode=args.kv_rotation_mode,
        kv_hot_window=args.kv_hot_window,
        kv_topk=args.kv_topk,
        kv_index_refresh_interval=args.kv_index_refresh_interval,
        kv_block_size=args.kv_block_size,
        kv_layer_share_stride=args.kv_layer_share_stride,
        kv_calibration_tokens=args.kv_calibration_tokens,
    )
    print(f"demo workspace written to {workdir}")
    print(f"prompt_ids={args.prompt_ids}")
    print(f"generated_ids={sampled['generated_ids']}")
    print(f"runtime_cache={sampled['runtime_cache']}")
    print(f"step_times_ms={sampled['step_times_ms']}")
    return 0


def _cmd_demo_gpt_resume(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-resume-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    session_dir = workdir / "session"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    engine = GPT2StreamingEngine(
        export_dir,
        cache_mode="session",
        kv_cache_precision=args.kv_cache_precision,
        kv_key_precision=args.kv_key_precision,
        kv_value_precision=args.kv_value_precision,
        kv_quant_seed=args.kv_quant_seed,
        kv_rotation_mode=args.kv_rotation_mode,
        kv_hot_window=args.kv_hot_window,
        kv_topk=args.kv_topk,
        kv_index_refresh_interval=args.kv_index_refresh_interval,
        kv_block_size=args.kv_block_size,
        kv_layer_share_stride=args.kv_layer_share_stride,
        kv_calibration_tokens=args.kv_calibration_tokens,
    )
    first_part = engine.generate_advanced(
        args.prompt_ids,
        max_new_tokens=args.first_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    engine.save_session(session_dir, generated_ids=first_part["generated_ids"], last_logits=first_part["last_logits"])
    resumed = gpt2_resume_generation(
        export_dir,
        session_dir,
        max_new_tokens=args.second_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode="session",
        kv_cache_precision=args.kv_cache_precision,
        kv_key_precision=args.kv_key_precision,
        kv_value_precision=args.kv_value_precision,
        kv_quant_seed=args.kv_quant_seed,
        kv_rotation_mode=args.kv_rotation_mode,
        kv_hot_window=args.kv_hot_window,
        kv_topk=args.kv_topk,
        kv_index_refresh_interval=args.kv_index_refresh_interval,
        kv_block_size=args.kv_block_size,
        kv_layer_share_stride=args.kv_layer_share_stride,
        kv_calibration_tokens=args.kv_calibration_tokens,
    )

    if args.do_sample:
        full = gpt2_generate_sample(
            export_dir,
            prompt_ids=args.prompt_ids,
            max_new_tokens=args.first_new_tokens + args.second_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            cache_mode="fresh",
            kv_cache_precision=args.kv_cache_precision,
            kv_key_precision=args.kv_key_precision,
            kv_value_precision=args.kv_value_precision,
            kv_quant_seed=args.kv_quant_seed,
            kv_rotation_mode=args.kv_rotation_mode,
            kv_hot_window=args.kv_hot_window,
            kv_topk=args.kv_topk,
            kv_index_refresh_interval=args.kv_index_refresh_interval,
            kv_block_size=args.kv_block_size,
            kv_layer_share_stride=args.kv_layer_share_stride,
            kv_calibration_tokens=args.kv_calibration_tokens,
        )
    else:
        full = gpt2_generate_greedy(
            export_dir,
            prompt_ids=args.prompt_ids,
            max_new_tokens=args.first_new_tokens + args.second_new_tokens,
            cache_mode="fresh",
            kv_cache_precision=args.kv_cache_precision,
            kv_key_precision=args.kv_key_precision,
            kv_value_precision=args.kv_value_precision,
            kv_quant_seed=args.kv_quant_seed,
            kv_rotation_mode=args.kv_rotation_mode,
            kv_hot_window=args.kv_hot_window,
            kv_topk=args.kv_topk,
            kv_index_refresh_interval=args.kv_index_refresh_interval,
            kv_block_size=args.kv_block_size,
            kv_layer_share_stride=args.kv_layer_share_stride,
            kv_calibration_tokens=args.kv_calibration_tokens,
        )

    print(f"demo workspace written to {workdir}")
    print(f"first_part_ids={first_part['generated_ids']}")
    print(f"resumed_ids={resumed['generated_ids']}")
    print(f"full_ids={full['generated_ids']}")
    print(f"match={resumed['generated_ids'] == full['generated_ids']}")
    return 0


def _cmd_demo_gpt_remote(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM

    workdir = Path(args.output).resolve() if args.output else Path("gpt-remote-demo").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    export_dir = workdir / "export"
    export_huggingface_model(
        args.model_ref,
        export_dir,
        block_rows=args.block_rows,
        local_files_only=False,
        trust_remote_code=args.trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_ref,
        local_files_only=False,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    input_ids = torch.tensor([args.prompt_ids], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    hf_logits = outputs.logits[0].detach().cpu().numpy().astype(np.float32)
    our_logits = infer_gpt2_causal_lm_logits(export_dir, token_ids=args.prompt_ids)
    max_abs_err = float(np.max(np.abs(hf_logits - our_logits)))

    ours = gpt2_generate_greedy(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        cache_mode="fresh",
    )
    with torch.no_grad():
        hf_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
    hf_ids = hf_output[0].detach().cpu().tolist()

    print(f"demo workspace written to {workdir}")
    print(f"model_ref={args.model_ref}")
    print(f"max_abs_err={max_abs_err:.8f}")
    print(f"our_generated_ids={ours['generated_ids']}")
    print(f"hf_generated_ids={hf_ids}")
    print(f"match={ours['generated_ids'] == hf_ids}")
    return 0


def _cmd_benchmark_gpt_cache(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-cache-benchmark").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    benchmark = benchmark_gpt2_generation_cache(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        kv_cache_precision=args.kv_cache_precision,
        kv_key_precision=args.kv_key_precision,
        kv_value_precision=args.kv_value_precision,
        kv_quant_seed=args.kv_quant_seed,
        kv_rotation_mode=args.kv_rotation_mode,
        kv_hot_window=args.kv_hot_window,
        kv_topk=args.kv_topk,
        kv_index_refresh_interval=args.kv_index_refresh_interval,
        kv_block_size=args.kv_block_size,
        kv_layer_share_stride=args.kv_layer_share_stride,
        kv_calibration_tokens=args.kv_calibration_tokens,
    )

    print(f"demo workspace written to {workdir}")
    print(f"prompt_ids={benchmark['prompt_ids']}")
    print(f"max_new_tokens={benchmark['max_new_tokens']}")
    print(f"kv_cache_precision={benchmark['kv_cache_precision']}")
    print(f"kv_rotation_mode={benchmark['kv_rotation_mode']}")
    print(f"kv_hot_window={benchmark['kv_hot_window']}")
    print(f"kv_topk={benchmark['kv_topk']}")
    print(f"kv_index_refresh_interval={benchmark['kv_index_refresh_interval']}")
    print(f"kv_block_size={benchmark['kv_block_size']}")
    print(f"kv_layer_share_stride={benchmark['kv_layer_share_stride']}")
    for name, result in benchmark["runs"].items():
        print(
            f"{name}: total_time_s={result['total_time_s']:.6f} "
            f"avg_step_ms={result['avg_step_ms']:.3f} "
            f"kv_cache_bytes={result['kv_cache_bytes']} "
            f"speedup_vs_no_cache={result['speedup_vs_no_cache']:.3f} "
            f"runtime_cache={result['runtime_cache']}"
        )
    return 0


def _cmd_benchmark_gpt_suite(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-suite-benchmark").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    report = benchmark_gpt2_generation_suite(
        export_dir,
        prompt_lengths=args.prompt_lengths,
        max_new_tokens=args.max_new_tokens,
        warm_repeats=args.warm_repeats,
        kv_cache_precision=args.kv_cache_precision,
        kv_key_precision=args.kv_key_precision,
        kv_value_precision=args.kv_value_precision,
        kv_quant_seed=args.kv_quant_seed,
        kv_rotation_mode=args.kv_rotation_mode,
        kv_hot_window=args.kv_hot_window,
        kv_topk=args.kv_topk,
        kv_index_refresh_interval=args.kv_index_refresh_interval,
        kv_block_size=args.kv_block_size,
        kv_layer_share_stride=args.kv_layer_share_stride,
        kv_calibration_tokens=args.kv_calibration_tokens,
    )
    report_path = workdir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"demo workspace written to {workdir}")
    print(f"report={report_path}")
    print(f"kv_cache_precision={report['kv_cache_precision']}")
    print(f"kv_rotation_mode={report['kv_rotation_mode']}")
    print(f"kv_hot_window={report['kv_hot_window']}")
    print(f"kv_topk={report['kv_topk']}")
    print(f"kv_index_refresh_interval={report['kv_index_refresh_interval']}")
    print(f"kv_block_size={report['kv_block_size']}")
    print(f"kv_layer_share_stride={report['kv_layer_share_stride']}")
    for prompt_length in args.prompt_lengths:
        entry = report["suite"][str(prompt_length)]
        no_cache = entry["runs"]["no_cache"]
        fresh = entry["runs"]["fresh_cache"]
        warm = entry["session_warm_avg"]
        print(
            f"prompt_len={prompt_length}: "
            f"no_cache={no_cache['total_time_s']:.6f}s "
            f"fresh_cache={fresh['total_time_s']:.6f}s "
            f"session_warm_avg={warm['total_time_s']:.6f}s "
            f"fresh_kv_bytes={fresh['kv_cache_bytes']}"
        )
    return 0


def _cmd_benchmark_gpt_kv_modes(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-kv-mode-benchmark").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    share_suffix = f"-share{args.kv_layer_share_stride}" if int(args.kv_layer_share_stride) > 1 else ""
    variants = [
        {"name": "fp32", "kv_cache_precision": "fp32", "kv_rotation_mode": "qr", "kv_hot_window": args.kv_hot_window},
        {"name": "turbo-int8-qr", "kv_cache_precision": "turbo-int8", "kv_rotation_mode": "qr", "kv_hot_window": args.kv_hot_window},
        {"name": "turbo-int8-hadamard", "kv_cache_precision": "turbo-int8", "kv_rotation_mode": "hadamard", "kv_hot_window": args.kv_hot_window},
        {"name": "turbo-4bit", "kv_cache_precision": "turbo-4bit", "kv_rotation_mode": "hadamard", "kv_hot_window": args.kv_hot_window},
    ]
    if args.include_qjl:
        variants.append({"name": "turbo-qjl", "kv_cache_precision": "turbo-qjl", "kv_rotation_mode": "hadamard", "kv_hot_window": args.kv_hot_window})
    if args.include_adaptive:
        variants.append(
            {
                "name": "adaptive",
                "kv_cache_precision": "adaptive",
                "kv_rotation_mode": "hadamard",
                "kv_hot_window": args.kv_hot_window,
                "kv_index_refresh_interval": args.kv_index_refresh_interval,
                "kv_block_size": args.kv_block_size,
                "kv_layer_share_stride": args.kv_layer_share_stride,
                "kv_calibration_tokens": args.kv_calibration_tokens,
            }
        )
    if args.include_asymmetric:
        variants.extend(
            [
                {
                    "name": "turbo-4bitk-int8v",
                    "kv_cache_precision": "turbo-int8",
                    "kv_key_precision": "turbo-4bit",
                    "kv_value_precision": "turbo-int8",
                    "kv_rotation_mode": "hadamard",
                    "kv_hot_window": args.kv_hot_window,
                    "kv_index_refresh_interval": args.kv_index_refresh_interval,
                    "kv_block_size": args.kv_block_size,
                    "kv_layer_share_stride": args.kv_layer_share_stride,
                },
                {
                    "name": "turbo-int8k-4bitv",
                    "kv_cache_precision": "turbo-int8",
                    "kv_key_precision": "turbo-int8",
                    "kv_value_precision": "turbo-4bit",
                    "kv_rotation_mode": "hadamard",
                    "kv_hot_window": args.kv_hot_window,
                    "kv_index_refresh_interval": args.kv_index_refresh_interval,
                    "kv_block_size": args.kv_block_size,
                    "kv_layer_share_stride": args.kv_layer_share_stride,
                },
            ]
        )
    if args.kv_topk > 0 and args.kv_hot_window > 0:
        variants.append({"name": f"turbo-int8-topk{args.kv_topk}{share_suffix}", "kv_cache_precision": "turbo-int8", "kv_rotation_mode": "hadamard", "kv_hot_window": args.kv_hot_window, "kv_topk": args.kv_topk, "kv_index_refresh_interval": args.kv_index_refresh_interval, "kv_block_size": args.kv_block_size, "kv_layer_share_stride": args.kv_layer_share_stride})
        variants.append({"name": f"turbo-4bit-topk{args.kv_topk}{share_suffix}", "kv_cache_precision": "turbo-4bit", "kv_rotation_mode": "hadamard", "kv_hot_window": args.kv_hot_window, "kv_topk": args.kv_topk, "kv_index_refresh_interval": args.kv_index_refresh_interval, "kv_block_size": args.kv_block_size, "kv_layer_share_stride": args.kv_layer_share_stride})
        if args.include_qjl:
            variants.append({"name": f"turbo-qjl-topk{args.kv_topk}{share_suffix}", "kv_cache_precision": "turbo-qjl", "kv_rotation_mode": "hadamard", "kv_hot_window": args.kv_hot_window, "kv_topk": args.kv_topk, "kv_index_refresh_interval": args.kv_index_refresh_interval, "kv_block_size": args.kv_block_size, "kv_layer_share_stride": args.kv_layer_share_stride})

    report = benchmark_gpt2_kv_mode_matrix(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        kv_variants=variants,
        kv_quant_seed=args.kv_quant_seed,
    )
    report_path = workdir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"demo workspace written to {workdir}")
    print(f"report={report_path}")
    for name, metrics in report["variants"].items():
        logit_delta = metrics["logit_comparison_vs_baseline"]
        if logit_delta is None:
            print(
                f"{name}: perplexity={metrics['prompt_perplexity']:.6f} "
                f"time_s={metrics['total_time_s']:.6f} kv_cache_bytes={metrics['kv_cache_bytes']}"
            )
            continue
        print(
            f"{name}: perplexity={metrics['prompt_perplexity']:.6f} "
            f"time_s={metrics['total_time_s']:.6f} "
            f"kv_cache_bytes={metrics['kv_cache_bytes']} "
            f"cosine={logit_delta['cosine_similarity']:.6f} "
            f"max_abs_err={logit_delta['max_abs_err']:.6f} "
            f"match={metrics['generated_match_vs_baseline']}"
        )
    return 0


def _cmd_benchmark_gpt_kv_policy(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-kv-policy-benchmark").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=args.max_position_embeddings,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    report = run_adaptive_policy_benchmark(
        export_dir,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        kv_quant_seed=args.kv_quant_seed,
        hot_window=args.kv_hot_window,
        session_root=workdir / "sessions",
    )
    report_path = workdir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"demo workspace written to {workdir}")
    print(f"report={report_path}")
    for name, metrics in report["variants"].items():
        print(
            f"{name}: "
            f"time_s={metrics['total_time_s']:.6f} "
            f"kv_cache_bytes={metrics['kv_cache_bytes']} "
            f"session_total_bytes={metrics['session_total_bytes']} "
            f"switch_count={metrics['switch_count']} "
            f"match={metrics['generated_match_vs_baseline']}"
        )
    return 0


def _directory_size_bytes(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _prompt_ids_for_length(length: int, vocab_size: int) -> list[int]:
    return [int(((index * 7) + 3) % vocab_size) for index in range(length)]


def _variant_engine_kwargs(variant: dict[str, Any], *, kv_quant_seed: int) -> dict[str, Any]:
    kv_key_precision = variant.get("kv_key_precision")
    kv_value_precision = variant.get("kv_value_precision")
    return {
        "kv_cache_precision": str(variant["kv_cache_precision"]),
        "kv_key_precision": str(kv_key_precision) if kv_key_precision is not None else None,
        "kv_value_precision": str(kv_value_precision) if kv_value_precision is not None else None,
        "kv_quant_seed": int(kv_quant_seed),
        "kv_rotation_mode": str(variant.get("kv_rotation_mode", "hadamard")),
        "kv_hot_window": int(variant.get("kv_hot_window", 0)),
        "kv_topk": int(variant.get("kv_topk", 0)),
        "kv_index_refresh_interval": int(variant.get("kv_index_refresh_interval", 8)),
        "kv_block_size": int(variant.get("kv_block_size", 0)),
        "kv_layer_share_stride": int(variant.get("kv_layer_share_stride", 0)),
        "kv_calibration_tokens": int(variant.get("kv_calibration_tokens", 128)),
    }


def _benchmark_gpt_session_size_variants(
    export_dir: Path,
    *,
    prompt_lengths: list[int],
    max_new_tokens: int,
    kv_quant_seed: int,
    kv_variants: list[dict[str, Any]],
    vocab_size: int,
    session_root: Path,
) -> dict[str, Any]:
    if session_root.exists():
        shutil.rmtree(session_root)
    session_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "prompt_lengths": [int(length) for length in prompt_lengths],
        "max_new_tokens": int(max_new_tokens),
        "kv_quant_seed": int(kv_quant_seed),
        "variant_order": [str(variant["name"]) for variant in kv_variants],
        "variants": {},
    }

    for prompt_length in prompt_lengths:
        prompt_ids = _prompt_ids_for_length(int(prompt_length), int(vocab_size))
        per_variant: dict[str, Any] = {}
        baseline_session_bytes: int | None = None
        baseline_kv_bytes: int | None = None
        baseline_npz_bytes: int | None = None

        for variant in kv_variants:
            variant_name = str(variant["name"])
            session_dir = session_root / f"{variant_name}-len{prompt_length}"
            engine = GPT2StreamingEngine(
                export_dir,
                cache_mode="session",
                **_variant_engine_kwargs(variant, kv_quant_seed=kv_quant_seed),
            )
            run = engine.generate(prompt_ids, max_new_tokens=int(max_new_tokens))
            engine.save_session(session_dir, generated_ids=run["generated_ids"], last_logits=run["last_logits"])
            session_meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
            session_total_bytes = _directory_size_bytes(session_dir)
            kv_npz_bytes = int((session_dir / "kv_cache.npz").stat().st_size)
            session_json_bytes = int((session_dir / "session.json").stat().st_size)
            entry = {
                "kv_cache_precision": str(variant["kv_cache_precision"]),
                "kv_key_precision": variant.get("kv_key_precision"),
                "kv_value_precision": variant.get("kv_value_precision"),
                "kv_rotation_mode": str(variant.get("kv_rotation_mode", "hadamard")),
                "kv_hot_window": int(variant.get("kv_hot_window", 0)),
                "kv_topk": int(variant.get("kv_topk", 0)),
                "kv_index_refresh_interval": int(variant.get("kv_index_refresh_interval", 8)),
                "kv_block_size": int(variant.get("kv_block_size", 0)),
                "kv_layer_share_stride": int(variant.get("kv_layer_share_stride", 0)),
                "logical_kv_cache_bytes": int(run["kv_cache_bytes"]),
                "session_total_bytes": session_total_bytes,
                "kv_npz_bytes": kv_npz_bytes,
                "session_json_bytes": session_json_bytes,
                "session_format_version": int(session_meta.get("session_format_version", 1)),
                "kv_session_payload": session_meta.get("kv_session_payload"),
            }
            if baseline_session_bytes is None:
                baseline_session_bytes = session_total_bytes
                baseline_kv_bytes = int(run["kv_cache_bytes"])
                baseline_npz_bytes = kv_npz_bytes
                entry["session_size_ratio_vs_fp32"] = 1.0
                entry["logical_kv_ratio_vs_fp32"] = 1.0
                entry["npz_size_ratio_vs_fp32"] = 1.0
            else:
                entry["session_size_ratio_vs_fp32"] = (
                    float(baseline_session_bytes) / float(session_total_bytes) if session_total_bytes else 0.0
                )
                entry["logical_kv_ratio_vs_fp32"] = (
                    float(baseline_kv_bytes) / float(run["kv_cache_bytes"]) if run["kv_cache_bytes"] else 0.0
                )
                entry["npz_size_ratio_vs_fp32"] = (
                    float(baseline_npz_bytes) / float(kv_npz_bytes) if kv_npz_bytes else 0.0
                )
            per_variant[variant_name] = entry
        report["variants"][str(prompt_length)] = per_variant
    return report


def _build_gpt_landscape_variants(
    *,
    kv_hot_window: int,
    kv_topk: int,
    refresh_no_cache: int,
    refresh_with_cache: int,
    kv_block_size: int,
    kv_layer_share_stride: int,
    kv_calibration_tokens: int,
) -> list[dict[str, Any]]:
    topk_label = max(int(kv_topk), 0)
    block_label = max(int(kv_block_size), 0)
    share_stride = max(int(kv_layer_share_stride), 0)
    refresh_no_cache = max(int(refresh_no_cache), 1)
    refresh_with_cache = max(int(refresh_with_cache), 1)
    hot_window = max(int(kv_hot_window), 0)
    calibration_tokens = max(int(kv_calibration_tokens), 0)
    variants = [
        {"name": "fp32", "kv_cache_precision": "fp32", "kv_rotation_mode": "qr", "kv_hot_window": hot_window},
        {
            "name": "turbo-int8-qr",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "qr",
            "kv_hot_window": hot_window,
        },
        {
            "name": "turbo-int8-hadamard",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
        },
        {
            "name": "turbo-4bit",
            "kv_cache_precision": "turbo-4bit",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
        },
        {
            "name": "adaptive",
            "kv_cache_precision": "adaptive",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
            "kv_calibration_tokens": calibration_tokens,
        },
        {
            "name": "turbo-4bitk-int8v",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-4bit",
            "kv_value_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
        },
        {
            "name": "turbo-int8k-4bitv",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-int8",
            "kv_value_precision": "turbo-4bit",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
        },
        {
            "name": f"turbo-int8-topk{topk_label}-refresh{refresh_no_cache}",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
            "kv_topk": topk_label,
            "kv_index_refresh_interval": refresh_no_cache,
        },
        {
            "name": f"turbo-int8-topk{topk_label}-refresh{refresh_with_cache}",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
            "kv_topk": topk_label,
            "kv_index_refresh_interval": refresh_with_cache,
        },
        {
            "name": f"turbo-int8-topk{topk_label}-refresh{refresh_with_cache}-block{block_label}",
            "kv_cache_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
            "kv_topk": topk_label,
            "kv_index_refresh_interval": refresh_with_cache,
            "kv_block_size": block_label,
        },
        {
            "name": f"turbo-4bitk-int8v-topk{topk_label}-refresh{refresh_with_cache}-block{block_label}",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-4bit",
            "kv_value_precision": "turbo-int8",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
            "kv_topk": topk_label,
            "kv_index_refresh_interval": refresh_with_cache,
            "kv_block_size": block_label,
        },
        {
            "name": f"turbo-int8k-4bitv-topk{topk_label}-refresh{refresh_with_cache}-block{block_label}",
            "kv_cache_precision": "turbo-int8",
            "kv_key_precision": "turbo-int8",
            "kv_value_precision": "turbo-4bit",
            "kv_rotation_mode": "hadamard",
            "kv_hot_window": hot_window,
            "kv_topk": topk_label,
            "kv_index_refresh_interval": refresh_with_cache,
            "kv_block_size": block_label,
        },
    ]
    if share_stride > 1:
        variants.extend(
            [
                {
                    "name": f"turbo-int8-topk{topk_label}-refresh{refresh_with_cache}-block{block_label}-share{share_stride}",
                    "kv_cache_precision": "turbo-int8",
                    "kv_rotation_mode": "hadamard",
                    "kv_hot_window": hot_window,
                    "kv_topk": topk_label,
                    "kv_index_refresh_interval": refresh_with_cache,
                    "kv_block_size": block_label,
                    "kv_layer_share_stride": share_stride,
                },
                {
                    "name": f"turbo-4bitk-int8v-topk{topk_label}-refresh{refresh_with_cache}-block{block_label}-share{share_stride}",
                    "kv_cache_precision": "turbo-int8",
                    "kv_key_precision": "turbo-4bit",
                    "kv_value_precision": "turbo-int8",
                    "kv_rotation_mode": "hadamard",
                    "kv_hot_window": hot_window,
                    "kv_topk": topk_label,
                    "kv_index_refresh_interval": refresh_with_cache,
                    "kv_block_size": block_label,
                    "kv_layer_share_stride": share_stride,
                },
                {
                    "name": f"turbo-int8k-4bitv-topk{topk_label}-refresh{refresh_with_cache}-block{block_label}-share{share_stride}",
                    "kv_cache_precision": "turbo-int8",
                    "kv_key_precision": "turbo-int8",
                    "kv_value_precision": "turbo-4bit",
                    "kv_rotation_mode": "hadamard",
                    "kv_hot_window": hot_window,
                    "kv_topk": topk_label,
                    "kv_index_refresh_interval": refresh_with_cache,
                    "kv_block_size": block_label,
                    "kv_layer_share_stride": share_stride,
                },
            ]
        )
    return variants


def _landscape_row_tags(runtime_entry: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    if runtime_entry.get("kv_cache_precision") == "adaptive":
        tags.append("adaptive")
    if runtime_entry.get("kv_key_precision") or runtime_entry.get("kv_value_precision"):
        tags.append("asymmetric")
    if int(runtime_entry.get("kv_topk", 0)) > 0:
        tags.append("selective")
    if int(runtime_entry.get("kv_topk", 0)) > 0 and int(runtime_entry.get("kv_index_refresh_interval", 1)) > 1:
        tags.append("index-cache")
    if int(runtime_entry.get("kv_block_size", 0)) > 0:
        tags.append("block-scoring")
    if int(runtime_entry.get("kv_layer_share_stride", 0)) > 1:
        tags.append("layer-share")
    rotation = runtime_entry.get("kv_rotation_mode")
    if rotation == "hadamard":
        tags.append("hadamard")
    elif rotation == "qr":
        tags.append("qr")
    return tags


def _build_gpt_landscape_rows(
    runtime_report: dict[str, Any],
    session_report: dict[str, Any],
    *,
    session_summary_length: int,
) -> list[dict[str, Any]]:
    baseline_name = str(runtime_report["baseline_variant"])
    baseline_runtime = runtime_report["variants"][baseline_name]
    baseline_time = float(baseline_runtime["total_time_s"])
    baseline_kv_bytes = int(baseline_runtime["kv_cache_bytes"])
    session_prompt_key = str(int(session_summary_length))
    session_by_variant = session_report["variants"][session_prompt_key]

    rows: list[dict[str, Any]] = []
    for variant_name, runtime_entry in runtime_report["variants"].items():
        session_entry = session_by_variant[variant_name]
        comparison = runtime_entry.get("logit_comparison_vs_baseline") or {}
        session_sizes = {
            length_key: int(session_report["variants"][length_key][variant_name]["session_total_bytes"])
            for length_key in session_report["variants"]
        }
        rows.append(
            {
                "name": variant_name,
                "tags": _landscape_row_tags(runtime_entry),
                "kv_cache_precision": runtime_entry.get("kv_cache_precision"),
                "kv_key_precision": runtime_entry.get("kv_key_precision"),
                "kv_value_precision": runtime_entry.get("kv_value_precision"),
                "kv_rotation_mode": runtime_entry.get("kv_rotation_mode"),
                "kv_hot_window": int(runtime_entry.get("kv_hot_window", 0)),
                "kv_topk": int(runtime_entry.get("kv_topk", 0)),
                "kv_index_refresh_interval": int(runtime_entry.get("kv_index_refresh_interval", 1)),
                "kv_block_size": int(runtime_entry.get("kv_block_size", 0)),
                "kv_layer_share_stride": int(runtime_entry.get("kv_layer_share_stride", 0)),
                "prompt_perplexity": float(runtime_entry["prompt_perplexity"]),
                "total_time_s": float(runtime_entry["total_time_s"]),
                "avg_step_ms": float(runtime_entry["avg_step_ms"]),
                "speedup_vs_fp32": (baseline_time / float(runtime_entry["total_time_s"]))
                if float(runtime_entry["total_time_s"])
                else float("inf"),
                "kv_cache_bytes": int(runtime_entry["kv_cache_bytes"]),
                "kv_cache_ratio_vs_fp32": (float(baseline_kv_bytes) / float(runtime_entry["kv_cache_bytes"]))
                if int(runtime_entry["kv_cache_bytes"])
                else 0.0,
                "generated_match_vs_baseline": bool(runtime_entry["generated_match_vs_baseline"]),
                "cosine_similarity": None
                if not comparison
                else float(comparison.get("cosine_similarity", 0.0)),
                "max_abs_err": None if not comparison else float(comparison.get("max_abs_err", 0.0)),
                "mean_abs_err": None if not comparison else float(comparison.get("mean_abs_err", 0.0)),
                "current_kv_mode": runtime_entry.get("current_kv_mode"),
                "kv_mode_trace": list(runtime_entry.get("kv_mode_trace") or []),
                "switch_events": list(runtime_entry.get("switch_events") or []),
                "policy_baseline_loss": runtime_entry.get("policy_baseline_loss"),
                "policy_recent_loss": runtime_entry.get("policy_recent_loss"),
                "mode_histogram": dict(runtime_entry.get("mode_histogram") or {}),
                "kv_kurtosis_profile": runtime_entry.get("kv_kurtosis_profile"),
                "session_summary_prompt_length": int(session_summary_length),
                "session_total_bytes": int(session_entry["session_total_bytes"]),
                "session_size_ratio_vs_fp32": float(session_entry["session_size_ratio_vs_fp32"]),
                "session_kv_npz_bytes": int(session_entry["kv_npz_bytes"]),
                "session_npz_ratio_vs_fp32": float(session_entry["npz_size_ratio_vs_fp32"]),
                "session_logical_kv_bytes": int(session_entry["logical_kv_cache_bytes"]),
                "session_logical_kv_ratio_vs_fp32": float(session_entry["logical_kv_ratio_vs_fp32"]),
                "session_sizes_by_prompt_length": session_sizes,
                "kv_selective_stats": runtime_entry.get("kv_selective_stats"),
                "kv_cross_layer_overlap": runtime_entry.get("kv_cross_layer_overlap"),
            }
        )
    return rows


def _cmd_benchmark_gpt_session_size(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else Path("gpt-session-size-benchmark").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    max_prompt_length = max(args.prompt_lengths)
    max_positions = max(int(args.max_position_embeddings), max_prompt_length + int(args.max_new_tokens) + 1)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=max_positions,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    variants = [
        {"name": "fp32", "kv_cache_precision": "fp32", "kv_rotation_mode": "qr"},
        {"name": "turbo-int8-hadamard", "kv_cache_precision": "turbo-int8", "kv_rotation_mode": "hadamard"},
        {"name": "turbo-4bit", "kv_cache_precision": "turbo-4bit", "kv_rotation_mode": "hadamard"},
    ]
    report = _benchmark_gpt_session_size_variants(
        export_dir,
        prompt_lengths=[int(length) for length in args.prompt_lengths],
        max_new_tokens=int(args.max_new_tokens),
        kv_quant_seed=int(args.kv_quant_seed),
        kv_variants=variants,
        vocab_size=int(args.vocab_size),
        session_root=workdir / "sessions",
    )
    report["kv_hot_window"] = int(args.kv_hot_window)

    report_path = workdir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"demo workspace written to {workdir}")
    print(f"report={report_path}")
    for prompt_length in args.prompt_lengths:
        for variant_name, entry in report["variants"][str(prompt_length)].items():
            print(
                f"prompt_len={prompt_length} {variant_name}: "
                f"session_bytes={entry['session_total_bytes']} "
                f"kv_npz_bytes={entry['kv_npz_bytes']} "
                f"logical_kv_bytes={entry['logical_kv_cache_bytes']} "
                f"session_ratio_vs_fp32={entry['session_size_ratio_vs_fp32']:.4f}"
            )
    return 0


def _cmd_benchmark_gpt_landscape(args: argparse.Namespace) -> int:
    workdir = Path(args.output).resolve() if args.output else (Path("verification") / "gpt-kv-landscape").resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    session_prompt_lengths = [int(length) for length in args.session_prompt_lengths]
    if int(args.session_summary_length) not in session_prompt_lengths:
        session_prompt_lengths.append(int(args.session_summary_length))
    session_prompt_lengths = sorted(set(session_prompt_lengths))
    max_prompt_length = max([int(args.prompt_length)] + session_prompt_lengths)
    max_positions = max(int(args.max_position_embeddings), max_prompt_length + int(args.max_new_tokens) + 1)

    model_dir = workdir / "tiny-gpt2"
    export_dir = workdir / "export"
    _cmd_build_tiny_gpt2(
        argparse.Namespace(
            output=model_dir,
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            max_position_embeddings=max_positions,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        )
    )
    export_huggingface_model(str(model_dir), export_dir, block_rows=args.block_rows, local_files_only=True)

    variants = _build_gpt_landscape_variants(
        kv_hot_window=int(args.kv_hot_window),
        kv_topk=int(args.kv_topk),
        refresh_no_cache=int(args.kv_refresh_no_cache),
        refresh_with_cache=int(args.kv_refresh_with_cache),
        kv_block_size=int(args.kv_block_size),
        kv_layer_share_stride=int(args.kv_layer_share_stride),
        kv_calibration_tokens=int(args.kv_calibration_tokens),
    )
    runtime_prompt_ids = _prompt_ids_for_length(int(args.prompt_length), int(args.vocab_size))
    runtime_report = benchmark_gpt2_kv_mode_matrix(
        export_dir,
        prompt_ids=runtime_prompt_ids,
        max_new_tokens=int(args.max_new_tokens),
        kv_variants=variants,
        kv_quant_seed=int(args.kv_quant_seed),
    )
    session_report = _benchmark_gpt_session_size_variants(
        export_dir,
        prompt_lengths=session_prompt_lengths,
        max_new_tokens=int(args.max_new_tokens),
        kv_quant_seed=int(args.kv_quant_seed),
        kv_variants=variants,
        vocab_size=int(args.vocab_size),
        session_root=workdir / "sessions",
    )
    rows = _build_gpt_landscape_rows(
        runtime_report,
        session_report,
        session_summary_length=int(args.session_summary_length),
    )
    report = {
        "runtime_prompt_length": int(args.prompt_length),
        "runtime_prompt_ids": runtime_prompt_ids,
        "session_prompt_lengths": session_prompt_lengths,
        "session_summary_length": int(args.session_summary_length),
        "max_new_tokens": int(args.max_new_tokens),
        "kv_quant_seed": int(args.kv_quant_seed),
        "variant_order": [str(variant["name"]) for variant in variants],
        "rows": rows,
        "runtime_matrix": runtime_report,
        "session_matrix": session_report,
    }

    report_path = workdir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"demo workspace written to {workdir}")
    print(f"report={report_path}")
    for row in rows:
        print(
            f"{row['name']}: "
            f"time_s={row['total_time_s']:.4f} "
            f"speedup_vs_fp32={row['speedup_vs_fp32']:.4f} "
            f"kv_ratio_vs_fp32={row['kv_cache_ratio_vs_fp32']:.4f} "
            f"session_ratio_vs_fp32={row['session_size_ratio_vs_fp32']:.4f} "
            f"match={row['generated_match_vs_baseline']}"
        )
    return 0


def _cmd_benchmark_transformers_kv(args: argparse.Namespace) -> int:
    output_path = Path(args.output).resolve()
    report_path = output_path if output_path.suffix.lower() == ".json" else output_path / "benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    variants = build_transformers_variant_set(
        str(args.variant_set),
        kv_quant_seed=int(args.kv_quant_seed),
        kv_hot_window=int(args.kv_hot_window),
        kv_calibration_tokens=int(args.kv_calibration_tokens),
        kv_adaptive_medium_kurtosis=float(args.adaptive_medium_kurtosis),
        kv_adaptive_high_kurtosis=float(args.adaptive_high_kurtosis),
    )

    report = run_transformers_kv_benchmark(
        args.model_ref,
        prompt_ids=None if args.prompt_ids is None else [int(token_id) for token_id in args.prompt_ids],
        prompt_text=args.prompt_text,
        prompt_length=int(args.prompt_length),
        max_new_tokens=int(args.max_new_tokens),
        kv_variants=variants,
        kv_quant_seed=int(args.kv_quant_seed),
        kv_hot_window=int(args.kv_hot_window),
        kv_calibration_tokens=int(args.kv_calibration_tokens),
        kv_adaptive_high_kurtosis=float(args.adaptive_high_kurtosis),
        kv_adaptive_medium_kurtosis=float(args.adaptive_medium_kurtosis),
        local_files_only=bool(args.local_files_only),
        device=args.device,
        trust_remote_code=bool(args.trust_remote_code),
    )
    report["published_context"] = published_benchmark_context()
    report["variant_set"] = str(args.variant_set)

    rows: list[dict[str, Any]] = []
    for variant_name in report["variant_order"]:
        entry = report["variants"][variant_name]
        comparison = entry.get("logit_comparison_vs_baseline") or {}
        rows.append(
            {
                "name": variant_name,
                "kv_cache_precision": entry.get("kv_cache_precision"),
                "kv_key_precision": entry.get("kv_key_precision"),
                "kv_value_precision": entry.get("kv_value_precision"),
                "kv_key_scaling_strategy": entry.get("kv_key_scaling_strategy"),
                "kv_value_scaling_strategy": entry.get("kv_value_scaling_strategy"),
                "kv_rotation_mode": entry.get("kv_rotation_mode"),
                "prompt_perplexity": float(entry["prompt_perplexity"]),
                "total_time_s": float(entry["total_time_s"]),
                "avg_step_ms": float(entry["avg_step_ms"]),
                "tokens_per_second": float(entry["tokens_per_second"]),
                "speedup_vs_native": float(entry["speedup_vs_native"]),
                "speedup_vs_fp32": float(entry["speedup_vs_fp32"]),
                "kv_cache_bytes": int(entry["kv_cache_bytes"]),
                "kv_cache_ratio_vs_native": float(entry["kv_cache_ratio_vs_native"]),
                "kv_cache_ratio_vs_fp32_equivalent": float(entry["kv_cache_ratio_vs_fp32_equivalent"]),
                "kv_cache_ratio_vs_fp32": float(entry["kv_cache_ratio_vs_fp32"]),
                "session_total_bytes": int(entry["session_total_bytes"]),
                "session_size_ratio_vs_native": float(entry["session_size_ratio_vs_native"]),
                "session_size_ratio_vs_fp32_equivalent": float(entry["session_size_ratio_vs_fp32_equivalent"]),
                "session_size_ratio_vs_fp32": float(entry["session_size_ratio_vs_fp32"]),
                "session_meta_bytes": int(entry["session_meta_bytes"]),
                "session_npz_bytes": int(entry["session_npz_bytes"]),
                "session_save_time_ms": float(entry["session_save_time_ms"]),
                "session_load_time_ms": float(entry["session_load_time_ms"]),
                "prompt_perplexity_delta_pct_vs_native": float(entry["prompt_perplexity_delta_pct_vs_native"]),
                "prompt_perplexity_delta_pct_vs_fp32": float(entry["prompt_perplexity_delta_pct_vs_fp32"]),
                "generated_match_vs_baseline": bool(entry["generated_match_vs_baseline"]),
                "cosine_similarity": None if not comparison else float(comparison.get("cosine_similarity", 0.0)),
                "cosine_similarity_vs_baseline": None
                if not comparison
                else float(comparison.get("cosine_similarity", 0.0)),
                "max_abs_err": None if not comparison else float(comparison.get("max_abs_err", 0.0)),
                "max_abs_err_vs_baseline": None if not comparison else float(comparison.get("max_abs_err", 0.0)),
                "mean_abs_err": None if not comparison else float(comparison.get("mean_abs_err", 0.0)),
                "current_kv_mode": entry.get("current_kv_mode"),
                "kv_kurtosis_profile": entry.get("kv_kurtosis_profile"),
                "layer_mode_counts": entry.get("layer_mode_counts"),
                "layer_kv_mode_counts": entry.get("layer_kv_mode_counts"),
                "kv_norm_ratio_per_layer": entry.get("kv_norm_ratio_per_layer"),
                "protected_layer_indices": entry.get("protected_layer_indices"),
                "sparse_v_skip_ratio": entry.get("sparse_v_skip_ratio"),
                "gpu_peak_memory_mb": entry.get("gpu_peak_memory_mb"),
                "gpu_peak_memory_delta_vs_native_mb": entry.get("gpu_peak_memory_delta_vs_native_mb"),
                "gpu_peak_memory_delta_vs_fp32_mb": entry.get("gpu_peak_memory_delta_vs_fp32_mb"),
                "model_device": entry.get("model_device"),
                "cache_device": entry.get("cache_device"),
                "native_kv_dtype": entry.get("native_kv_dtype"),
                "native_element_size_bytes": entry.get("native_element_size_bytes"),
                "total_inference_footprint_bytes": entry.get("total_inference_footprint_bytes"),
            }
        )
    report["rows"] = rows

    report_path.write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")

    print(f"saved transformers KV benchmark to {report_path}")
    for row in rows:
        print(
            f"{row['name']}: "
            f"time_s={row['total_time_s']:.4f} "
            f"speedup_vs_native={row['speedup_vs_native']:.4f} "
            f"kv_ratio_vs_native={row['kv_cache_ratio_vs_native']:.4f} "
            f"session_ratio_vs_native={row['session_size_ratio_vs_native']:.4f} "
            f"kv_ratio_vs_fp32eq={row['kv_cache_ratio_vs_fp32_equivalent']:.4f} "
            f"save_ms={row['session_save_time_ms']:.2f} "
            f"load_ms={row['session_load_time_ms']:.2f} "
            f"ppl_delta_native_pct={row['prompt_perplexity_delta_pct_vs_native']:.2f} "
            f"match={row['generated_match_vs_baseline']}"
        )
    return 0


def _cmd_benchmark_local_models(args: argparse.Namespace) -> int:
    model_refs = args.model_refs or list(DEFAULT_MODEL_REFS)
    report = benchmark_models(
        model_refs,
        prompt_path=args.prompt_path,
        workspace_root=args.workspace_root,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        torch_dtype=args.torch_dtype,
    )
    json_path, markdown_path = save_benchmark_report(report, args.output)
    print(f"saved benchmark report to {json_path}")
    print(f"saved benchmark summary to {markdown_path}")
    _print_json(report)
    return 0


def _cmd_benchmark_tool_calling(args: argparse.Namespace) -> int:
    report = benchmark_tool_calling(
        args.model_refs,
        case_path=args.case_path,
        workspace_root=args.workspace_root,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        torch_dtype=args.torch_dtype,
        limit_cases=args.limit_cases,
    )
    json_path, markdown_path = save_tool_call_report(report, args.output)
    print(f"saved tool-call report to {json_path}")
    print(f"saved tool-call summary to {markdown_path}")
    _print_json(report)
    return 0


def _cmd_prepare_best_assistants(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    report = load_benchmark_report(args.report)
    recommendations = dict(report.get("recommendations", {}) or {})
    existing_aliases = {
        str(item.get("model_ref")): str(item.get("alias"))
        for item in runtime.list_models()
        if item.get("model_ref") and item.get("alias")
    }
    prepared: dict[str, str] = {}
    configured: list[dict[str, Any]] = []

    for assistant_id in ("general", "code", "legal"):
        item = recommendations.get(assistant_id)
        if not item:
            continue
        model_ref = str(item["model_ref"])
        alias = prepared.get(model_ref) or existing_aliases.get(model_ref)
        if alias is None:
            alias = _prepare_alias_for_model_ref(model_ref)
            runtime.prepare_model(
                model_ref=model_ref,
                alias=alias,
                block_rows=args.block_rows,
                compress=args.compress,
                local_files_only=args.local_files_only,
                trust_remote_code=args.trust_remote_code,
                force=args.force,
                chat_format=args.chat_format,
                n_ctx=args.n_ctx,
            )
        prepared[model_ref] = alias
        configured.append(runtime.configure_assistant(assistant_id, alias=alias))

    _print_json({"prepared_aliases": prepared, "assistants": configured})
    return 0


def _cmd_eval_finetuned_model(args: argparse.Namespace) -> int:
    report = benchmark_models(
        args.model_refs,
        prompt_path=args.prompt_path,
        workspace_root=args.workspace_root,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        torch_dtype=args.torch_dtype,
    )
    json_path, markdown_path = save_benchmark_report(report, args.output)
    print(f"saved benchmark report to {json_path}")
    print(f"saved benchmark summary to {markdown_path}")

    if args.baseline_report:
        baseline = load_benchmark_report(args.baseline_report)
        pairs = None
        if args.baseline_model_ref:
            pairs = [(args.baseline_model_ref, model_ref) for model_ref in args.model_refs]
        comparison = compare_benchmark_reports(baseline, report, model_pairs=pairs)
        comparison_json, comparison_md = save_benchmark_comparison(comparison, args.output)
        print(f"saved benchmark comparison to {comparison_json}")
        print(f"saved benchmark comparison summary to {comparison_md}")

    _print_json(report)
    return 0


def _cmd_prepare_model(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    info = runtime.prepare_model(
        model_ref=args.model_ref,
        alias=args.alias,
        block_rows=args.block_rows,
        compress=args.compress,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        force=args.force,
        chat_format=args.chat_format,
        n_ctx=args.n_ctx,
    )
    _print_json(info)
    return 0


def _cmd_list_models(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    _print_json(
        {
            "workspace_root": workspace_root(args.workspace_root),
            "models": runtime.list_models(),
        }
    )
    return 0


def _cmd_model_info(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    _print_json(runtime.model_info(args.alias))
    return 0


def _cmd_run_gpt(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    session_id = args.session_id
    if args.save_session and not session_id:
        session_id = _default_session_id(args.alias)
    result = runtime.generate(
        alias=args.alias,
        prompt_ids=args.prompt_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
        session_id=session_id if (args.save_session or session_id) else None,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved generation result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_run_gpt_text(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    session_id = args.session_id
    if args.save_session and not session_id:
        session_id = _default_session_id(args.alias)
    messages = None
    if args.system_prompt or args.user_prompt:
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        if args.user_prompt:
            messages.append({"role": "user", "content": args.user_prompt})
    result = runtime.generate_text(
        alias=args.alias,
        prompt=args.prompt,
        messages=messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
        session_id=session_id if (args.save_session or session_id) else None,
        add_special_tokens=args.add_special_tokens,
        skip_special_tokens=not args.keep_special_tokens,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved text generation result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_resume_gpt(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    result = runtime.resume(
        alias=args.alias,
        session_id=args.session_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
        save_session=not args.no_save_session,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved resumed result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_resume_gpt_text(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    result = runtime.resume_text(
        alias=args.alias,
        session_id=args.session_id,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        cache_mode=args.cache_mode,
        save_session=not args.no_save_session,
        skip_special_tokens=not args.keep_special_tokens,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved resumed text result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_list_tools(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    _print_json({"tools": runtime.tool_manifest()})
    return 0


def _cmd_call_tool(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    arguments = json.loads(args.arguments) if args.arguments else {}
    result = runtime.call_tool(args.tool_name, arguments)
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved tool result to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_agent_add_text(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    runner = runtime.agent_runner()
    result = runner.add_knowledge_text(
        args.agent_name,
        args.text,
        source=args.source,
        metadata=json.loads(args.metadata) if args.metadata else None,
    )
    _print_json(result)
    return 0


def _cmd_agent_add_file(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    runner = runtime.agent_runner()
    result = runner.add_knowledge_file(
        args.agent_name,
        args.file_path,
        metadata=json.loads(args.metadata) if args.metadata else None,
    )
    _print_json(result)
    return 0


def _cmd_agent_search(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    runner = runtime.agent_runner()
    result = runner.search_knowledge(args.agent_name, args.query, top_k=args.top_k)
    _print_json(result)
    return 0


def _cmd_agent_run(args: argparse.Namespace) -> int:
    runtime = HelixRuntime(root=args.workspace_root)
    runner = runtime.agent_runner()
    result = runner.run(
        goal=args.goal,
        agent_name=args.agent_name,
        default_model_alias=args.default_model_alias,
        local_planner_alias=args.local_planner_alias,
        remote_model_ref=args.remote_model_ref,
        prefer_remote=args.prefer_remote,
        trust_remote_code=args.trust_remote_code,
        max_steps=args.max_steps,
        generation_max_new_tokens=args.generation_max_new_tokens,
    )
    if args.output_json:
        args.output_json.write_text(json.dumps(_json_ready(result), indent=2), encoding="utf-8")
        print(f"saved agent run to {args.output_json}")
    else:
        _print_json(result)
    return 0


def _cmd_serve_api(args: argparse.Namespace) -> int:
    root = workspace_root(args.workspace_root)
    print(f"serving helix api on http://{args.host}:{args.port}")
    print(f"workspace_root={root}")
    serve_api(host=args.host, port=args.port, root=root)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Block-compressed streaming tensor prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert a .npy matrix into the block store.")
    convert.add_argument("input", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--block-rows", type=int, default=256)
    convert.add_argument("--level", type=int, default=6)
    convert.set_defaults(func=_cmd_convert)

    verify = subparsers.add_parser("verify", help="Verify all compressed blocks.")
    verify.add_argument("store", type=Path)
    verify.set_defaults(func=_cmd_verify)

    matvec = subparsers.add_parser("matvec", help="Run streaming matrix-vector multiply.")
    matvec.add_argument("store", type=Path)
    matvec.add_argument("vector", type=Path)
    matvec.add_argument("--output", type=Path)
    matvec.set_defaults(func=_cmd_matvec)

    benchmark = subparsers.add_parser("benchmark", help="Compare dense and streaming execution.")
    benchmark.add_argument("--rows", type=int, default=4096)
    benchmark.add_argument("--cols", type=int, default=2048)
    benchmark.add_argument("--block-rows", type=int, default=256)
    benchmark.add_argument("--seed", type=int, default=7)
    benchmark.set_defaults(func=_cmd_benchmark)

    demo = subparsers.add_parser("demo", help="Create a full local demo workspace.")
    demo.add_argument("--rows", type=int, default=1024)
    demo.add_argument("--cols", type=int, default=512)
    demo.add_argument("--block-rows", type=int, default=128)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", type=Path)
    demo.set_defaults(func=_cmd_demo)

    convert_hf = subparsers.add_parser(
        "convert-hf",
        help="Load a Hugging Face model and export supported tensors into block stores.",
    )
    convert_hf.add_argument("model_ref")
    convert_hf.add_argument("output", type=Path)
    convert_hf.add_argument("--block-rows", type=int, default=256)
    convert_hf.add_argument("--level", type=int, default=6)
    convert_hf.add_argument("--local-files-only", action="store_true")
    convert_hf.add_argument("--trust-remote-code", action="store_true")
    convert_hf.set_defaults(func=_cmd_convert_hf)

    convert_npz = subparsers.add_parser(
        "convert-npz",
        help="Export a local .npz tensor bundle through the Hugging Face-style manifest pipeline.",
    )
    convert_npz.add_argument("input", type=Path)
    convert_npz.add_argument("output", type=Path)
    convert_npz.add_argument("--block-rows", type=int, default=256)
    convert_npz.set_defaults(func=_cmd_convert_npz)

    build_tiny_bert = subparsers.add_parser(
        "build-tiny-bert",
        help="Create a tiny local Hugging Face masked LM for controlled export tests.",
    )
    build_tiny_bert.add_argument("output", type=Path)
    build_tiny_bert.add_argument("--vocab-size", type=int, default=64)
    build_tiny_bert.add_argument("--hidden-size", type=int, default=32)
    build_tiny_bert.add_argument("--max-position-embeddings", type=int, default=16)
    build_tiny_bert.add_argument("--type-vocab-size", type=int, default=2)
    build_tiny_bert.add_argument("--num-hidden-layers", type=int, default=0)
    build_tiny_bert.set_defaults(func=_cmd_build_tiny_bert)

    build_tiny_gpt2 = subparsers.add_parser(
        "build-tiny-gpt2",
        help="Create a tiny local GPT2 causal LM for export tests.",
    )
    build_tiny_gpt2.add_argument("output", type=Path)
    build_tiny_gpt2.add_argument("--vocab-size", type=int, default=64)
    build_tiny_gpt2.add_argument("--hidden-size", type=int, default=32)
    build_tiny_gpt2.add_argument("--max-position-embeddings", type=int, default=16)
    build_tiny_gpt2.add_argument("--num-layers", type=int, default=2)
    build_tiny_gpt2.add_argument("--num-heads", type=int, default=4)
    build_tiny_gpt2.set_defaults(func=_cmd_build_tiny_gpt2)

    demo_hf_infer = subparsers.add_parser(
        "demo-hf-infer",
        help="Build, export, and validate a first real inference path on exported weights.",
    )
    demo_hf_infer.add_argument("--output", type=Path)
    demo_hf_infer.add_argument("--vocab-size", type=int, default=64)
    demo_hf_infer.add_argument("--hidden-size", type=int, default=32)
    demo_hf_infer.add_argument("--max-position-embeddings", type=int, default=16)
    demo_hf_infer.add_argument("--type-vocab-size", type=int, default=2)
    demo_hf_infer.add_argument("--block-rows", type=int, default=8)
    demo_hf_infer.add_argument("--token-id", type=int, default=7)
    demo_hf_infer.add_argument("--token-type-id", type=int, default=0)
    demo_hf_infer.add_argument("--top-k", type=int, default=5)
    demo_hf_infer.set_defaults(func=_cmd_demo_hf_infer)

    demo_bert_block = subparsers.add_parser(
        "demo-bert-block",
        help="Build, export, and validate a full first BERT transformer block on exported weights.",
    )
    demo_bert_block.add_argument("--output", type=Path)
    demo_bert_block.add_argument("--vocab-size", type=int, default=64)
    demo_bert_block.add_argument("--hidden-size", type=int, default=32)
    demo_bert_block.add_argument("--max-position-embeddings", type=int, default=16)
    demo_bert_block.add_argument("--type-vocab-size", type=int, default=2)
    demo_bert_block.add_argument("--block-rows", type=int, default=8)
    demo_bert_block.add_argument("--token-ids", type=int, nargs="+", default=[7, 11])
    demo_bert_block.add_argument("--top-k", type=int, default=5)
    demo_bert_block.set_defaults(func=_cmd_demo_bert_block)

    demo_bert_stack = subparsers.add_parser(
        "demo-bert-stack",
        help="Build, export, and validate a multi-layer BERT masked LM on exported weights.",
    )
    demo_bert_stack.add_argument("--output", type=Path)
    demo_bert_stack.add_argument("--vocab-size", type=int, default=64)
    demo_bert_stack.add_argument("--hidden-size", type=int, default=32)
    demo_bert_stack.add_argument("--max-position-embeddings", type=int, default=16)
    demo_bert_stack.add_argument("--type-vocab-size", type=int, default=2)
    demo_bert_stack.add_argument("--num-hidden-layers", type=int, default=2)
    demo_bert_stack.add_argument("--block-rows", type=int, default=8)
    demo_bert_stack.add_argument("--token-ids", type=int, nargs="+", default=[7, 11, 13])
    demo_bert_stack.add_argument("--top-k", type=int, default=5)
    demo_bert_stack.set_defaults(func=_cmd_demo_bert_stack)

    demo_gpt_causal = subparsers.add_parser(
        "demo-gpt-causal",
        help="Build, export, and validate a tiny GPT2 causal LM on exported weights.",
    )
    demo_gpt_causal.add_argument("--output", type=Path)
    demo_gpt_causal.add_argument("--vocab-size", type=int, default=64)
    demo_gpt_causal.add_argument("--hidden-size", type=int, default=32)
    demo_gpt_causal.add_argument("--max-position-embeddings", type=int, default=16)
    demo_gpt_causal.add_argument("--num-layers", type=int, default=2)
    demo_gpt_causal.add_argument("--num-heads", type=int, default=4)
    demo_gpt_causal.add_argument("--block-rows", type=int, default=8)
    demo_gpt_causal.add_argument("--token-ids", type=int, nargs="+", default=[3, 5, 8])
    demo_gpt_causal.add_argument("--top-k", type=int, default=5)
    demo_gpt_causal.set_defaults(func=_cmd_demo_gpt_causal)

    demo_gpt_generate = subparsers.add_parser(
        "demo-gpt-generate",
        help="Run greedy token-by-token GPT generation with KV cache and compare against HF generate().",
    )
    demo_gpt_generate.add_argument("--output", type=Path)
    demo_gpt_generate.add_argument("--vocab-size", type=int, default=64)
    demo_gpt_generate.add_argument("--hidden-size", type=int, default=32)
    demo_gpt_generate.add_argument("--max-position-embeddings", type=int, default=16)
    demo_gpt_generate.add_argument("--num-layers", type=int, default=2)
    demo_gpt_generate.add_argument("--num-heads", type=int, default=4)
    demo_gpt_generate.add_argument("--block-rows", type=int, default=8)
    demo_gpt_generate.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8])
    demo_gpt_generate.add_argument("--max-new-tokens", type=int, default=4)
    demo_gpt_generate.add_argument(
        "--kv-mode",
        "--kv-cache-precision",
        dest="kv_cache_precision",
        choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"],
        default="fp32",
    )
    demo_gpt_generate.add_argument("--kv-rotation", dest="kv_rotation_mode", choices=["qr", "hadamard"], default="hadamard")
    demo_gpt_generate.add_argument("--kv-quant-seed", type=int, default=7)
    demo_gpt_generate.add_argument("--kv-hot-window", type=int, default=0)
    demo_gpt_generate.add_argument("--kv-topk", type=int, default=0)
    demo_gpt_generate.set_defaults(func=_cmd_demo_gpt_generate)

    demo_gpt_sample = subparsers.add_parser(
        "demo-gpt-sample",
        help="Run sampled GPT generation with temperature/top-k/top-p on the streaming engine.",
    )
    demo_gpt_sample.add_argument("--output", type=Path)
    demo_gpt_sample.add_argument("--vocab-size", type=int, default=64)
    demo_gpt_sample.add_argument("--hidden-size", type=int, default=32)
    demo_gpt_sample.add_argument("--max-position-embeddings", type=int, default=16)
    demo_gpt_sample.add_argument("--num-layers", type=int, default=2)
    demo_gpt_sample.add_argument("--num-heads", type=int, default=4)
    demo_gpt_sample.add_argument("--block-rows", type=int, default=8)
    demo_gpt_sample.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8])
    demo_gpt_sample.add_argument("--max-new-tokens", type=int, default=4)
    demo_gpt_sample.add_argument("--temperature", type=float, default=0.9)
    demo_gpt_sample.add_argument("--top-k", type=int, default=10)
    demo_gpt_sample.add_argument("--top-p", type=float, default=0.9)
    demo_gpt_sample.add_argument("--seed", type=int, default=7)
    demo_gpt_sample.add_argument("--cache-mode", default="fresh")
    demo_gpt_sample.add_argument(
        "--kv-mode",
        "--kv-cache-precision",
        dest="kv_cache_precision",
        choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl", "adaptive"],
        default="fp32",
    )
    demo_gpt_sample.add_argument("--kv-rotation", dest="kv_rotation_mode", choices=["qr", "hadamard"], default="hadamard")
    demo_gpt_sample.add_argument("--kv-key-mode", dest="kv_key_precision", choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"])
    demo_gpt_sample.add_argument("--kv-value-mode", dest="kv_value_precision", choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"])
    demo_gpt_sample.add_argument("--kv-quant-seed", type=int, default=7)
    demo_gpt_sample.add_argument("--kv-calibration-tokens", type=int, default=128)
    demo_gpt_sample.add_argument("--kv-hot-window", type=int, default=0)
    demo_gpt_sample.add_argument("--kv-topk", type=int, default=0)
    demo_gpt_sample.add_argument("--kv-index-refresh-interval", type=int, default=8)
    demo_gpt_sample.add_argument("--kv-block-size", type=int, default=0)
    demo_gpt_sample.add_argument("--kv-layer-share-stride", type=int, default=0)
    demo_gpt_sample.set_defaults(func=_cmd_demo_gpt_sample)

    demo_gpt_resume = subparsers.add_parser(
        "demo-gpt-resume",
        help="Save a GPT session with KV cache and resume generation from disk.",
    )
    demo_gpt_resume.add_argument("--output", type=Path)
    demo_gpt_resume.add_argument("--vocab-size", type=int, default=64)
    demo_gpt_resume.add_argument("--hidden-size", type=int, default=32)
    demo_gpt_resume.add_argument("--max-position-embeddings", type=int, default=16)
    demo_gpt_resume.add_argument("--num-layers", type=int, default=2)
    demo_gpt_resume.add_argument("--num-heads", type=int, default=4)
    demo_gpt_resume.add_argument("--block-rows", type=int, default=8)
    demo_gpt_resume.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8])
    demo_gpt_resume.add_argument("--first-new-tokens", type=int, default=2)
    demo_gpt_resume.add_argument("--second-new-tokens", type=int, default=2)
    demo_gpt_resume.add_argument("--do-sample", action="store_true")
    demo_gpt_resume.add_argument("--temperature", type=float, default=0.9)
    demo_gpt_resume.add_argument("--top-k", type=int, default=10)
    demo_gpt_resume.add_argument("--top-p", type=float, default=0.9)
    demo_gpt_resume.add_argument("--seed", type=int, default=7)
    demo_gpt_resume.add_argument(
        "--kv-mode",
        "--kv-cache-precision",
        dest="kv_cache_precision",
        choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl", "adaptive"],
        default="fp32",
    )
    demo_gpt_resume.add_argument("--kv-rotation", dest="kv_rotation_mode", choices=["qr", "hadamard"], default="hadamard")
    demo_gpt_resume.add_argument("--kv-key-mode", dest="kv_key_precision", choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"])
    demo_gpt_resume.add_argument("--kv-value-mode", dest="kv_value_precision", choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"])
    demo_gpt_resume.add_argument("--kv-quant-seed", type=int, default=7)
    demo_gpt_resume.add_argument("--kv-calibration-tokens", type=int, default=128)
    demo_gpt_resume.add_argument("--kv-hot-window", type=int, default=0)
    demo_gpt_resume.add_argument("--kv-topk", type=int, default=0)
    demo_gpt_resume.add_argument("--kv-index-refresh-interval", type=int, default=8)
    demo_gpt_resume.add_argument("--kv-block-size", type=int, default=0)
    demo_gpt_resume.add_argument("--kv-layer-share-stride", type=int, default=0)
    demo_gpt_resume.set_defaults(func=_cmd_demo_gpt_resume)

    demo_gpt_remote = subparsers.add_parser(
        "demo-gpt-remote",
        help="Download a small Hugging Face GPT model, export it, and validate logits/generation.",
    )
    demo_gpt_remote.add_argument("--output", type=Path)
    demo_gpt_remote.add_argument("--model-ref", default="sshleifer/tiny-gpt2")
    demo_gpt_remote.add_argument("--block-rows", type=int, default=8)
    demo_gpt_remote.add_argument("--prompt-ids", type=int, nargs="+", default=[1, 2, 3])
    demo_gpt_remote.add_argument("--max-new-tokens", type=int, default=3)
    demo_gpt_remote.add_argument("--trust-remote-code", action="store_true")
    demo_gpt_remote.set_defaults(func=_cmd_demo_gpt_remote)

    benchmark_gpt_cache = subparsers.add_parser(
        "benchmark-gpt-cache",
        help="Compare GPT generation with no cache, fresh cache, and warmed session cache.",
    )
    benchmark_gpt_cache.add_argument("--output", type=Path)
    benchmark_gpt_cache.add_argument("--vocab-size", type=int, default=64)
    benchmark_gpt_cache.add_argument("--hidden-size", type=int, default=32)
    benchmark_gpt_cache.add_argument("--max-position-embeddings", type=int, default=16)
    benchmark_gpt_cache.add_argument("--num-layers", type=int, default=2)
    benchmark_gpt_cache.add_argument("--num-heads", type=int, default=4)
    benchmark_gpt_cache.add_argument("--block-rows", type=int, default=8)
    benchmark_gpt_cache.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8])
    benchmark_gpt_cache.add_argument("--max-new-tokens", type=int, default=4)
    benchmark_gpt_cache.add_argument(
        "--kv-mode",
        "--kv-cache-precision",
        dest="kv_cache_precision",
        choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl", "adaptive"],
        default="fp32",
    )
    benchmark_gpt_cache.add_argument("--kv-rotation", dest="kv_rotation_mode", choices=["qr", "hadamard"], default="hadamard")
    benchmark_gpt_cache.add_argument("--kv-key-mode", dest="kv_key_precision", choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"])
    benchmark_gpt_cache.add_argument("--kv-value-mode", dest="kv_value_precision", choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"])
    benchmark_gpt_cache.add_argument("--kv-quant-seed", type=int, default=7)
    benchmark_gpt_cache.add_argument("--kv-calibration-tokens", type=int, default=128)
    benchmark_gpt_cache.add_argument("--kv-hot-window", type=int, default=0)
    benchmark_gpt_cache.add_argument("--kv-topk", type=int, default=0)
    benchmark_gpt_cache.add_argument("--kv-index-refresh-interval", type=int, default=8)
    benchmark_gpt_cache.add_argument("--kv-block-size", type=int, default=0)
    benchmark_gpt_cache.add_argument("--kv-layer-share-stride", type=int, default=0)
    benchmark_gpt_cache.set_defaults(func=_cmd_benchmark_gpt_cache)

    benchmark_gpt_suite = subparsers.add_parser(
        "benchmark-gpt-suite",
        help="Benchmark GPT cache modes across multiple prompt lengths and save a JSON report.",
    )
    benchmark_gpt_suite.add_argument("--output", type=Path)
    benchmark_gpt_suite.add_argument("--vocab-size", type=int, default=64)
    benchmark_gpt_suite.add_argument("--hidden-size", type=int, default=32)
    benchmark_gpt_suite.add_argument("--max-position-embeddings", type=int, default=16)
    benchmark_gpt_suite.add_argument("--num-layers", type=int, default=2)
    benchmark_gpt_suite.add_argument("--num-heads", type=int, default=4)
    benchmark_gpt_suite.add_argument("--block-rows", type=int, default=8)
    benchmark_gpt_suite.add_argument("--prompt-lengths", type=int, nargs="+", default=[1, 2, 4, 8])
    benchmark_gpt_suite.add_argument("--max-new-tokens", type=int, default=4)
    benchmark_gpt_suite.add_argument("--warm-repeats", type=int, default=2)
    benchmark_gpt_suite.add_argument(
        "--kv-mode",
        "--kv-cache-precision",
        dest="kv_cache_precision",
        choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl", "adaptive"],
        default="fp32",
    )
    benchmark_gpt_suite.add_argument("--kv-rotation", dest="kv_rotation_mode", choices=["qr", "hadamard"], default="hadamard")
    benchmark_gpt_suite.add_argument("--kv-key-mode", dest="kv_key_precision", choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"])
    benchmark_gpt_suite.add_argument("--kv-value-mode", dest="kv_value_precision", choices=["fp32", "turbo-int8", "turbo-4bit", "turbo-qjl"])
    benchmark_gpt_suite.add_argument("--kv-quant-seed", type=int, default=7)
    benchmark_gpt_suite.add_argument("--kv-calibration-tokens", type=int, default=128)
    benchmark_gpt_suite.add_argument("--kv-hot-window", type=int, default=0)
    benchmark_gpt_suite.add_argument("--kv-topk", type=int, default=0)
    benchmark_gpt_suite.add_argument("--kv-index-refresh-interval", type=int, default=8)
    benchmark_gpt_suite.add_argument("--kv-block-size", type=int, default=0)
    benchmark_gpt_suite.add_argument("--kv-layer-share-stride", type=int, default=0)
    benchmark_gpt_suite.set_defaults(func=_cmd_benchmark_gpt_suite)

    benchmark_gpt_kv_modes = subparsers.add_parser(
        "benchmark-gpt-kv-modes",
        help="Compare fp32, turbo-int8, turbo-4bit, and deprecated turbo-qjl on the GPT streaming engine.",
    )
    benchmark_gpt_kv_modes.add_argument("--output", type=Path)
    benchmark_gpt_kv_modes.add_argument("--vocab-size", type=int, default=64)
    benchmark_gpt_kv_modes.add_argument("--hidden-size", type=int, default=32)
    benchmark_gpt_kv_modes.add_argument("--max-position-embeddings", type=int, default=16)
    benchmark_gpt_kv_modes.add_argument("--num-layers", type=int, default=2)
    benchmark_gpt_kv_modes.add_argument("--num-heads", type=int, default=4)
    benchmark_gpt_kv_modes.add_argument("--block-rows", type=int, default=8)
    benchmark_gpt_kv_modes.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8, 13, 21, 34])
    benchmark_gpt_kv_modes.add_argument("--max-new-tokens", type=int, default=4)
    benchmark_gpt_kv_modes.add_argument("--kv-quant-seed", type=int, default=7)
    benchmark_gpt_kv_modes.add_argument("--kv-calibration-tokens", type=int, default=128)
    benchmark_gpt_kv_modes.add_argument("--kv-hot-window", type=int, default=0)
    benchmark_gpt_kv_modes.add_argument("--kv-topk", type=int, default=0)
    benchmark_gpt_kv_modes.add_argument("--kv-index-refresh-interval", type=int, default=8)
    benchmark_gpt_kv_modes.add_argument("--kv-block-size", type=int, default=0)
    benchmark_gpt_kv_modes.add_argument("--kv-layer-share-stride", type=int, default=0)
    benchmark_gpt_kv_modes.add_argument("--include-qjl", action="store_true", help="Include deprecated turbo-qjl research mode.")
    benchmark_gpt_kv_modes.add_argument("--include-adaptive", action="store_true")
    benchmark_gpt_kv_modes.add_argument("--include-asymmetric", action="store_true")
    benchmark_gpt_kv_modes.set_defaults(func=_cmd_benchmark_gpt_kv_modes)

    benchmark_gpt_kv_policy = subparsers.add_parser(
        "benchmark-gpt-kv-policy",
        help="Benchmark static fp32/int8/4bit against the adaptive KV policy on the GPT streaming engine.",
    )
    benchmark_gpt_kv_policy.add_argument("--output", type=Path)
    benchmark_gpt_kv_policy.add_argument("--vocab-size", type=int, default=64)
    benchmark_gpt_kv_policy.add_argument("--hidden-size", type=int, default=32)
    benchmark_gpt_kv_policy.add_argument("--max-position-embeddings", type=int, default=16)
    benchmark_gpt_kv_policy.add_argument("--num-layers", type=int, default=2)
    benchmark_gpt_kv_policy.add_argument("--num-heads", type=int, default=4)
    benchmark_gpt_kv_policy.add_argument("--block-rows", type=int, default=8)
    benchmark_gpt_kv_policy.add_argument("--prompt-ids", type=int, nargs="+", default=[3, 5, 8, 13, 21, 34])
    benchmark_gpt_kv_policy.add_argument("--max-new-tokens", type=int, default=4)
    benchmark_gpt_kv_policy.add_argument("--kv-quant-seed", type=int, default=7)
    benchmark_gpt_kv_policy.add_argument("--kv-hot-window", type=int, default=4)
    benchmark_gpt_kv_policy.set_defaults(func=_cmd_benchmark_gpt_kv_policy)

    benchmark_gpt_landscape = subparsers.add_parser(
        "benchmark-gpt-landscape",
        help="Run a single unified KV benchmark report across dense, compressed, selective, and session-size variants.",
    )
    benchmark_gpt_landscape.add_argument("--output", type=Path)
    benchmark_gpt_landscape.add_argument("--prompt-length", type=int, default=112)
    benchmark_gpt_landscape.add_argument("--session-prompt-lengths", type=int, nargs="+", default=[128, 512, 1024])
    benchmark_gpt_landscape.add_argument("--session-summary-length", type=int, default=512)
    benchmark_gpt_landscape.add_argument("--max-new-tokens", type=int, default=2)
    benchmark_gpt_landscape.add_argument("--vocab-size", type=int, default=128)
    benchmark_gpt_landscape.add_argument("--hidden-size", type=int, default=64)
    benchmark_gpt_landscape.add_argument("--max-position-embeddings", type=int, default=1024)
    benchmark_gpt_landscape.add_argument("--num-layers", type=int, default=4)
    benchmark_gpt_landscape.add_argument("--num-heads", type=int, default=4)
    benchmark_gpt_landscape.add_argument("--block-rows", type=int, default=16)
    benchmark_gpt_landscape.add_argument("--kv-hot-window", type=int, default=4)
    benchmark_gpt_landscape.add_argument("--kv-topk", type=int, default=8)
    benchmark_gpt_landscape.add_argument("--kv-refresh-no-cache", type=int, default=1)
    benchmark_gpt_landscape.add_argument("--kv-refresh-with-cache", type=int, default=8)
    benchmark_gpt_landscape.add_argument("--kv-block-size", type=int, default=16)
    benchmark_gpt_landscape.add_argument("--kv-layer-share-stride", type=int, default=0)
    benchmark_gpt_landscape.add_argument("--kv-calibration-tokens", type=int, default=128)
    benchmark_gpt_landscape.add_argument("--kv-quant-seed", type=int, default=7)
    benchmark_gpt_landscape.set_defaults(func=_cmd_benchmark_gpt_landscape)

    benchmark_gpt_session_size = subparsers.add_parser(
        "benchmark-gpt-session-size",
        help="Compare on-disk GPT session sizes for fp32, turbo-int8, and turbo-4bit.",
    )
    benchmark_gpt_session_size.add_argument("--output", type=Path)
    benchmark_gpt_session_size.add_argument("--vocab-size", type=int, default=64)
    benchmark_gpt_session_size.add_argument("--hidden-size", type=int, default=32)
    benchmark_gpt_session_size.add_argument("--max-position-embeddings", type=int, default=1024)
    benchmark_gpt_session_size.add_argument("--num-layers", type=int, default=2)
    benchmark_gpt_session_size.add_argument("--num-heads", type=int, default=4)
    benchmark_gpt_session_size.add_argument("--block-rows", type=int, default=8)
    benchmark_gpt_session_size.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 512, 1024])
    benchmark_gpt_session_size.add_argument("--max-new-tokens", type=int, default=1)
    benchmark_gpt_session_size.add_argument("--kv-quant-seed", type=int, default=7)
    benchmark_gpt_session_size.add_argument("--kv-hot-window", type=int, default=0)
    benchmark_gpt_session_size.set_defaults(func=_cmd_benchmark_gpt_session_size)

    benchmark_transformers_kv = subparsers.add_parser(
        "benchmark-transformers-kv",
        help="Benchmark Helix KV compression through AutoModelForCausalLM without exporting to the GPT runtime.",
    )
    benchmark_transformers_kv.add_argument("model_ref")
    benchmark_transformers_kv.add_argument("--output", type=Path, default=Path("verification") / "transformers-kv-benchmark")
    benchmark_transformers_kv.add_argument("--prompt-ids", type=int, nargs="+")
    benchmark_transformers_kv.add_argument("--prompt-text")
    benchmark_transformers_kv.add_argument("--prompt-length", type=int, default=512)
    benchmark_transformers_kv.add_argument("--max-new-tokens", type=int, default=32)
    benchmark_transformers_kv.add_argument("--device")
    benchmark_transformers_kv.add_argument("--local-files-only", action="store_true")
    benchmark_transformers_kv.add_argument("--trust-remote-code", action="store_true")
    benchmark_transformers_kv.add_argument("--kv-hot-window", type=int, default=4)
    benchmark_transformers_kv.add_argument("--kv-quant-seed", type=int, default=7)
    benchmark_transformers_kv.add_argument("--kv-calibration-tokens", type=int, default=128)
    benchmark_transformers_kv.add_argument("--adaptive-high-kurtosis", type=float, default=20.0)
    benchmark_transformers_kv.add_argument("--adaptive-medium-kurtosis", type=float, default=9.0)
    benchmark_transformers_kv.add_argument("--variant-set", choices=["stable", "asymmetry-sweep", "community"], default="stable")
    benchmark_transformers_kv.set_defaults(func=_cmd_benchmark_transformers_kv)

    benchmark_local_models = subparsers.add_parser(
        "benchmark-local-models",
        help="Benchmark local candidate models for the three assistant roles.",
    )
    benchmark_local_models.add_argument("model_refs", nargs="*")
    benchmark_local_models.add_argument("--prompt-path", type=Path)
    benchmark_local_models.add_argument("--workspace-root", type=Path)
    benchmark_local_models.add_argument("--output", type=Path, default=Path("benchmark-output") / "local-models")
    benchmark_local_models.add_argument("--max-new-tokens", type=int, default=80)
    benchmark_local_models.add_argument("--max-input-tokens", type=int, default=512)
    benchmark_local_models.add_argument("--torch-dtype", default="auto")
    benchmark_local_models.add_argument("--trust-remote-code", action="store_true")
    benchmark_local_models.add_argument("--allow-downloads", dest="local_files_only", action="store_false")
    benchmark_local_models.set_defaults(func=_cmd_benchmark_local_models, local_files_only=True)

    benchmark_tool_calling_parser = subparsers.add_parser(
        "benchmark-tool-calling",
        help="Benchmark planner-style JSON tool selection against the local tool suites.",
    )
    benchmark_tool_calling_parser.add_argument("model_refs", nargs="+")
    benchmark_tool_calling_parser.add_argument("--case-path", type=Path)
    benchmark_tool_calling_parser.add_argument("--workspace-root", type=Path)
    benchmark_tool_calling_parser.add_argument(
        "--output", type=Path, default=Path("benchmark-output") / "tool-calling"
    )
    benchmark_tool_calling_parser.add_argument("--max-new-tokens", type=int, default=192)
    benchmark_tool_calling_parser.add_argument("--max-input-tokens", type=int, default=2048)
    benchmark_tool_calling_parser.add_argument("--torch-dtype", default="auto")
    benchmark_tool_calling_parser.add_argument("--limit-cases", type=int)
    benchmark_tool_calling_parser.add_argument("--trust-remote-code", action="store_true")
    benchmark_tool_calling_parser.add_argument("--allow-downloads", dest="local_files_only", action="store_false")
    benchmark_tool_calling_parser.set_defaults(func=_cmd_benchmark_tool_calling, local_files_only=True)

    prepare_best_assistants = subparsers.add_parser(
        "prepare-best-assistants",
        help="Prepare the recommended model aliases and bind them to the three assistants.",
    )
    prepare_best_assistants.add_argument("--report", type=Path, required=True)
    prepare_best_assistants.add_argument("--workspace-root", type=Path)
    prepare_best_assistants.add_argument("--block-rows", type=int, default=256)
    prepare_best_assistants.add_argument("--compress", choices=["cdnav3"])
    prepare_best_assistants.add_argument("--trust-remote-code", action="store_true")
    prepare_best_assistants.add_argument("--allow-downloads", dest="local_files_only", action="store_false")
    prepare_best_assistants.add_argument("--force", action="store_true")
    prepare_best_assistants.add_argument("--chat-format")
    prepare_best_assistants.add_argument("--n-ctx", type=int, default=4096)
    prepare_best_assistants.set_defaults(func=_cmd_prepare_best_assistants, local_files_only=True)

    eval_finetuned_model = subparsers.add_parser(
        "eval-finetuned-model",
        help="Benchmark one or more locally merged fine-tuned models and compare against a baseline report.",
    )
    eval_finetuned_model.add_argument("model_refs", nargs="+")
    eval_finetuned_model.add_argument("--baseline-report", type=Path)
    eval_finetuned_model.add_argument("--baseline-model-ref")
    eval_finetuned_model.add_argument("--prompt-path", type=Path)
    eval_finetuned_model.add_argument("--workspace-root", type=Path)
    eval_finetuned_model.add_argument("--output", type=Path, default=Path("benchmark-output") / "finetuned")
    eval_finetuned_model.add_argument("--max-new-tokens", type=int, default=80)
    eval_finetuned_model.add_argument("--max-input-tokens", type=int, default=512)
    eval_finetuned_model.add_argument("--torch-dtype", default="auto")
    eval_finetuned_model.add_argument("--trust-remote-code", action="store_true")
    eval_finetuned_model.add_argument("--allow-downloads", dest="local_files_only", action="store_false")
    eval_finetuned_model.set_defaults(func=_cmd_eval_finetuned_model, local_files_only=True)

    prepare_model = subparsers.add_parser(
        "prepare-model",
        help="Prepare a reusable model workspace from a Hugging Face ref or local model path.",
    )
    prepare_model.add_argument("model_ref")
    prepare_model.add_argument("--alias")
    prepare_model.add_argument("--workspace-root", type=Path)
    prepare_model.add_argument("--block-rows", type=int, default=256)
    prepare_model.add_argument("--compress", choices=["cdnav3"])
    prepare_model.add_argument("--allow-downloads", dest="local_files_only", action="store_false")
    prepare_model.add_argument("--trust-remote-code", action="store_true")
    prepare_model.add_argument("--force", action="store_true")
    prepare_model.add_argument("--chat-format")
    prepare_model.add_argument("--n-ctx", type=int, default=4096)
    prepare_model.set_defaults(func=_cmd_prepare_model, local_files_only=True)

    list_models = subparsers.add_parser(
        "list-models",
        help="List prepared model workspaces and any saved sessions.",
    )
    list_models.add_argument("--workspace-root", type=Path)
    list_models.set_defaults(func=_cmd_list_models)

    model_info = subparsers.add_parser(
        "model-info",
        help="Show metadata for a prepared model alias.",
    )
    model_info.add_argument("alias")
    model_info.add_argument("--workspace-root", type=Path)
    model_info.set_defaults(func=_cmd_model_info)

    run_gpt = subparsers.add_parser(
        "run-gpt",
        help="Run generation on a prepared GPT workspace alias and optionally persist a session.",
    )
    run_gpt.add_argument("alias")
    run_gpt.add_argument("--workspace-root", type=Path)
    run_gpt.add_argument("--prompt-ids", type=int, nargs="+", required=True)
    run_gpt.add_argument("--max-new-tokens", type=int, default=4)
    run_gpt.add_argument("--do-sample", action="store_true")
    run_gpt.add_argument("--temperature", type=float, default=0.9)
    run_gpt.add_argument("--top-k", type=int, default=10)
    run_gpt.add_argument("--top-p", type=float, default=0.9)
    run_gpt.add_argument("--seed", type=int)
    run_gpt.add_argument("--cache-mode", choices=["none", "fresh", "session"], default="session")
    run_gpt.add_argument("--save-session", action="store_true")
    run_gpt.add_argument("--session-id")
    run_gpt.add_argument("--output-json", type=Path)
    run_gpt.set_defaults(func=_cmd_run_gpt)

    run_gpt_text = subparsers.add_parser(
        "run-gpt-text",
        help="Run text generation on a prepared GPT workspace alias using its tokenizer.",
    )
    run_gpt_text.add_argument("alias")
    run_gpt_text.add_argument("--workspace-root", type=Path)
    run_gpt_text.add_argument("--prompt")
    run_gpt_text.add_argument("--system-prompt")
    run_gpt_text.add_argument("--user-prompt")
    run_gpt_text.add_argument("--max-new-tokens", type=int, default=64)
    run_gpt_text.add_argument("--do-sample", action="store_true")
    run_gpt_text.add_argument("--temperature", type=float, default=0.9)
    run_gpt_text.add_argument("--top-k", type=int, default=10)
    run_gpt_text.add_argument("--top-p", type=float, default=0.9)
    run_gpt_text.add_argument("--seed", type=int)
    run_gpt_text.add_argument("--cache-mode", choices=["none", "fresh", "session"], default="session")
    run_gpt_text.add_argument("--save-session", action="store_true")
    run_gpt_text.add_argument("--session-id")
    run_gpt_text.add_argument("--add-special-tokens", action="store_true")
    run_gpt_text.add_argument("--keep-special-tokens", action="store_true")
    run_gpt_text.add_argument("--output-json", type=Path)
    run_gpt_text.set_defaults(func=_cmd_run_gpt_text)

    resume_gpt = subparsers.add_parser(
        "resume-gpt",
        help="Resume generation from a saved session for a prepared GPT workspace alias.",
    )
    resume_gpt.add_argument("alias")
    resume_gpt.add_argument("session_id")
    resume_gpt.add_argument("--workspace-root", type=Path)
    resume_gpt.add_argument("--max-new-tokens", type=int, default=4)
    resume_gpt.add_argument("--do-sample", action="store_true")
    resume_gpt.add_argument("--temperature", type=float, default=0.9)
    resume_gpt.add_argument("--top-k", type=int, default=10)
    resume_gpt.add_argument("--top-p", type=float, default=0.9)
    resume_gpt.add_argument("--seed", type=int)
    resume_gpt.add_argument("--cache-mode", choices=["none", "fresh", "session"], default="session")
    resume_gpt.add_argument("--no-save-session", action="store_true")
    resume_gpt.add_argument("--output-json", type=Path)
    resume_gpt.set_defaults(func=_cmd_resume_gpt)

    resume_gpt_text = subparsers.add_parser(
        "resume-gpt-text",
        help="Resume text generation from a saved session for a prepared GPT workspace alias.",
    )
    resume_gpt_text.add_argument("alias")
    resume_gpt_text.add_argument("session_id")
    resume_gpt_text.add_argument("--workspace-root", type=Path)
    resume_gpt_text.add_argument("--max-new-tokens", type=int, default=64)
    resume_gpt_text.add_argument("--do-sample", action="store_true")
    resume_gpt_text.add_argument("--temperature", type=float, default=0.9)
    resume_gpt_text.add_argument("--top-k", type=int, default=10)
    resume_gpt_text.add_argument("--top-p", type=float, default=0.9)
    resume_gpt_text.add_argument("--seed", type=int)
    resume_gpt_text.add_argument("--cache-mode", choices=["none", "fresh", "session"], default="session")
    resume_gpt_text.add_argument("--no-save-session", action="store_true")
    resume_gpt_text.add_argument("--keep-special-tokens", action="store_true")
    resume_gpt_text.add_argument("--output-json", type=Path)
    resume_gpt_text.set_defaults(func=_cmd_resume_gpt_text)

    list_tools = subparsers.add_parser(
        "list-tools",
        help="List the tool registry exposed by the current runtime.",
    )
    list_tools.add_argument("--workspace-root", type=Path)
    list_tools.set_defaults(func=_cmd_list_tools)

    call_tool = subparsers.add_parser(
        "call-tool",
        help="Invoke one registered runtime tool with JSON arguments.",
    )
    call_tool.add_argument("tool_name")
    call_tool.add_argument("--workspace-root", type=Path)
    call_tool.add_argument("--arguments", default="{}")
    call_tool.add_argument("--output-json", type=Path)
    call_tool.set_defaults(func=_cmd_call_tool)

    agent_add_text = subparsers.add_parser(
        "agent-add-text",
        help="Add inline text to an agent knowledge base for later RAG retrieval.",
    )
    agent_add_text.add_argument("agent_name")
    agent_add_text.add_argument("text")
    agent_add_text.add_argument("--workspace-root", type=Path)
    agent_add_text.add_argument("--source", default="inline-text")
    agent_add_text.add_argument("--metadata")
    agent_add_text.set_defaults(func=_cmd_agent_add_text)

    agent_add_file = subparsers.add_parser(
        "agent-add-file",
        help="Ingest a file into an agent knowledge base. PDFs work if pypdf is installed.",
    )
    agent_add_file.add_argument("agent_name")
    agent_add_file.add_argument("file_path", type=Path)
    agent_add_file.add_argument("--workspace-root", type=Path)
    agent_add_file.add_argument("--metadata")
    agent_add_file.set_defaults(func=_cmd_agent_add_file)

    agent_search = subparsers.add_parser(
        "agent-search",
        help="Search an agent knowledge base.",
    )
    agent_search.add_argument("agent_name")
    agent_search.add_argument("query")
    agent_search.add_argument("--workspace-root", type=Path)
    agent_search.add_argument("--top-k", type=int, default=5)
    agent_search.set_defaults(func=_cmd_agent_search)

    agent_run = subparsers.add_parser(
        "agent-run",
        help="Run the planner -> tool -> observation -> final loop with memory and RAG.",
    )
    agent_run.add_argument("goal")
    agent_run.add_argument("--agent-name", default="default-agent")
    agent_run.add_argument("--workspace-root", type=Path)
    agent_run.add_argument("--default-model-alias")
    agent_run.add_argument("--local-planner-alias")
    agent_run.add_argument("--remote-model-ref")
    agent_run.add_argument("--prefer-remote", action="store_true")
    agent_run.add_argument("--trust-remote-code", action="store_true")
    agent_run.add_argument("--max-steps", type=int, default=4)
    agent_run.add_argument("--generation-max-new-tokens", type=int, default=128)
    agent_run.add_argument("--output-json", type=Path)
    agent_run.set_defaults(func=_cmd_agent_run)

    serve_api_parser = subparsers.add_parser(
        "serve-api",
        help="Expose prepared models through a small local JSON API.",
    )
    serve_api_parser.add_argument("--workspace-root", type=Path)
    serve_api_parser.add_argument("--host", default="127.0.0.1")
    serve_api_parser.add_argument("--port", type=int, default=8080)
    serve_api_parser.set_defaults(func=_cmd_serve_api)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
