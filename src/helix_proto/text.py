from __future__ import annotations

from pathlib import Path
from typing import Any


_TOKENIZER_CACHE: dict[str, Any] = {}


def _require_auto_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "text features need optional dependencies: pip install -e '.[hf]'"
        ) from exc
    return AutoTokenizer


def _require_fast_tokenizer():
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "toy tokenizer generation needs optional dependencies: pip install -e '.[hf]'"
        ) from exc
    return Tokenizer, models, pre_tokenizers, PreTrainedTokenizerFast


def tokenizer_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "tokenizer"


def prepare_tokenizer(
    model_ref: str,
    output_dir: str | Path,
    *,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    AutoTokenizer = _require_auto_tokenizer()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.save_pretrained(output_dir)

    return {
        "has_tokenizer": True,
        "tokenizer_dir": str(output_dir),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "special_tokens": dict(tokenizer.special_tokens_map),
    }


def try_prepare_tokenizer(
    model_ref: str,
    output_dir: str | Path,
    *,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    try:
        return prepare_tokenizer(
            model_ref,
            output_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "has_tokenizer": False,
            "tokenizer_dir": str(Path(output_dir)),
            "tokenizer_error": str(exc),
        }


def save_toy_tokenizer(output_dir: str | Path, *, vocab_size: int) -> dict[str, Any]:
    Tokenizer, models, pre_tokenizers, PreTrainedTokenizerFast = _require_fast_tokenizer()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if vocab_size < 8:
        raise ValueError("toy tokenizer needs vocab_size >= 8")

    special_tokens = ["<bos>", "<eos>", "<unk>"]
    base_tokens = [f"tok{i}" for i in range(vocab_size - len(special_tokens))]
    ordered_tokens = special_tokens + base_tokens
    vocab = {token: index for index, token in enumerate(ordered_tokens)}

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<bos>",
    )
    fast.save_pretrained(output_dir)

    return {
        "has_tokenizer": True,
        "tokenizer_dir": str(output_dir),
        "tokenizer_class": fast.__class__.__name__,
        "vocab_size": vocab_size,
        "special_tokens": dict(fast.special_tokens_map),
        "toy_tokenizer": True,
    }


def load_tokenizer(tokenizer_dir: str | Path):
    AutoTokenizer = _require_auto_tokenizer()
    path = str(Path(tokenizer_dir).resolve())
    cached = _TOKENIZER_CACHE.get(path)
    if cached is not None:
        return cached
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    _TOKENIZER_CACHE[path] = tokenizer
    return tokenizer


def encode_text(
    tokenizer_dir: str | Path,
    text: str,
    *,
    add_special_tokens: bool = False,
) -> list[int]:
    tokenizer = load_tokenizer(tokenizer_dir)
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [int(token_id) for token_id in encoded["input_ids"]]


def decode_tokens(
    tokenizer_dir: str | Path,
    token_ids: list[int],
    *,
    skip_special_tokens: bool = True,
) -> str:
    tokenizer = load_tokenizer(tokenizer_dir)
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def render_messages_prompt(
    messages: list[dict[str, str]],
    *,
    assistant_prefix: bool = True,
) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower() or "user"
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    if assistant_prefix:
        lines.append("assistant:")
    return "\n".join(lines)
