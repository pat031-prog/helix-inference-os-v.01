from __future__ import annotations

__version__ = "0.2.1-local"

from helix_substrate.cdna_encoder import decode_cdna_to_tensor, encode_tensor_to_cdna
from helix_substrate.cdnav3_writer import CDNAv3Writer
from helix_substrate.helix_linear import (
    HelixLinear,
    freeze_sidecar_phase,
    load_cdna_factors,
    load_helix_linear_from_cdnav3,
    swap_summary,
    swap_to_helix,
)
from helix_substrate.tensor_policy import (
    TensorClass,
    TensorPolicy,
    classify_tensor,
    get_default_policy,
    get_policy,
)

__all__ = [
    "CDNAv3Writer",
    "HelixLinear",
    "TensorClass",
    "TensorPolicy",
    "classify_tensor",
    "decode_cdna_to_tensor",
    "encode_tensor_to_cdna",
    "freeze_sidecar_phase",
    "get_cdna_factors",
    "get_default_policy",
    "get_policy",
    "load_cdna_factors",
    "load_helix_linear_from_cdnav3",
    "swap_summary",
    "swap_to_helix",
]


def get_cdna_factors(*args, **kwargs):
    return load_cdna_factors(*args, **kwargs)
