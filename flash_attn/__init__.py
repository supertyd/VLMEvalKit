"""Compatibility shim for Transformers FlashAttention2 imports.

The eval environment used by these scripts already ships vLLM's prebuilt
FlashAttention kernels under ``vllm.vllm_flash_attn``. This local module
exposes the small ``flash_attn`` API surface Transformers checks for and
forwards calls to vLLM's kernels.
"""

from __future__ import annotations

import torch

from vllm.vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func as _vllm_flash_attn_varlen_func,
)

__version__ = "2.8.0"


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    softcap: float = 0.0,
    **kwargs,
):
    return _vllm_flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        softcap=softcap,
        **kwargs,
    )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    softcap: float = 0.0,
    **kwargs,
):
    batch, seqlen_q = q.shape[:2]
    seqlen_k = k.shape[1]
    cu_q = torch.arange(
        0,
        (batch + 1) * seqlen_q,
        seqlen_q,
        device=q.device,
        dtype=torch.int32,
    )
    cu_k = torch.arange(
        0,
        (batch + 1) * seqlen_k,
        seqlen_k,
        device=k.device,
        dtype=torch.int32,
    )
    out = flash_attn_varlen_func(
        q=q.reshape(-1, q.shape[-2], q.shape[-1]),
        k=k.reshape(-1, k.shape[-2], k.shape[-1]),
        v=v.reshape(-1, v.shape[-2], v.shape[-1]),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        softcap=softcap,
        **kwargs,
    )
    if isinstance(out, tuple):
        attn_out = out[0].view(batch, seqlen_q, out[0].shape[-2], out[0].shape[-1])
        return (attn_out, *out[1:])
    return out.view(batch, seqlen_q, out.shape[-2], out.shape[-1])
