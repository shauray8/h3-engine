""" elementwise + norm + RoPE.

The DiT's non-GEMM math routes through this file:

* :func:`rms_norm`        — the pre-attention / pre-FFN norms (weighted, over ``hidden_size``)
* :func:`qk_norm`         — per-head RMSNorm on q and k before RoPE (over ``head_dim``)
* :func:`apply_rotary_emb`— 3-axis rotary, applied to q and k
* :func:`ada_modulate`    — ``rms_norm(x) * (1 + scale[idx]) + shift[idx]``
* :func:`gate_residual`   — ``residual + gate[idx] * y``
"""

from __future__ import annotations

import torch
from torch import nn

from h3_inference.kernels._dispatch import MISS, enabled, try_call

def rms_norm(norm_module: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    return norm_module(x)

def qk_norm(norm_module: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    return norm_module(x)

def apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if enabled("H3_FUSED_ROPE"):
        out = try_call("rope", hidden_states, cos, sin)
        if out is not MISS:
            return out

    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]

    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()

def _norm_modulate_impl(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:

    xn = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=norm_weight, eps=eps)
    return xn * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)

def _gate_residual_impl(
    residual: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return residual + gate.index_select(0, indices) * y

_compiled: dict[str, object] = {}

def _compiled_fn(name: str, fn):
    if name not in _compiled:
        _compiled[name] = torch.compile(fn, dynamic=False)
    return _compiled[name]

def norm_modulate(
    norm_module: nn.RMSNorm,
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor | None:
    """Fused RMSNorm + gather-modulate, or ``None`` if not covered."""

    if not enabled("H3_FUSED_MODULATE"):
        return None
    w = getattr(norm_module, "weight", None)
    if w is None:
        return None
    out = try_call("norm_modulate_h3", x, w, float(norm_module.eps), scale, shift, indices)
    if out is not MISS:
        return out
    return _compiled_fn("norm_modulate", _norm_modulate_impl)(
        x, w, float(norm_module.eps), scale, shift, indices
    )

def ada_modulate(
    normed: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """``normed * (1 + scale[indices]) + shift[indices]``."""

    if enabled("H3_FUSED_MODULATE"):
        out = try_call("ada_modulate_gather", normed, scale, shift, indices)
        if out is not MISS:
            return out
    return normed * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)

def _norm_rope_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Per-head RMSNorm followed by rotate-half RoPE, as one expression."""

    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    xn = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)
    c1 = cos[None, :, None, :half].to(xn.dtype)
    c2 = cos[None, :, None, half:rotary_dim].to(xn.dtype)
    s1 = sin[None, :, None, :half].to(xn.dtype)
    s2 = sin[None, :, None, half:rotary_dim].to(xn.dtype)
    x1 = xn[..., :half]
    x2 = xn[..., half:rotary_dim]
    return torch.cat((x1 * c1 - x2 * s1, x2 * c2 + x1 * s2, xn[..., rotary_dim:]), dim=-1)

_compiled_norm_rope = None

def _get_compiled():
    global _compiled_norm_rope
    if _compiled_norm_rope is None:
        _compiled_norm_rope = torch.compile(_norm_rope_impl, dynamic=False)
    return _compiled_norm_rope

def norm_rope(
    norm_module: nn.RMSNorm,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor | None:
    """Fused qk_norm + RoPE, or ``None`` if this call is not covered."""

    if not enabled("H3_FUSED_ROPE"):
        return None
    w = getattr(norm_module, "weight", None)
    if w is None or cos is None:
        return None
    out = try_call("norm_rope_h3", x, w, float(norm_module.eps), cos, sin)
    if out is not MISS:
        return out
    return _get_compiled()(x, w, float(norm_module.eps), cos, sin)

def gate_residual(
    residual: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """``residual + gate[indices] * y``."""

    if enabled("H3_FUSED_GATE"):
        out = try_call("gate_residual_gather", residual, y, gate, indices)
        if out is not MISS:
            return out
        return _compiled_fn("gate_residual", _gate_residual_impl)(residual, y, gate, indices)
    return residual + gate.index_select(0, indices) * y
