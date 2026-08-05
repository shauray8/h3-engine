"""FFN + QKVO GEMMs."""

from __future__ import annotations

import torch
from torch import nn

from h3_inference.kernels._dispatch import MISS, enabled, try_call

def qkv_projection(
    to_q: nn.Linear, to_k: nn.Linear, to_v: nn.Linear, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if enabled("H3_NVFP4"):
        out = try_call("qkv_projection", to_q, to_k, to_v, x)
        if out is not MISS:
            return out
        from h3_inference import nvfp4 as _nvfp4

        shared = _nvfp4.shared_qkv(to_q, to_k, to_v, x)
        if shared is not None:
            return shared
    return to_q(x), to_k(x), to_v(x)

def out_projection(to_out: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    if enabled("H3_NVFP4"):
        out = try_call("out_projection", to_out, x)
        if out is not MISS:
            return out
    return to_out(x)

def _swiglu_impl(gate_up: torch.Tensor) -> torch.Tensor:
    """``value * silu(gate)`` from the fused ``(T, 2*ffn_dim)`` projection output."""

    value, gate = gate_up.chunk(2, dim=-1)
    return value * torch.nn.functional.silu(gate)

_compiled_swiglu = None

def _swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    global _compiled_swiglu
    if _compiled_swiglu is None:
        _compiled_swiglu = torch.compile(_swiglu_impl, dynamic=False)
    return _compiled_swiglu(gate_up)

def ffn(ff: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The SwiGLU feed-forward: ``net.0`` (fused gate+up), ``net.2`` (down)."""

    if enabled("H3_NVFP4"):
        out = try_call("ffn", ff, x)
        if out is not MISS:
            return out
    if enabled("H3_FUSED_SWIGLU"):
        return ff.net[2](_swiglu(ff.net[0].proj(x)))
    return ff(x)

def context_projection(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Text/video/audio input projections and the two output heads."""
    return linear(x)

