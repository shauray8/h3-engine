from __future__ import annotations

import torch
import torch.nn.functional as F

from h3_inference.kernels._dispatch import MISS, enabled, try_call

def packed_self_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full self-attention over the packed sequence.
    Args:
        query/key/value: ``(B, T, H, D)`` — sequence-major, heads *not* yet transposed.
        attention_mask: ``None`` for a padless sequence (the fast path), otherwise a
            boolean ``(T, T)`` document mask.
    Returns:
        ``(B, T, H, D)``, same layout as the inputs.
    """
    if attention_mask is None and enabled("H3_TL_ATTN"):
        b, t, h, d = query.shape
        out = try_call(
            "attention", query.reshape(b, t, h * d), key.reshape(b, t, h * d),
            value.reshape(b, t, h * d), h,
        )
        if out is not MISS:
            return out.reshape(b, t, h, d)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    return out.transpose(1, 2)
