from __future__ import annotations

import torch
from torch import nn

def token_refiner(refiner: nn.Module, text_embeds: torch.Tensor) -> torch.Tensor:
    """The 2-block refiner applied to the projected text stream."""
    return refiner(text_embeds)

def adaln_projection(adaln_proj: nn.Module, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return adaln_proj(temb)

def timestep_embedding(time_proj: nn.Module, time_embedder: nn.Module, timestep: torch.Tensor) -> torch.Tensor:
    temb = time_proj(timestep)
    return time_embedder(temb.to(time_embedder.linear_1.weight.dtype))

