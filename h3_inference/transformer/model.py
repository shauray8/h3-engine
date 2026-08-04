from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from h3_inference.kernels import elementwise_norm_rope as _ew
from h3_inference.kernels import ffn_qkvo_gemms as _gemm
from h3_inference.kernels import other as _other
from h3_inference.kernels import packed_self_attn as _attn
from h3_inference.transformer.ops import FeedForward, TimestepEmbedding, Timesteps

# H3 tags every row of the packed sequence with its modality and keeps one set of AdaLN
# modulation parameters per (timestep, modality) pair: 0 = video, 1 = text, 2 = audio.
MODALITY_NUM = 3

@dataclass
class H3TransformerOutput:
    sample: torch.Tensor
    audio_sample: torch.Tensor

class H3RotaryPosEmbed(nn.Module):
    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = position_ids.to(torch.float32)
        freqs = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=1)
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()

class H3AdaLayerNormModulation(nn.Module):
    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, 6 * hidden_size * MODALITY_NUM, bias=True)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(nn.functional.silu(temb).to(self.linear.weight.dtype))
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)

class H3AdaLayerNormOut(nn.Module):
    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)
        self.folded_modulation: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor) -> torch.Tensor:
        if self.folded_modulation is not None:
            shift, scale = self.folded_modulation
        else:
            shift, scale = self.linear(nn.functional.silu(temb).to(self.linear.weight.dtype)).chunk(2, dim=-1)
        hidden_states = _ew.rms_norm(self.norm, hidden_states)
        return _ew.ada_modulate(hidden_states, scale, shift, timestep_indices)

class H3Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dim_head: int, qk_norm_eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head

        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        # `to_out[1]` is an inert Dropout, kept so the projection sits at key `to_out.0`.
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, hidden_size, bias=False), nn.Dropout(0.0)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = _gemm.qkv_projection(self.to_q, self.to_k, self.to_v, hidden_states)

        query = query.unflatten(-1, (self.heads, -1))
        key = key.unflatten(-1, (self.heads, -1))
        value = value.unflatten(-1, (self.heads, -1))

        fused_q = _ew.norm_rope(self.norm_q, query, *rotary_emb) if rotary_emb is not None else None
        if fused_q is not None:
            query = fused_q
            key = _ew.norm_rope(self.norm_k, key, *rotary_emb)
        else:
            query = _ew.qk_norm(self.norm_q, query)
            key = _ew.qk_norm(self.norm_k, key)
            if rotary_emb is not None:
                query = _ew.apply_rotary_emb(query, *rotary_emb)
                key = _ew.apply_rotary_emb(key, *rotary_emb)

        hidden_states = _attn.packed_self_attention(query, key, value, attention_mask)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return _gemm.out_projection(self.to_out[0], hidden_states)

class H3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = H3Attention(hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states

class H3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                H3TokenRefinerBlock(
                    hidden_size, num_attention_heads, attention_head_dim, ffn_dim, norm_eps, qk_norm_eps
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)

class H3TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = H3Attention(hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps)
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, bias=False)
        self.adaln_proj = H3AdaLayerNormModulation(time_embed_dim=time_embed_dim, hidden_size=hidden_size)
        self.folded_modulation: tuple[torch.Tensor, ...] | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.folded_modulation is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.folded_modulation
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = _other.adaln_projection(
                self.adaln_proj, temb
            )

        residual = hidden_states
        norm_hidden_states = _ew.norm_modulate(self.norm1, hidden_states, scale_msa, shift_msa, adaln_indices)
        if norm_hidden_states is None:
            norm_hidden_states = _ew.ada_modulate(
                _ew.rms_norm(self.norm1, hidden_states), scale_msa, shift_msa, adaln_indices
            )
        attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
        hidden_states = _ew.gate_residual(residual, attn_output, gate_msa, adaln_indices)

        residual = hidden_states
        norm_hidden_states = _ew.norm_modulate(self.norm2, hidden_states, scale_mlp, shift_mlp, adaln_indices)
        if norm_hidden_states is None:
            norm_hidden_states = _ew.ada_modulate(
                _ew.rms_norm(self.norm2, hidden_states), scale_mlp, shift_mlp, adaln_indices
            )
        ff_output = _gemm.ffn(self.ff, norm_hidden_states)
        hidden_states = _ew.gate_residual(residual, ff_output, gate_mlp, adaln_indices)

        return hidden_states

class H3Transformer3DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.config = dict(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=tuple(patch_size),
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )
        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)

        self.time_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(
            in_channels=freq_dim, time_embed_dim=time_embed_hidden_dim, out_dim=time_embed_dim
        )
        self.rope = H3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)

        self.token_refiner = H3TokenRefiner(
            hidden_size, num_attention_heads, attention_head_dim, ffn_dim,
            num_refiner_layers, norm_eps, qk_norm_eps, final_norm_eps,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                H3TransformerBlock(
                    hidden_size, num_attention_heads, attention_head_dim, ffn_dim,
                    time_embed_dim, norm_eps, qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = H3AdaLayerNormOut(hidden_size, time_embed_dim, eps=final_norm_eps)
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)

    KEEP_IN_FP32 = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out", "rope")

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
    ) -> H3TransformerOutput:
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`, got "
                f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )

        rotary_emb = self.rope(position_ids)

        video_embeds = _gemm.context_projection(self.proj_in, hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = _gemm.context_projection(
            self.audio_proj_in, audio_hidden_states.to(self.audio_proj_in.weight.dtype)
        )
        text_embeds = _gemm.context_projection(
            self.context_embedder, encoder_hidden_states.to(self.context_embedder.weight.dtype)
        )
        text_embeds = _other.token_refiner(self.token_refiner, text_embeds)

        hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = _other.timestep_embedding(self.time_proj, self.time_embedder, timestep)
        adaln_indices = timestep_indices * MODALITY_NUM + token_tags.clamp(min=0)
        attention_mask = None
        is_pad = token_tags < 0
        if bool(is_pad.any()):
            attention_mask = is_pad[None, :] == is_pad[:, None]

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)

        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video_output = _gemm.context_projection(self.proj_out, hidden_states).index_select(1, video_indices)
        audio_output = _gemm.context_projection(self.audio_proj_out, hidden_states).index_select(1, audio_indices)

        return H3TransformerOutput(sample=video_output, audio_sample=audio_output)
