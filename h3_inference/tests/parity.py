
from __future__ import annotations

import argparse
import sys

import torch

from h3_inference.kernels._dispatch import enabled
from h3_inference.transformer.model import H3Transformer3DModel

ACCEL_FLAGS = ("H3_FUSED_ROPE", "H3_FUSED_MODULATE", "H3_FUSED_GATE", "H3_TL_ATTN", "H3_NVFP4")

FP32_REASSOC_TOL = 1e-4
BF16_REASSOC_TOL = 5e-2

def accel_flags() -> list[str]:
    """Which accelerated rows are armed for this process."""
    return [f for f in ACCEL_FLAGS if enabled(f)]

SMALL = dict(
    num_attention_heads=4,
    attention_head_dim=32,      # heads*dim = 128 != hidden_size, as in the real config
    hidden_size=96,
    num_layers=2,
    num_refiner_layers=2,
    ffn_dim=192,
    in_channels=8,
    audio_in_channels=6,
    patch_size=(1, 2, 2),
    text_dim=64,
    freq_dim=32,
    time_embed_hidden_dim=96,
    time_embed_dim=48,
    rope_freq_dim=4,            # rotates 2*3*4 = 24 of 32 head channels, leaving a pass-through
    rope_theta=10000.0,
    norm_eps=1e-5,
    qk_norm_eps=1e-5,
    final_norm_eps=1e-5,
)

def _build_reference(diffusers_src: str):
    """Import the vendored diffusers branch and build its MiniMaxH3Transformer3DModel."""
    if diffusers_src not in sys.path:
        sys.path.insert(0, diffusers_src)
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel

    return MiniMaxH3Transformer3DModel(**SMALL)

def _inputs(cfg: dict, device: str, dtype: torch.dtype, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_text, n_audio, n_video = 7, 5, 12
    seq = n_text + n_audio + n_video
    patch_dim = cfg["in_channels"] * cfg["patch_size"][0] * cfg["patch_size"][1] * cfg["patch_size"][2]

    def rnd(*shape, dt=dtype):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(device=device, dtype=dt)

    text_indices = torch.arange(0, n_text, device=device)
    audio_indices = torch.arange(n_text, n_text + n_audio, device=device)
    video_indices = torch.arange(n_text + n_audio, seq, device=device)

    token_tags = torch.empty(seq, dtype=torch.long, device=device)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    # Two distinct noise levels, as a real t2va forward has (target video and target audio
    # ride different schedules).
    timestep = torch.tensor([0.9, 0.4], dtype=torch.float32, device=device)
    timestep_indices = torch.zeros(seq, dtype=torch.long, device=device)
    timestep_indices[audio_indices] = 1

    position_ids = torch.stack(
        [
            torch.arange(seq, device=device) % 3,
            torch.arange(seq, device=device) % 5,
            torch.arange(seq, device=device) % 7,
        ],
        dim=-1,
    )

    return dict(
        hidden_states=rnd(1, n_video, patch_dim),
        audio_hidden_states=rnd(1, n_audio, cfg["audio_in_channels"]),
        encoder_hidden_states=rnd(1, n_text, cfg["text_dim"]),
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    )

def _apply_dtype(model: torch.nn.Module, dtype: torch.dtype, mixed: bool) -> None:
    """Cast a built model to the checkpoint's precision policy (or uniformly)."""
    model.to(dtype)
    if not mixed:
        return
    keep = H3Transformer3DModel.KEEP_IN_FP32
    for name, module in model.named_modules():
        if any(marker in name for marker in keep):
            module.to(torch.float32)


def run_case(diffusers_src: str, device: str, dtype: torch.dtype, mixed: bool, with_padding: bool) -> bool:
    reference = _build_reference(diffusers_src)
    ours = H3Transformer3DModel(**SMALL)

    # One state dict into both. Identical keys is itself part of the claim: `strict=True`
    # fails loudly if the fork renamed or dropped anything.
    state = reference.state_dict()
    ours.load_state_dict(state, strict=True)

    reference.eval().requires_grad_(False)
    ours.eval().requires_grad_(False)
    reference.to(device)
    ours.to(device)
    _apply_dtype(reference, dtype, mixed)
    _apply_dtype(ours, dtype, mixed)

    kwargs = _inputs(SMALL, device, dtype)
    if with_padding:
        # Append padding rows (tag -1). They must land in their own attention document,
        # which is the only path that builds a mask at all.
        pad = 4
        seq = kwargs["position_ids"].shape[0]
        kwargs["token_tags"] = torch.cat([kwargs["token_tags"], torch.full((pad,), -1, dtype=torch.long, device=device)])
        kwargs["timestep_indices"] = torch.cat(
            [kwargs["timestep_indices"], torch.zeros(pad, dtype=torch.long, device=device)]
        )
        kwargs["position_ids"] = torch.cat(
            [kwargs["position_ids"], torch.zeros(pad, 3, dtype=kwargs["position_ids"].dtype, device=device)]
        )
        del seq

    with torch.no_grad():
        ref_out = reference(**kwargs, return_dict=True)
        our_out = ours(**kwargs)

    same_video = torch.equal(ref_out.sample, our_out.sample)
    same_audio = torch.equal(ref_out.audio_sample, our_out.audio_sample)

    def _worst(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a.float() - b.float()).abs().max().item()

    dv = _worst(ref_out.sample, our_out.sample)
    da = _worst(ref_out.audio_sample, our_out.audio_sample)
    label = f"{'mixed' if mixed else str(dtype).replace('torch.', '')}{' +pad' if with_padding else ''}"

    if same_video and same_audio:
        print(f"  {label:<18} bit-identical: True")
        return True

    if not accel_flags():
        print(f"  {label:<18} bit-identical: FALSE   max|dv|={dv:.3e} max|da|={da:.3e}")
        return False

    tol = FP32_REASSOC_TOL if dtype is torch.float32 and not mixed else BF16_REASSOC_TOL
    ok = max(dv, da) <= tol
    verdict = "reassociated" if ok else "DIVERGED"
    print(f"  {label:<18} {verdict:<14} max|dv|={dv:.3e} max|da|={da:.3e}  (tol {tol:.0e})")
    return ok

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diffusers-src",
        default="/workspace/Sana/models/minimax_h3/baseline/diffusers_src/src",
        help="The vendored diffusers branch (PR #14355) to compare against.",
    )
    parser.add_argument("--device", default="cpu", help="cpu is enough; structure is what is checked")
    parser.add_argument("--with-padding", action="store_true", help="also run the masked (padded) layout")
    args = parser.parse_args()

    armed = accel_flags()
    print(f"parity vs {args.diffusers_src} on {args.device}")
    if armed:
        print(f"accelerated rows armed: {', '.join(armed)}")
        print("  -> the bar is 'same arithmetic, different rounding', not bit-identity.")
        print("     Run with no H3_* flags set to check the fork itself.")
    ok = True
    ok &= run_case(args.diffusers_src, args.device, torch.float32, mixed=False, with_padding=False)
    ok &= run_case(args.diffusers_src, args.device, torch.bfloat16, mixed=False, with_padding=False)
    ok &= run_case(args.diffusers_src, args.device, torch.bfloat16, mixed=True, with_padding=False)
    if args.with_padding:
        ok &= run_case(args.diffusers_src, args.device, torch.float32, mixed=False, with_padding=True)

    if ok:
        print("ACCEL PARITY OK — reassociation within tolerance" if armed else "PARITY OK")
    else:
        print("PARITY FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
