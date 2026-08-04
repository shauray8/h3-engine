from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

MODEL_DIR = os.environ.get("H3_PIPELINE_PATH", "/workspace/h3-model-diffusers")
OUT_DIR = Path(os.environ.get("H3_VIDEO_DIR", "/workspace/H3/videos"))

PROMPTS = {
    "dialogue_diner": (
        "Two people talk in a rain-streaked late-night diner booth with red vinyl seats. "
        "A woman in an olive field jacket sits across from a man in a corduroy blazer. "
        "Shot-reverse-shot: the camera favours her, then cuts to an over-the-shoulder on him. "
        "A neon sign glows outside the window. "
        "She speaks first, clearly, from 00:00.400 to 00:02.100: \"You really drove all this way?\" "
        "Hold the beat — silence, only rain and a distant fridge hum, until 00:03.300. "
        "He answers from 00:03.300 to 00:04.900: \"I said I would.\" "
        "Natural room tone, close-mic dialogue, no music."
    ),
    "aesthetic_night_market": (
        "35mm anamorphic night market, handheld. A lantern canopy overhead, wet asphalt "
        "throwing coloured reflections, steam plumes rising between stalls. A vendor's wok "
        "erupts in flame, lighting his face from below. A woman in a translucent poncho over "
        "a mustard cardigan checks her phone, the screen lighting her chin. Motorbikes weave "
        "in the background. Shallow depth of field, warm practical lights, film grain. "
        "Ambient market sound: sizzling, chatter, a scooter passing."
    ),
}

class _DiTAdapter(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(**config)
        self.dtype = torch.bfloat16

    def forward(self, *args, attention_kwargs=None, return_dict=True, **kwargs):
        out = self.model(*args, **kwargs)
        if return_dict:
            return out
        return (out.sample, out.audio_sample)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

NVFP4_CKPT = os.environ.get("H3_NVFP4_MODEL_PATH", "/workspace/h3-model-nvfp4")

def pipeline_timesteps(pipe, steps: int, device: str) -> list[torch.Tensor]:
    """The exact timestep vector the DiT will see at each step of a `t2va` request.

    The fold frees the AdaLN projections, so it has to be given **every** timestep the
    sampler will visit — and `schedule_fold.schedule_timesteps`, which the benchmarks use,
    is a synthetic linear grid that the real sampler does not follow. Arming with it and
    then rendering raised, correctly and loudly:

        schedule_fold: timestep (0,) is not in the precomputed schedule and the adaln_proj
        weights have been freed. cached: [(333333, 166667), (666667, 333333), ...]

    That is the fold's hard-error doing its job — the alternative would have been 49 steps
    of silently wrong modulation. The real schedule comes from two schedulers with
    different shifts (12.0 video, 3.0 audio), so it is read off them here rather than
    guessed.
    """
    pipe.scheduler.set_timesteps(steps, device=device)
    pipe.audio_scheduler.set_timesteps(steps, device=device)
    out = []
    for t, at in zip(pipe.scheduler.timesteps, pipe.audio_scheduler.timesteps):
        row = torch.tensor([float(t), float(at)], dtype=torch.float32, device=device)
        out.append(torch.unique(row, sorted=True))
    return out

def _build_dit(device: str, steps: int, timesteps=None):
    from h3_inference import nvfp4, schedule_fold
    from h3_inference.loading import load_transformer

    prequantized = nvfp4.enabled() and Path(NVFP4_CKPT, "transformer").is_dir()
    t0 = time.perf_counter()
    armed = []
    if prequantized:
        model, s = nvfp4.load_transformer_nvfp4(NVFP4_CKPT, device=device)
        armed.append(f"H3_NVFP4 (pre-quantized, {s['linears_loaded']} linears)")
    else:
        model = load_transformer("/workspace/h3-model", device=device)
    print(f"[render] DiT loaded in {time.perf_counter() - t0:.1f} s, "
          f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB"
          f"{' (pre-quantized)' if prequantized else ''}", flush=True)

    if schedule_fold.enabled():
        plan = timesteps if timesteps is not None else schedule_fold.schedule_timesteps(steps)
        fold = schedule_fold.arm(model, plan, free=True)
        armed.append(f"H3_FOLD_ADALN ({len(plan)} steps, freed {fold.freed_gib:.2f} GiB)")
    if nvfp4.enabled() and not prequantized:
        s = nvfp4.quantize_blocks(model)
        armed.append(f"H3_NVFP4 ({s['linears_quantized']} linears, freed {s['freed_gib']:.2f} GiB)")
    for flag in ("H3_FUSED_ROPE", "H3_FUSED_MODULATE", "H3_FUSED_GATE"):
        if os.environ.get(flag) == "1" or os.environ.get("H3_KERNELS") == "1":
            armed.append(flag)
    print(f"[render] armed: {armed or ['(none — stock path)']}", flush=True)
    print(f"[render] DiT resident {torch.cuda.memory_allocated() / 1024**3:.2f} GiB", flush=True)
    return model, armed

@torch.no_grad()
def quantize_text_encoder(text_encoder, precision: str, device: str = "cuda") -> str:
    def footprint(m) -> float:
        return (sum(p.numel() * p.element_size() for p in m.parameters())
                + sum(b.numel() * b.element_size() for b in m.buffers())) / 1024**3

    before = footprint(text_encoder)
    if precision == "bf16":
        return f"bf16, unquantized, {before:.2f} GiB"

    import torch.nn as nn
    layers = text_encoder.model.language_model.layers
    n = 0
    t0 = time.perf_counter()
    for layer in layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                mod = getattr(parent, name, None)
                if not isinstance(mod, nn.Linear):
                    continue
                mod = mod.to(device)
                setattr(parent, name, _Fp8Linear(mod) if precision == "fp8" else _Fp4Linear(mod))
                del mod
                n += 1
    # Whatever was not quantized (embeddings, norms, the vision tower) still has to cross.
    text_encoder.to(device)
    after = footprint(text_encoder)
    return (f"{precision}, {n} linears, {before:.2f} -> {after:.2f} GiB "
            f"(freed {before - after:.2f}) in {time.perf_counter() - t0:.1f} s on {device}")


class _Fp8Linear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        w = linear.weight.data
        fmax = torch.finfo(torch.float8_e4m3fn).max
        scale = (w.abs().amax(dim=1, keepdim=True).float() / fmax).clamp(min=1e-12)
        self.register_buffer("qweight", (w.float() / scale).clamp(-fmax, fmax).to(torch.float8_e4m3fn))
        self.register_buffer("scale", scale.to(w.dtype))
        self.bias = linear.bias
        self.out_features, self.in_features = w.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.to(x.dtype) * self.scale
        return torch.nn.functional.linear(x, w, self.bias)

class _Fp4Linear(torch.nn.Module):
    """NVFP4 weight-only, block-16"""

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        from h3_inference.nvfp4 import FP4_BLOCK, _quantize

        w = linear.weight.data.contiguous()
        qdata, scale = _quantize(w.to(torch.bfloat16))
        self.register_buffer("qdata", qdata)
        self.register_buffer("scale", scale)
        self.block = FP4_BLOCK
        self.bias = linear.bias
        self.out_features, self.in_features = w.shape
        self.orig_dtype = w.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from maiku.kernels.quant.nvfp4 import _B

        hi = (self.qdata >> 4).to(torch.uint8)
        lo = (self.qdata & 0xF).to(torch.uint8)
        nib = torch.stack((lo, hi), dim=-1).reshape(self.out_features, self.in_features)
        w = _E2M1_LUT.to(x.device)[nib.long()].reshape(self.out_features, -1, self.block)
        w = (w * self.scale.to(torch.float32).unsqueeze(-1)).reshape(self.out_features, self.in_features)
        return torch.nn.functional.linear(x, w.to(x.dtype), self.bias)

_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

def _save(videos, audio, sample_rate: int, path: Path, fps: int = 24) -> None:
    import av
    import numpy as np

    frames = videos[0]
    if isinstance(frames, torch.Tensor):
        frames = frames.float().clamp(0, 1).mul(255).byte().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim == 4 and frames.shape[1] == 3:      # (F, 3, H, W) -> (F, H, W, 3)
        frames = frames.transpose(0, 2, 3, 1)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 1).__mul__(255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.height, stream.width = int(frames.shape[1]), int(frames.shape[2])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "17"}

    astream = None
    if audio is not None:
        wav = audio[0]
        wav = wav.float().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
        if wav.ndim == 1:
            wav = wav[None, :]
        astream = container.add_stream("aac", rate=int(sample_rate))
        astream.layout = "stereo" if wav.shape[0] == 2 else "mono"

    for frame in frames:
        container.mux(stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")))
    container.mux(stream.encode())

    if astream is not None:
        pcm = np.ascontiguousarray((np.clip(wav, -1, 1) * 32767).astype(np.int16))
        aframe = av.AudioFrame.from_ndarray(pcm, format="s16p", layout=astream.layout.name)
        aframe.sample_rate = int(sample_rate)
        for packet in astream.encode(aframe):
            container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)
    container.close()

def main() -> int:
    p = argparse.ArgumentParser(prog="h3_inference.render", description=__doc__)
    p.add_argument("--label", required=True, help="output subfolder: baseline | optimized")
    p.add_argument("--prompts", nargs="*", default=None,
                   help=f"named prompts from this file: {', '.join(PROMPTS)}")
    p.add_argument("--prompt", default=None,
                   help="free-text prompt; overrides --prompts. Names the output after --name.")
    p.add_argument("--name", default="custom", help="output filename stem when --prompt is used")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--frames", type=int, default=124)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--te-precision", default="fp8", choices=["bf16", "fp8", "nvfp4"])
    args = p.parse_args()

    from diffusers import ModularPipeline

    print(f"[render] label={args.label}  {args.width}x{args.height} {args.frames}f "
          f"{args.steps} steps  seed {args.seed}", flush=True)

    pipe = ModularPipeline.from_pretrained(MODEL_DIR)
    pipe.load_components(names=["text_encoder", "tokenizer", "processor", "vae",
                                "audio_vae", "scheduler", "audio_scheduler"],
                         dtype=torch.bfloat16)
    print(f"[render] pipeline components loaded, "
          f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB", flush=True)

    model, armed = _build_dit("cuda", args.steps, pipeline_timesteps(pipe, args.steps, "cuda"))
    cfg = json.loads(Path(MODEL_DIR, "transformer", "config.json").read_text())
    pipe.transformer = _DiTAdapter(model, cfg)

    te_status = quantize_text_encoder(pipe.text_encoder, args.te_precision)
    print(f"[render] text encoder: {te_status}", flush=True)

    for comp in ("vae", "audio_vae"):
        getattr(pipe, comp).to("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    resident = torch.cuda.memory_allocated() / 1024**3
    print(f"[render] all resident: {resident:.2f} GiB of {total:.1f} GiB "
          f"({total - resident:.1f} free) — no offload, nothing evicted", flush=True)

    if args.prompt:
        jobs = [(args.name, args.prompt)]
    else:
        jobs = [(n, PROMPTS[n]) for n in (args.prompts or list(PROMPTS))]

    out_dir = OUT_DIR / args.label
    for name, prompt_text in jobs:
        print(f"[render] === {name} ===", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        state = pipe(
            prompt=prompt_text,
            height=args.height, width=args.width, num_frames=args.frames,
            num_inference_steps=args.steps,
            generator=torch.Generator("cuda").manual_seed(args.seed),
            output_type="pt",
        )
        wall = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**3
        path = out_dir / f"{name}.mp4"
        audio = state.get("audio")
        if audio is not None:
            a = audio.float()
            print(f"[render] audio {tuple(a.shape)} peak {a.abs().max():.4f} "
                  f"rms {a.pow(2).mean().sqrt():.4f}", flush=True)
        _save(state.get("videos"), audio, state.get("sampling_rate") or 32_000, path)
        meta = {"prompt": name, "text": prompt_text, "label": args.label, "wall_s": round(wall, 2),
                "peak_gib": round(peak, 2), "armed": armed, "steps": args.steps,
                "size": [args.width, args.height], "frames": args.frames,
                "seed": args.seed, "te_precision": args.te_precision}
        (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[render] -> {path}   {wall:.1f} s   peak {peak:.2f} GiB", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
