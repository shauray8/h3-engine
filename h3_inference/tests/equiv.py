"""
    python -m h3_inference.equiv run  --flags H3_FUSED_ROPE=1 H3_FUSED_MODULATE=1
    python -m h3_inference.equiv run  --flags H3_KERNELS=1 --cell accept
    python -m h3_inference.equiv ref  --cell micro                # refresh the baseline
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from h3_inference import nvfp4, schedule_fold
from h3_inference.constants import CELLS, Cell
from h3_inference.kernels import _dispatch
from h3_inference.schedule_fold import schedule_timesteps

MODEL_PATH = os.environ.get("H3_MODEL_PATH", "/workspace/h3-model")
REF_DIR = Path(os.environ.get("H3_EQUIV_REF_DIR", "/workspace/H3/equiv_refs"))

ALL_FLAGS = (
    "H3_KERNELS",
    "H3_FUSED_ROPE",
    "H3_FUSED_MODULATE",
    "H3_FUSED_GATE",
    "H3_FUSED_SWIGLU",
    "H3_FOLD_ADALN",
    "H3_TL_ATTN",
    "H3_NVFP4",
)
PASS_DB = 35.0
FAIL_DB = 20.0

MICRO = Cell(
    name="micro",
    width=320,
    height=192,
    num_frames=124,
    steps=50,
    note="drift-only cell; 3,171 packed rows. NOT a timing cell attention is a minority here",
)

EQUIV_CELLS = {**CELLS, MICRO.name: MICRO}

def _sigma_grid(steps: int, device: str) -> torch.Tensor:
    """`steps` points from 1.0 down to 0.0 inclusive -> `steps - 1` model evaluations."""
    return torch.linspace(1.0, 0.0, steps, dtype=torch.float32, device=device)

def _trajectory(model, kwargs: dict, steps: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Flow-match Euler over the sigma grid; returns the final (video, audio) latents.

    ``x <- x + (sigma_next - sigma) * v`` with ``v`` the model's velocity, which is the
    update the released scheduler performs. The timestep vector per evaluation comes from
    :func:`schedule_fold.schedule_timesteps`, the one definition `tempo` and `accept` also
    use, so all three harnesses drive the AdaLN table identically.
    """
    sigmas = _sigma_grid(steps, device)
    timesteps = schedule_timesteps(steps, device)
    video = kwargs["hidden_states"].clone()
    audio = kwargs["audio_hidden_states"].clone()

    for i in range(steps - 1):
        step_kwargs = dict(kwargs)
        step_kwargs["hidden_states"] = video
        step_kwargs["audio_hidden_states"] = audio
        step_kwargs["timestep"] = timesteps[i]

        with torch.no_grad():
            out = model(**step_kwargs)

        dt = float(sigmas[i + 1]) - float(sigmas[i])
        video = video + dt * out.sample.to(video.dtype)
        audio = audio + dt * out.audio_sample.to(audio.dtype)

    return video, audio

def _drift(ref: torch.Tensor, cand: torch.Tensor) -> dict:
    a = ref.float().flatten()
    b = cand.float().flatten()
    diff = b - a
    mse = float(torch.mean(diff * diff))
    peak = float(a.max() - a.min())
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(peak) - 10.0 * math.log10(mse)
    return {
        "max_abs": float(diff.abs().max()),
        "rel_l2_pct": 100.0 * float(diff.norm() / a.norm()),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "psnr_db": psnr,
        "ref_rms": float(a.pow(2).mean().sqrt()),
    }

def _set_flags(flags: dict[str, str]) -> None:
    for f in ALL_FLAGS:
        os.environ.pop(f, None)
    os.environ.update(flags)

def _parse_flags(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--flags takes KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key.startswith("H3_"):
            raise SystemExit(f"--flags only accepts H3_* keys, got {key!r}")
        out[key] = value
    return out

def _run_arm(model, cell: Cell, steps: int, seed: int, flags: dict[str, str], device: str):
    """One trajectory under one flag set. Returns (video, audio, armed-counts, seconds)."""
    from h3_inference.tempo import synthetic_forward_kwargs

    _set_flags(flags)
    _dispatch.armed_counts(reset=True)
    kwargs = synthetic_forward_kwargs(cell, device=device, dtype=torch.bfloat16, seed=seed)

    folded = schedule_fold.enabled()
    if folded:
        schedule_fold.arm(model, schedule_timesteps(steps, device), free=False)

    quantized = nvfp4.enabled()
    if quantized:
        summary = nvfp4.quantize_blocks(model)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    video, audio = _trajectory(model, kwargs, steps, device)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0

    armed = _dispatch.armed_counts()
    if folded:
        armed["H3_FOLD_ADALN"] = schedule_fold.status(model)
        schedule_fold.uninstall(model)
    if quantized:
        armed["H3_NVFP4"] = (f"{summary['linears_quantized']} linears, "
                             f"freed {summary['freed_gib']:.2f} GiB, {nvfp4.report(model)}")
    return video, audio, armed, secs

def _ref_path(cell: Cell, steps: int, seed: int) -> Path:
    return REF_DIR / f"{cell.name}_s{steps}_seed{seed}.pt"

def _load_dit(device: str = "cuda"):
    from h3_inference.loading import load_transformer

    print(f"loading DiT from {MODEL_PATH} ...", flush=True)
    t0 = time.perf_counter()
    model = load_transformer(MODEL_PATH, device=device)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    return model

def _baseline(model, cell: Cell, steps: int, seed: int, device: str, refresh: bool):
    """The baseline arm, from cache when it is there and the cache is trustworthy."""
    path = _ref_path(cell, steps, seed)
    if path.exists() and not refresh:
        blob = torch.load(path, map_location=device)
        print(f"baseline: cached {path.name} (from {blob['written']})")
        return blob["video"], blob["audio"], None

    video, audio, armed, secs = _run_arm(model, cell, steps, seed, {}, device)
    if armed:
        raise SystemExit(f"baseline arm had rows armed: {armed} — refusing to cache it")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video": video,
            "audio": audio,
            "cell": cell.name,
            "steps": steps,
            "seed": seed,
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        path,
    )
    print(f"baseline: computed in {secs:.1f} s -> {path.name}")
    return video, audio, secs

def run(args) -> int:
    cell = EQUIV_CELLS[args.cell]
    steps = args.steps or cell.steps
    device = "cuda"
    flags = _parse_flags(args.flags)
    if not flags:
        raise SystemExit("nothing to compare — pass --flags H3_...=1 (or use the `ref` command)")

    model = _load_dit(device)
    print(f"cell={cell.name}  {cell.width}x{cell.height} {cell.num_frames}f  "
          f"steps={steps} -> {steps - 1} evals   maiku: {_dispatch.maiku_status()}")

    ref_video, ref_audio, _ = _baseline(model, cell, steps, args.seed, device, args.refresh)
    cand_video, cand_audio, armed, secs = _run_arm(model, cell, steps, args.seed, flags, device)
    print(f"candidate: {' '.join(f'{k}={v}' for k, v in flags.items())}  ({secs:.1f} s)")
    named = {k for k in flags if k in ALL_FLAGS and flags[k] == "1"}
    if not armed:
        print("\n  FAIL — no row armed during the candidate arm. The flags did not take "
              "effect;\n         this run compared the baseline against itself.")
        return 1
    print("  armed: " + ", ".join(f"{k}x{v}" for k, v in sorted(armed.items())))
    if named and named != {"H3_KERNELS"} and not (named & set(armed)):
        print(f"\n  FAIL — none of the requested rows {sorted(named)} armed; "
              f"only {sorted(armed)} did.")
        return 1

    rows = {"video": _drift(ref_video, cand_video), "audio": _drift(ref_audio, cand_audio)}

    print(f"\n  {'modality':<9} {'max|d|':>10} {'rel L2':>9} {'cosine':>12} {'PSNR':>10}")
    print("  " + "-" * 54)
    for name, d in rows.items():
        psnr = "  exact" if d["psnr_db"] == float("inf") else f"{d['psnr_db']:8.2f} dB"
        print(f"  {name:<9} {d['max_abs']:10.3e} {d['rel_l2_pct']:8.3f}% "
              f"{d['cosine']:12.7f} {psnr:>10}")

    worst = min(d["psnr_db"] for d in rows.values())
    exact = all(d["max_abs"] == 0.0 for d in rows.values())

    print()
    if exact:
        print("  BIT-IDENTICAL — rows armed and the trajectory is unchanged to the last bit.")
        verdict = 0
    elif worst >= PASS_DB:
        print(f"  PASS — {worst:.2f} dB, the reassociation class. Above {PASS_DB:.0f} dB the "
              f"render is a formality,\n         but it is still owed before default-on.")
        verdict = 0
    elif worst >= FAIL_DB:
        print(f"  REVIEW — {worst:.2f} dB, between {FAIL_DB:.0f} and {PASS_DB:.0f} dB. This is the "
              f"band where the numbers rank\n           changes and do not decide them. Render "
              f"it and watch it before landing.")
        verdict = 0
    else:
        print(f"  FAIL — {worst:.2f} dB, below the {FAIL_DB:.0f} dB floor. The candidate is not "
              f"tracking the\n         baseline trajectory; treat this as a bug, not a trade.")
        verdict = 1

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"cell": cell.name, "steps": steps, "seed": args.seed, "flags": flags,
             "armed": armed, "drift": rows, "worst_psnr_db": worst}, indent=2) + "\n")
        print(f"  -> {args.json}")
    return verdict

def ref(args) -> int:
    """Compute (or recompute) the baseline trajectory and cache it."""
    cell = EQUIV_CELLS[args.cell]
    steps = args.steps or cell.steps
    model = _load_dit("cuda")
    print(f"cell={cell.name}  steps={steps} -> {steps - 1} evals")
    _baseline(model, cell, steps, args.seed, "cuda", refresh=True)
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(prog="h3_inference.equiv", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name, fn in (("run", run), ("ref", ref)):
        p = sub.add_parser(name)
        p.add_argument("--cell", default="micro", choices=sorted(EQUIV_CELLS))
        p.add_argument("--steps", type=int, default=None,
                       help="sigma-grid points; evaluations are one fewer. Defaults to the cell's.")
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(fn=fn)
        if name == "run":
            p.add_argument("--flags", nargs="*", default=[], metavar="H3_X=1")
            p.add_argument("--refresh", action="store_true", help="recompute the cached baseline")
            p.add_argument("--json", default=None)

    args = parser.parse_args()
    return args.fn(args)

if __name__ == "__main__":
    sys.exit(main())
