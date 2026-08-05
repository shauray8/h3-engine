"""
Usage
    python -m h3_inference.accept run --tag base  --reps 8
    python -m h3_inference.accept run --tag base2 --reps 8
    python -m h3_inference.accept ab base base2        # MUST be INCONCLUSIVE

    python -m h3_inference.accept run --tag rope --reps 8 --env H3_KERNELS=1 H3_FUSED_ROPE=1
    python -m h3_inference.accept ab base rope
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

RESULTS = Path(os.environ.get("H3_ACCEPT_DIR", "/workspace/H3/accept_results"))

NOISE_FLOOR_PCT = float(os.environ.get("H3_NOISE_FLOOR_PCT", "0.7"))

SWITCHES = (
    "H3_KERNELS",
    "H3_TL_ATTN",
    "H3_FUSED_ROPE",
    "H3_FUSED_MODULATE",
    "H3_FUSED_GATE",
    "H3_FUSED_SWIGLU",
    "H3_FOLD_ADALN",
    "H3_NVFP4",
)
DROPPED = ("H3_NO_MAIKU",)

def _child(tag: str, cell_name: str, reps: int, warmup: int) -> int:
    import torch

    from h3_inference.constants import CELLS, packed_layout
    from h3_inference.kernels import _dispatch
    from h3_inference.loading import load_transformer
    from h3_inference.tempo import MODEL_PATH, CudaStopwatch, synthetic_forward_kwargs

    cell = CELLS[cell_name]
    layout = packed_layout(cell)
    n_evals = cell.steps - 1

    model = load_transformer(MODEL_PATH, device="cuda")
    kwargs = synthetic_forward_kwargs(cell, "cuda", torch.bfloat16, seed=0)

    sigmas = torch.linspace(1.0, 0.0, cell.steps, device="cuda", dtype=torch.float32)

    def one_denoise() -> list[float]:
        per_eval = []
        for i in range(n_evals):
            kwargs["timestep"] = torch.stack([sigmas[i], sigmas[i] * 0.5])
            with CudaStopwatch() as sw:
                model(**kwargs)
            per_eval.append(sw.seconds * 1000.0)
        return per_eval

    print(f"  [{tag}] maiku: {_dispatch.maiku_status()}", flush=True)
    print(f"  [{tag}] cell={cell.name} T={layout.seq_len} evals/rep={n_evals}", flush=True)

    with torch.no_grad():
        ramp = []
        for i in range(warmup):
            ev = one_denoise()
            ramp.append(sum(ev) / 1000.0)
            print(f"  [{tag}] warmup {i + 1}/{warmup}: {ramp[-1]:.3f} s (discarded)", flush=True)
        torch.cuda.empty_cache()

        out = []
        for i in range(reps):
            torch.cuda.reset_peak_memory_stats()
            ev = one_denoise()
            out.append({
                "evals_s": sum(ev) / 1000.0,
                "per_eval_ms": ev,
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9,
            })
            print(f"  [{tag}] rep {i + 1}/{reps}: {out[-1]['evals_s']:.3f} s", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "workload": {
            "cell": cell.name,
            "width": cell.width,
            "height": cell.height,
            "num_frames": cell.num_frames,
            "steps": cell.steps,
            "n_evals": n_evals,
            "seq_len": layout.seq_len,
            "model_path": MODEL_PATH,
        },
        "warmup_discarded": warmup,
        "warmup_ramp_s": ramp,
        "maiku": _dispatch.maiku_status(),
        "env": {k: os.environ.get(k, "") for k in SWITCHES},
        "reps": out,
    }
    (RESULTS / f"{tag}.json").write_text(json.dumps(payload, indent=2))
    med = statistics.median(r["evals_s"] for r in out)
    print(f"[{tag}] median {med:.3f} s over {reps} reps -> {RESULTS / f'{tag}.json'}")
    return 0

def run(args: argparse.Namespace) -> int:
    env = dict(os.environ)
    for k in SWITCHES:
        env[k] = "0"
    for k in DROPPED:
        env.pop(k, None)
    for kv in args.env or []:
        k, _, v = kv.partition("=")
        if k not in SWITCHES and not k.startswith("H3_"):
            raise SystemExit(f"--env {kv}: not an H3_* flag. Configs are defined by H3_* only.")
        env[k] = v
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") or str(Path(__file__).resolve().parents[1])

    on = [f"{k}={v}" for k, v in (kv.partition("=")[::2] for kv in args.env or [])]
    print(f"[accept] tag={args.tag} cell={args.cell} reps={args.reps} "
          f"warmup={args.warmup} config={on or '(all off)'}")
    return subprocess.call(
        [sys.executable, "-m", "h3_inference.accept", "_child",
         "--tag", args.tag, "--cell", args.cell,
         "--reps", str(args.reps), "--warmup", str(args.warmup)],
        env=env,
    )

def _bootstrap_ci(a: list[float], b: list[float], n: int = 20000, alpha: float = 0.05):
    rng = random.Random(0xC0FFEE)
    diffs = []
    for _ in range(n):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(statistics.median(rb) - statistics.median(ra))
    diffs.sort()
    return diffs[int(alpha / 2 * n)], diffs[int((1 - alpha / 2) * n)]


def _ramp_verdict(reps: list[float]) -> str:
    n = len(reps)
    if n < 4:
        return "n<4, cannot assess"
    up = sum(1 for i in range(n) for j in range(i + 1, n) if reps[j] > reps[i])
    frac = up / (n * (n - 1) / 2)
    if frac >= 0.80:
        return f"STILL RAMPING (monotone fraction {frac:.2f}) — increase --warmup"
    if frac <= 0.20:
        return f"still settling downward (monotone fraction {frac:.2f}) — increase --warmup"
    return f"settled (monotone fraction {frac:.2f})"

def compare(args: argparse.Namespace) -> int:
    pa = json.loads((RESULTS / f"{args.a}.json").read_text())
    pb = json.loads((RESULTS / f"{args.b}.json").read_text())
    if pa["workload"] != pb["workload"]:
        print("REFUSING to compare: the two runs used different workloads.")
        print(f"  {args.a}: {pa['workload']}\n  {args.b}: {pb['workload']}")
        return 2

    wa = [r["evals_s"] for r in pa["reps"]]
    wb = [r["evals_s"] for r in pb["reps"]]
    ma, mb = statistics.median(wa), statistics.median(wb)
    d = mb - ma
    lo, hi = _bootstrap_ci(wa, wb)
    sda = statistics.stdev(wa) if len(wa) > 1 else 0.0
    sdb = statistics.stdev(wb) if len(wb) > 1 else 0.0
    spread = max(sda, sdb)

    W = "-" * 78
    wl = pa["workload"]
    print("\n" + W)
    print(f" ACCEPTANCE TEST — DiT wall only            {args.a}  ->  {args.b}")
    print(W)
    print(f" workload: cell={wl['cell']}  {wl['width']}x{wl['height']}  {wl['num_frames']}f  "
          f"T={wl['seq_len']}  {wl['n_evals']} evals/rep")
    for tag, p in ((args.a, pa), (args.b, pb)):
        on = {k: v for k, v in p["env"].items() if v not in ("0", "")}
        print(f" {tag:<12} config={on or '(all off)'}  maiku={p.get('maiku', '?')}")
    print(W)
    for tag, w in ((args.a, wa), (args.b, wb)):
        sd = statistics.stdev(w) if len(w) > 1 else 0.0
        print(f" {tag:<12} n={len(w)}  median {statistics.median(w):8.3f} s   "
              f"mean {statistics.mean(w):8.3f} +/- {sd:.3f}   min {min(w):8.3f}   max {max(w):8.3f}")
    print(W)
    print(f" delta (median)   {d:+.3f} s   ({d / ma * 100:+.2f} %)     speedup {ma / mb:.4f}x")
    print(f" 95% bootstrap CI [{lo:+.3f}, {hi:+.3f}] s   "
          f"({lo / ma * 100:+.2f} %, {hi / ma * 100:+.2f} %)")
    print(f" run-to-run sd    {spread:.3f} s  ({spread / ma * 100:.2f} % of baseline)")
    print(W)
    print(f" warmup: {pa['warmup_discarded']} discarded")
    print(f"   {args.a:<10} {_ramp_verdict(wa)}")
    print(f"   {args.b:<10} {_ramp_verdict(wb)}")
    resolution = 2 * spread / ma * 100
    print(f" resolution: this pair cannot resolve effects below ~{resolution:.1f} %")
    print(W)

    delta_pct = abs(d) / ma * 100
    floor = args.floor if args.floor is not None else NOISE_FLOOR_PCT
    print(f" noise floor      {floor:.2f} % (measured between-process; see NOISE_FLOOR_PCT)")
    print(W)

    if lo <= 0.0 <= hi:
        need = max(4, int((spread / max(abs(d), 1e-9)) ** 2 * 8)) if d else 999
        print(" VERDICT: INCONCLUSIVE — the CI straddles zero. This change is not")
        print(f"          distinguishable from run-to-run noise at n={len(wa)}/{len(wb)}.")
        print(f"          Either it does nothing, or you need ~{min(need, 60)} reps to resolve it.")
        print("          A tight INCONCLUSIVE and a real null look identical. Check the ramp")
        print("          verdict above before believing this is a null.")
    elif delta_pct < floor:
        # The CI excluding zero is necessary but NOT sufficient: the null pair proved this
        # harness produces CI-excludes-zero results at ~0.5 % with no change at all.
        print(f" VERDICT: INCONCLUSIVE — |delta| {delta_pct:.2f} % is below the measured")
        print(f"          between-process noise floor of {floor:.2f} %. The CI excludes zero,")
        print("          but the null pair (identical config, two processes) also excluded")
        print("          zero at this magnitude. The CI resamples within each arm and cannot")
        print("          see process-to-process drift, so it is not evidence at this scale.")
        print("          To resolve: re-run both arms and require the sign to be reproducible,")
        print("          or measure at the production cell where the effect is larger.")
    elif hi < 0:
        print(f" VERDICT: ACCEPT — {args.b} is faster by {-d:.3f} s "
              f"({-d / ma * 100:.2f} %), CI excludes zero and clears the {floor:.2f} % floor.")
    else:
        print(f" VERDICT: REJECT — {args.b} is SLOWER by {d:.3f} s "
              f"({d / ma * 100:.2f} %), CI excludes zero and clears the {floor:.2f} % floor.")
    print(W)

    pka = statistics.median(r["peak_gb"] for r in pa["reps"])
    pkb = statistics.median(r["peak_gb"] for r in pb["reps"])
    print(" DIAGNOSTIC ONLY, never the accept criterion:")
    print(f"   {'peak VRAM GB':<18}{pka:10.2f}{pkb:10.2f}{pkb - pka:+10.2f}")
    mea = statistics.median(statistics.median(r["per_eval_ms"]) for r in pa["reps"])
    meb = statistics.median(statistics.median(r["per_eval_ms"]) for r in pb["reps"])
    print(f"   {'per-eval ms':<18}{mea:10.1f}{meb:10.1f}{meb - mea:+10.1f}")
    print(W + "\n")
    return 0

def main() -> int:
    p = argparse.ArgumentParser(description="H3 end-to-end DiT acceptance gate")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="time one config in a clean process (warmup + N reps)")
    pr.add_argument("--tag", required=True)
    pr.add_argument("--cell", default="accept", choices=("accept", "prod_ab", "production"))
    pr.add_argument("--reps", type=int, default=5)
    pr.add_argument("--warmup", type=int, default=2,
                    help="discarded reps. This box ramps for ~2-3 reps; 2 suffices on prod_ab")
    pr.add_argument("--env", nargs="*", help="H3_* flags defining this config")
    pr.set_defaults(func=run)

    pc = sub.add_parser("_child", help=argparse.SUPPRESS)
    pc.add_argument("--tag", required=True)
    pc.add_argument("--cell", default="accept")
    pc.add_argument("--reps", type=int, default=8)
    pc.add_argument("--warmup", type=int, default=3)
    pc.set_defaults(func=lambda a: _child(a.tag, a.cell, a.reps, a.warmup))

    pab = sub.add_parser("ab", help="compare two stored runs and emit a verdict")
    pab.add_argument("a")
    pab.add_argument("b")
    pab.add_argument("--floor", type=float, default=None,
                     help=f"between-process noise floor, %%. Default {NOISE_FLOOR_PCT}, measured.")
    pab.set_defaults(func=compare)

    args = p.parse_args()
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
