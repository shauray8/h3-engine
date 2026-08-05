"""Kernel-level profile: where the time goes, and whether launches are costing anything.

    python -m h3_inference.kprofile --cell production --flags H3_KERNELS=1
    python -m h3_inference.kprofile --cell production --flags H3_KERNELS=1 --graph
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import sys
import time
from collections import defaultdict

import torch

from h3_inference.constants import CELLS

MODEL_PATH = os.environ.get("H3_MODEL_PATH", "/workspace/h3-model")
COPY_GBS = 1460.9
PEAK_TFLOPS = 423.5

def _set_flags(items: list[str]) -> dict[str, str]:
    from h3_inference.equiv import ALL_FLAGS

    for f in ALL_FLAGS:
        os.environ.pop(f, None)
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        if not k.startswith("H3_"):
            raise SystemExit(f"--flags only accepts H3_* keys, got {k!r}")
        os.environ[k] = v or "1"
        out[k] = v or "1"
    return out

def _load(cell, steps_for_fold: int):
    from h3_inference import nvfp4, schedule_fold
    from h3_inference.loading import load_transformer

    prequant = nvfp4.enabled() and os.path.isdir("/workspace/h3-model-nvfp4/transformer")
    if prequant:
        model, _ = nvfp4.load_transformer_nvfp4("/workspace/h3-model-nvfp4", device="cuda")
    else:
        model = load_transformer(MODEL_PATH, device="cuda")
    if schedule_fold.enabled():
        schedule_fold.arm(model, schedule_fold.schedule_timesteps(steps_for_fold), free=True)
    if nvfp4.enabled() and not prequant:
        nvfp4.quantize_blocks(model)
    return model

def _classify(name: str) -> str:
    n = name.lower()
    if "flash" in n or "fmha" in n or "attention" in n or "mha" in n:
        return "attention"
    if "scaled_mm" in n or "nvfp4" in n or "blockscaled" in n or "cutlass" in n and "gemm" in n:
        return "gemm-fp4"
    if "gemm" in n or "cutlass" in n or "sm90" in n or "sm120" in n or "ampere" in n or "cublas" in n:
        return "gemm-bf16"
    if "triton" in n or "inductor" in n:
        return "compiled-elementwise"
    if "cutile" in n or "quant" in n:
        return "cutile-quant"
    if "elementwise" in n or "vectorized" in n or "copy" in n or "cat" in n or "index" in n:
        return "torch-elementwise"
    if "reduce" in n or "norm" in n:
        return "reduction"
    return "other"

def run(args) -> int:
    cell = CELLS[args.cell]
    flags = _set_flags(args.flags)
    from h3_inference.tempo import synthetic_forward_kwargs
    from h3_inference.schedule_fold import schedule_timesteps

    print(f"cell={cell.name}  flags={flags or '(stock)'}", flush=True)
    model = _load(cell, cell.steps)
    kwargs = synthetic_forward_kwargs(cell, "cuda", torch.bfloat16, seed=0)
    timesteps = schedule_timesteps(cell.steps)

    def one(i: int):
        kwargs["timestep"] = timesteps[i % len(timesteps)]
        return model(**kwargs)

    with torch.no_grad():
        for i in range(args.warmup):
            one(i)
        torch.cuda.synchronize()
        walls = []
        for i in range(args.reps):
            t0 = time.perf_counter()
            one(i)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1000.0)
        wall_ms = statistics.median(walls)

        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            for i in range(args.prof_reps):
                one(i)
            torch.cuda.synchronize()

    from torch.autograd import DeviceType

    events = [e for e in prof.key_averages()
              if e.device_type == DeviceType.CUDA and e.self_device_time_total > 0]
    total_us = sum(e.self_device_time_total for e in events) / args.prof_reps
    total_launches = sum(e.count for e in events) / args.prof_reps

    buckets: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for e in events:
        b = buckets[_classify(e.key)]
        b[0] += e.self_device_time_total / args.prof_reps / 1000.0
        b[1] += e.count / args.prof_reps

    print(f"\nwall (clean)        {wall_ms:9.1f} ms/eval")
    print(f"GPU kernel time     {total_us / 1000:9.1f} ms/eval  over {total_launches:.0f} launches")
    gap = wall_ms - total_us / 1000
    print(f"gap (launch + idle) {gap:9.1f} ms/eval  = {gap / wall_ms * 100:.2f} % of wall"
          f"   -> {gap * 1000 / max(total_launches, 1):.1f} us/launch")
    print("\n  If the gap is under ~1 %, CUDA graphs and mega-kernels cannot pay for "
          "themselves\n  here — the launches are already hidden behind the kernels ahead of "
          "them.")

    print(f"\n{'bucket':<24}{'ms/eval':>10}{'% GPU':>8}{'launches':>10}{'us/launch':>11}")
    print("-" * 63)
    for name, (ms, count) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
        print(f"{name:<24}{ms:>10.2f}{ms / (total_us / 1000) * 100:>7.1f}%{count:>10.0f}"
              f"{ms * 1000 / max(count, 1):>11.1f}")

    print(f"\ntop {args.top} kernels by total device time")
    print(f"{'ms/eval':>9}{'calls':>8}{'us/call':>10}  kernel")
    print("-" * 100)
    for e in sorted(events, key=lambda e: -e.self_device_time_total)[: args.top]:
        ms = e.self_device_time_total / args.prof_reps / 1000.0
        calls = e.count / args.prof_reps
        print(f"{ms:>9.2f}{calls:>8.0f}{ms * 1000 / max(calls, 1):>10.1f}  {e.key[:88]}")

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"\nchrome trace -> {args.trace}")

    if args.graph:
        _cuda_graph_ab(model, kwargs, timesteps, wall_ms)
    return 0

@torch.no_grad()
def _cuda_graph_ab(model, kwargs, timesteps, eager_ms: float) -> None:
    print("\n--- CUDA graph A/B " + "-" * 46)
    static = dict(kwargs)
    static["timestep"] = timesteps[0].clone()

    fold = getattr(model, "_schedule_fold", None)
    ctx = fold.pinned(static["timestep"]) if fold is not None else contextlib.nullcontext()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            model(**static)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with ctx:
            side2 = torch.cuda.Stream()
            side2.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side2):
                for _ in range(2):
                    model(**static)
            torch.cuda.current_stream().wait_stream(side2)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                static_out = model(**static)
    except Exception as exc:
        print(f"  capture FAILED: {type(exc).__name__}: {str(exc)[:160]}")
        print("  -> this axis is unavailable; launch overhead cannot be removed by graphs")
        return

    torch.cuda.synchronize()
    reps = 5
    samples = []
    for i in range(reps + 2):
        static["timestep"].copy_(timesteps[i % len(timesteps)])
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        if i >= 2:
            samples.append((time.perf_counter() - t0) * 1000.0)
    graph_ms = statistics.median(samples)
    print(f"  eager  {eager_ms:9.1f} ms/eval")
    print(f"  graph  {graph_ms:9.1f} ms/eval   delta {graph_ms - eager_ms:+.1f} ms "
          f"({(eager_ms - graph_ms) / eager_ms * 100:+.2f} %)")
    del static_out

def main() -> int:
    p = argparse.ArgumentParser(prog="h3_inference.kprofile", description=__doc__)
    p.add_argument("--cell", default="production", choices=sorted(CELLS))
    p.add_argument("--flags", nargs="*", default=[])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--prof-reps", type=int, default=2)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--trace", default=None, help="write a chrome trace here")
    p.add_argument("--graph", action="store_true",
                   help="also capture a CUDA graph and A/B it — the upper bound on any "
                        "launch-overhead win")
    return run(p.parse_args())

if __name__ == "__main__":
    sys.exit(main())
