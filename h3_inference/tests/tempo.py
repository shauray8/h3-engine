"""MiniMax-H3 profiling harness.

    python -m h3_inference.tempo bench    [--cell production|accept]
    python -m h3_inference.tempo rows     [--cell ...] [--blocks N]
    python -m h3_inference.tempo roofline
    python -m h3_inference.tempo phase0   [--cell ...]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time
from collections import OrderedDict

import torch

from h3_inference import constants as C
from h3_inference.constants import CELLS, Cell, packed_layout
from h3_inference.kernels import _dispatch

MODEL_PATH = os.environ.get("H3_MODEL_PATH", "/workspace/h3-model")

def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

class CudaStopwatch:
    """Wall time of a region, CUDA-synchronised on both sides."""
    def __enter__(self) -> "CudaStopwatch":
        _sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        _sync()
        self.seconds = time.perf_counter() - self._t0

class EventTimer:
    __slots__ = ("spans", "calls")

    def __init__(self) -> None:
        self.spans: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.calls = 0

    @contextlib.contextmanager
    def span(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.spans.append((start, end))
            self.calls += 1

    def total_ms(self) -> float:
        if not self.spans:
            return 0.0
        _sync()
        return sum(s.elapsed_time(e) for s, e in self.spans)

    def reset(self) -> None:
        self.spans.clear()
        self.calls = 0

def synthetic_forward_kwargs(cell: Cell, device: str, dtype: torch.dtype, seed: int = 0) -> dict:
    L = packed_layout(cell)
    g = torch.Generator(device="cpu").manual_seed(seed)
    patch_dim = C.PATCH_SIZE[0] * C.PATCH_SIZE[1] * C.PATCH_SIZE[2] * 24  # in_channels = 24

    def rnd(*shape, dt=dtype):
        return torch.randn(*shape, generator=g, dtype=torch.float32).to(device=device, dtype=dt)

    n_text, n_audio, n_video = L.num_text_rows, L.num_audio_rows, L.num_video_rows
    seq = L.seq_len

    text_indices = torch.arange(0, n_text, device=device)
    audio_indices = torch.arange(n_text, n_text + n_audio, device=device)
    video_indices = torch.arange(n_text + n_audio, seq, device=device)

    token_tags = torch.empty(seq, dtype=torch.long, device=device)
    token_tags[text_indices] = C.MODALITY_TEXT
    token_tags[audio_indices] = C.MODALITY_AUDIO
    token_tags[video_indices] = C.MODALITY_VIDEO

    timestep = torch.tensor([0.9, 0.4], dtype=torch.float32, device=device)
    timestep_indices = torch.zeros(seq, dtype=torch.long, device=device)
    timestep_indices[audio_indices] = 1

    # The real (t, h, w) grid for the video tail; the prefix rows carry the reference's
    # own convention (text ramps on t, audio continues from it). Exact values do not move
    # any timing, but the *ranges* keep the rotary math on realistic magnitudes.
    t_idx = torch.arange(L.num_latent_frames, device=device).repeat_interleave(L.grid_height * L.grid_width)
    h_idx = torch.arange(L.grid_height, device=device).repeat_interleave(L.grid_width).repeat(L.num_latent_frames)
    w_idx = torch.arange(L.grid_width, device=device).repeat(L.num_latent_frames * L.grid_height)
    position_ids = torch.zeros(seq, 3, dtype=torch.float32, device=device)
    position_ids[text_indices, 0] = torch.arange(n_text, dtype=torch.float32, device=device)
    position_ids[audio_indices, 0] = float(n_text) + torch.arange(
        n_audio, dtype=torch.float32, device=device
    ) % max(1, n_audio // C.AUDIO_CHANNELS)
    position_ids[video_indices, 0] = t_idx.float()
    position_ids[video_indices, 1] = h_idx.float()
    position_ids[video_indices, 2] = w_idx.float()

    return dict(
        hidden_states=rnd(1, n_video, patch_dim),
        audio_hidden_states=rnd(1, n_audio, 32),
        encoder_hidden_states=rnd(1, n_text, 5120),
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    )

def dit_terms(cell: Cell, config: dict) -> "OrderedDict[str, dict]":
    T = packed_layout(cell).seq_len
    d = config["hidden_size"]
    H = config["num_attention_heads"]
    Dh = config["attention_head_dim"]
    inner = H * Dh
    ffn = config["ffn_dim"]
    esz = 2  # bfloat16

    rows: "OrderedDict[str, dict]" = OrderedDict()

    def gemm(name, M, K, N, n_calls=1):
        rows[name] = {
            "row": "ffn_qkvo_gemms",
            "flops": 2 * M * K * N * n_calls,
            "bytes": (M * K + K * N + M * N) * esz * n_calls,
            "calls": n_calls,
        }

    gemm("qkv_proj", T, d, inner, 3)
    gemm("out_proj", T, inner, d, 1)
    rows["ffn"] = {
        "row": "ffn_qkvo_gemms",
        "flops": 2 * T * d * (2 * ffn) + 2 * T * ffn * d,
        "bytes": ((T * d + d * 2 * ffn + T * 2 * ffn) + (T * ffn + ffn * d + T * d)) * esz,
        "calls": 1,
    }

    # --- Attention: QK^T and PV, both 2*T*T*inner. No weights.
    rows["attention"] = {
        "row": "packed_self_attn",
        "flops": 4 * T * T * inner,
        # Flash-class: read q/k/v once, write o once. The T^2 score matrix is never
        # materialised in HBM, which is exactly the property worth auditing.
        "bytes": 4 * T * inner * esz,
        "calls": 1,
    }

    # --- Bytes-bound rows. FLOPs are negligible; what matters is the traffic.
    # norm1 + norm2: read x, write normed.
    rows["rms_norm"] = {"row": "elementwise_norm_rope", "flops": 0, "bytes": 2 * (2 * T * d) * esz, "calls": 2}
    # qk_norm on q and k, each (T, inner).
    rows["qk_norm"] = {"row": "elementwise_norm_rope", "flops": 0, "bytes": 2 * (2 * T * inner) * esz, "calls": 2}
    # RoPE on q and k. Compulsory: read once, write once (cos/sin are tiny).
    rows["rope"] = {"row": "elementwise_norm_rope", "flops": 0, "bytes": 2 * (2 * T * inner) * esz, "calls": 2}
    # ada_modulate x2: compulsory is read x + write result. The (T, d) gathers the eager
    # path materialises are NOT compulsory — that gap is the whole opportunity.
    rows["ada_modulate"] = {"row": "elementwise_norm_rope", "flops": 0, "bytes": 2 * (2 * T * d) * esz, "calls": 2}
    # gate_residual x2: read residual + y, write result.
    rows["gate_residual"] = {"row": "elementwise_norm_rope", "flops": 0, "bytes": 2 * (3 * T * d) * esz, "calls": 2}

    n_ts_rows = 6
    rows["adaln_proj"] = {
        "row": "other",
        "flops": 2 * n_ts_rows * config["time_embed_dim"] * 6 * d * 3,
        "bytes": (config["time_embed_dim"] * 6 * d * 3 + n_ts_rows * 6 * d * 3) * esz,
        "calls": 1,
    }

    return rows


def eager_modulate_bytes(cell: Cell, config: dict) -> int:
    T = packed_layout(cell).seq_len
    d = config["hidden_size"]
    esz = 2
    per_modulate = (1 + 2 + 3 + 1 + 3) * T * d * esz
    per_gate = (1 + 3 + 3) * T * d * esz
    return 2 * per_modulate + 2 * per_gate

def _time_ms(fn, iters: int = 30, warmup: int = 8) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / iters

def measure_ceilings(device: str = "cuda") -> dict:
    dt = torch.bfloat16
    results = {}

    best = 0.0
    for M, K, N in ((16384, 5376, 7168), (16384, 5376, 28672), (16384, 14336, 5376), (8192, 8192, 8192)):
        a = torch.randn(M, K, device=device, dtype=dt)
        b = torch.randn(K, N, device=device, dtype=dt)
        ms = _time_ms(lambda: torch.mm(a, b))
        tflops = 2 * M * K * N / (ms * 1e-3) / 1e12
        results[f"gemm_{M}x{K}x{N}"] = round(tflops, 1)
        best = max(best, tflops)
        del a, b
    torch.cuda.empty_cache()
    results["peak_tflops_bf16"] = round(best, 1)

    n = 1 << 28  # 256 Mi elements = 512 MB in bf16
    src = torch.empty(n, device=device, dtype=dt)
    dst = torch.empty_like(src)
    ms = _time_ms(lambda: dst.copy_(src))
    results["copy_gbs"] = round(2 * n * 2 / (ms * 1e-3) / 1e9, 1)
    del src, dst
    torch.cuda.empty_cache()

    props = torch.cuda.get_device_properties(0)
    results["device"] = props.name
    results["sm_count"] = props.multi_processor_count
    results["l2_bytes"] = getattr(props, "L2_cache_size", 0)
    return results

_INSTRUMENTED = [
    ("packed_self_attn", "packed_self_attention", "attention"),
    ("ffn_qkvo_gemms", "qkv_projection", "qkv_proj"),
    ("ffn_qkvo_gemms", "out_projection", "out_proj"),
    ("ffn_qkvo_gemms", "ffn", "ffn"),
    ("ffn_qkvo_gemms", "context_projection", "context_proj"),
    ("elementwise_norm_rope", "rms_norm", "rms_norm"),
    ("elementwise_norm_rope", "qk_norm", "qk_norm"),
    ("elementwise_norm_rope", "apply_rotary_emb", "rope"),
    # The fused replacement for qk_norm+rope. Instrumented so that turning the row on
    # cannot move its cost into "dark matter" and look like a bigger win than it is —
    # an unattributed fast path and a real speedup are indistinguishable from the total.
    ("elementwise_norm_rope", "norm_rope", "norm_rope"),
    ("elementwise_norm_rope", "norm_modulate", "norm_modulate"),
    ("elementwise_norm_rope", "ada_modulate", "ada_modulate"),
    ("elementwise_norm_rope", "gate_residual", "gate_residual"),
    ("other", "adaln_projection", "adaln_proj"),
    ("other", "token_refiner", "token_refiner"),
    ("other", "timestep_embedding", "time_embed"),
]

class RowHooks:
    """Wrap every Target-Map entry point in a CUDA-event timer."""

    def __init__(self) -> None:
        self.timers: "OrderedDict[str, EventTimer]" = OrderedDict()
        self._saved: list[tuple[object, str, object]] = []

    def __enter__(self) -> "RowHooks":
        import importlib

        for mod_name, attr, label in _INSTRUMENTED:
            module = importlib.import_module(f"h3_inference.kernels.{mod_name}")
            original = getattr(module, attr)
            timer = self.timers.setdefault(label, EventTimer())
            self._saved.append((module, attr, original))

            def make(orig, tm):
                def wrapped(*args, **kwargs):
                    with tm.span():
                        return orig(*args, **kwargs)

                return wrapped

            setattr(module, attr, make(original, timer))
        return self

    def __exit__(self, *exc: object) -> None:
        for module, attr, original in self._saved:
            setattr(module, attr, original)
        self._saved.clear()

    def reset(self) -> None:
        for timer in self.timers.values():
            timer.reset()

def _load_dit(device: str = "cuda"):
    from h3_inference.loading import load_transformer

    print(f"loading DiT from {MODEL_PATH} ...", flush=True)
    with CudaStopwatch() as sw:
        model = load_transformer(MODEL_PATH, device=device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params / 1e9:.2f} B params in {sw.seconds:.1f} s   "
          f"resident {torch.cuda.memory_allocated() / 1024**3:.1f} GiB", flush=True)
    return model

def _arm_fold(model, cell: Cell) -> str:
    from h3_inference import schedule_fold

    if not schedule_fold.enabled():
        return "off"
    fold = schedule_fold.arm(model, schedule_fold.schedule_timesteps(cell.steps), free=True)
    return f"armed — freed {fold.freed_gib:.2f} GiB, {schedule_fold.status(model)}"

def _arm_nvfp4(model) -> str:
    from h3_inference import nvfp4

    if not nvfp4.enabled():
        return "off"
    summary = nvfp4.quantize_blocks(model)
    return (f"armed — {summary['linears_quantized']} linears, freed "
            f"{summary['freed_gib']:.2f} GiB, resident {summary['resident_gib']:.2f} GiB")

def run_bench(args) -> int:
    cell = CELLS[args.cell]
    L = packed_layout(cell)
    model = _load_dit()
    kwargs = synthetic_forward_kwargs(cell, "cuda", torch.bfloat16, seed=args.seed)
    from h3_inference.schedule_fold import schedule_timesteps

    timesteps = schedule_timesteps(cell.steps) if args.walk_sigmas else None

    print(f"\ncell={cell.name}  {cell.width}x{cell.height} {cell.num_frames}f  "
          f"T={L.seq_len} (video {L.num_video_rows} + prefix {L.num_prefix_rows})")
    print(f"maiku: {_dispatch.maiku_status()}")
    print(f"fold:  {_arm_fold(model, cell)}")
    print(f"nvfp4: {_arm_nvfp4(model)}")
    resident = torch.cuda.memory_allocated() / 1024**3

    def one_eval(i: int) -> None:
        if timesteps is not None:
            kwargs["timestep"] = timesteps[i % len(timesteps)]
        model(**kwargs)

    with torch.no_grad():
        for i in range(args.warmup):
            one_eval(i)
        _sync()
        torch.cuda.reset_peak_memory_stats()

        per_step = []
        for i in range(args.reps):
            with CudaStopwatch() as sw:
                one_eval(args.warmup + i)
            per_step.append(sw.seconds * 1000.0)

    peak = torch.cuda.max_memory_allocated() / 1024**3
    median = statistics.median(per_step)
    sd = statistics.stdev(per_step) if len(per_step) > 1 else 0.0

    config = model.config
    terms = dit_terms(cell, config)
    flops_per_block = sum(t["flops"] for t in terms.values())
    total_flops = flops_per_block * config["num_layers"]

    print(f"\nper-eval  median {median:.1f} ms   sd {sd:.2f} ms ({sd / median * 100:.2f} %)   n={args.reps}")
    print(f"          min {min(per_step):.1f}  max {max(per_step):.1f}")
    print(f"achieved  {total_flops / (median * 1e-3) / 1e12:.1f} TFLOP/s over {total_flops / 1e12:.0f} TFLOP")
    print(f"peak VRAM {peak:.2f} GiB   resident weights {resident:.2f} GiB")
    print(f"\nprojected full generation ({cell.steps} steps -> {cell.steps - 1} evals): "
          f"{median * (cell.steps - 1) / 1000:.1f} s of denoise")
    if args.json:
        _write_json(args.json, {
            "cell": cell.name, "seq_len": L.seq_len, "per_eval_ms": per_step,
            "median_ms": median, "sd_ms": sd, "peak_gib": peak,
            "tflops": total_flops / (median * 1e-3) / 1e12,
        })
    return 0

def run_rows(args) -> int:
    cell = CELLS[args.cell]
    L = packed_layout(cell)
    model = _load_dit()
    kwargs = synthetic_forward_kwargs(cell, "cuda", torch.bfloat16, seed=args.seed)
    from h3_inference.schedule_fold import schedule_timesteps

    timesteps = schedule_timesteps(cell.steps) if args.walk_sigmas else None
    print(f"maiku: {_dispatch.maiku_status()}   fold: {_arm_fold(model, cell)}")
    print(f"nvfp4: {_arm_nvfp4(model)}")

    def one_eval(i: int) -> None:
        if timesteps is not None:
            kwargs["timestep"] = timesteps[i % len(timesteps)]
        model(**kwargs)

    with torch.no_grad(), RowHooks() as hooks:
        for i in range(args.warmup):
            one_eval(i)
        hooks.reset()
        _sync()
        with CudaStopwatch() as sw:
            for i in range(args.reps):
                one_eval(args.warmup + i)
        totals = {name: (t.total_ms() / args.reps, t.calls // args.reps) for name, t in hooks.timers.items()}

    wall_ms = sw.seconds * 1000.0 / args.reps
    attributed = sum(ms for ms, _ in totals.values())

    print(f"\ncell={cell.name}  T={L.seq_len}   per-eval wall {wall_ms:.1f} ms "
          f"(instrumented; events add overhead — quote `bench` for wall)")
    print(f"{'op':<16}{'row':<24}{'calls':>7}{'ms':>10}{'% attr':>9}{'us/call':>10}")
    print("-" * 76)
    row_of = {label: mod for mod, _, label in _INSTRUMENTED}
    for name, (ms, calls) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
        if calls == 0:
            continue
        print(f"{name:<16}{row_of[name]:<24}{calls:>7}{ms:>10.2f}{ms / attributed * 100:>8.1f}%"
              f"{ms / calls * 1000:>10.1f}")
    print("-" * 76)
    print(f"{'attributed':<40}{'':>7}{attributed:>10.2f}{100.0:>8.1f}%")
    print(f"{'unattributed (dark matter)':<40}{'':>7}{wall_ms - attributed:>10.2f}"
          f"{(wall_ms - attributed) / wall_ms * 100:>8.1f}%")
    print("\nDark matter is everything not inside a Target-Map entry point: the index_copy "
          "scatter,\nthe adaln_indices arithmetic, `.flatten`/`.unflatten` repacks, and launch gaps.")
    if args.json:
        _write_json(args.json, {"cell": cell.name, "wall_ms": wall_ms,
                                "rows": {k: {"ms": v[0], "calls": v[1]} for k, v in totals.items()}})
    return 0

def run_roofline(args) -> int:
    print("measuring this box's ceilings (best-of, not mean) ...")
    ceilings = measure_ceilings()
    for k, v in ceilings.items():
        print(f"  {k:<28} {v}")
    if args.json:
        _write_json(args.json, ceilings)
    return 0

def run_phase0(args) -> int:
    cell = CELLS[args.cell]
    L = packed_layout(cell)

    if args.ceilings:
        ceilings = json.loads(open(args.ceilings).read())
    else:
        print("measuring ceilings ...", flush=True)
        ceilings = measure_ceilings()
    peak_flops = ceilings["peak_tflops_bf16"] * 1e12
    bw = ceilings["copy_gbs"] * 1e9

    measured = None
    if args.rows:
        measured = json.loads(open(args.rows).read())

    from h3_inference.loading import transformer_config

    config = transformer_config(MODEL_PATH)
    n_layers = config["num_layers"]
    n_evals = cell.steps - 1
    terms = dit_terms(cell, config)

    print(f"\ncell={cell.name}  T={L.seq_len}  d={config['hidden_size']}  "
          f"{n_layers} blocks x {n_evals} evals")
    print(f"ceilings: {peak_flops / 1e12:.0f} TFLOP/s bf16, {bw / 1e9:.0f} GB/s copy\n")

    header = f"{'row':<16}{'calls':>7}{'TFLOP':>10}{'W/P s':>9}{'bytes/BW s':>12}{'FLOOR s':>10}"
    if measured:
        header += f"{'measured':>10}{'slack':>8}  binds"
    else:
        header += "  binds"
    print(header)
    print("-" * len(header))

    tot_flop = tot_wp = tot_bytes = tot_floor = tot_meas = 0.0
    for name, t in terms.items():
        scale = n_layers * n_evals
        flops = t["flops"] * scale
        byts = t["bytes"] * scale
        wp = flops / peak_flops
        bt = byts / bw
        floor = max(wp, bt)
        binds = "W/P" if wp >= bt else "bytes"
        tot_flop += flops
        tot_wp += wp
        tot_bytes += bt
        tot_floor += floor

        line = (f"{name:<16}{t['calls'] * scale:>7}{flops / 1e12:>10.1f}{wp:>9.2f}{bt:>12.2f}{floor:>10.2f}")
        if measured:
            ms = measured["rows"].get(name, {}).get("ms", 0.0)
            meas = ms * n_evals / 1000.0
            tot_meas += meas
            slack = meas / floor if floor > 0 else float("nan")
            line += f"{meas:>10.2f}{slack:>7.2f}x  {binds}"
            if floor > meas * 1.02:
                line += "   <-- FLOOR ABOVE MEASUREMENT: modelling bug"
        else:
            line += f"  {binds}"
        print(line)

    print("-" * len(header))
    line = f"{'TOTAL':<16}{'':>7}{tot_flop / 1e12:>10.1f}{tot_wp:>9.2f}{tot_bytes:>12.2f}{tot_floor:>10.2f}"
    if measured:
        line += f"{tot_meas:>10.2f}{tot_meas / tot_floor:>7.2f}x"
    print(line)

    eager = eager_modulate_bytes(cell, config) * n_layers * n_evals
    compulsory = sum(terms[k]["bytes"] for k in ("ada_modulate", "gate_residual")) * n_layers * n_evals
    print(f"\nEager gather-modulate traffic: {eager / 1e12:.2f} TB against a "
          f"{compulsory / 1e12:.2f} TB compulsory floor "
          f"({eager / compulsory:.1f}x) = {eager / bw:.1f} s vs {compulsory / bw:.1f} s.")
    print("That gap is the fused gather-modulate prize; it is bounded by the bytes column, "
          "not by\nthe FLOP column, and it is the only row where the eager form is "
          "structurally redundant.")
    if not measured:
        print("\nNo measured column: run `tempo rows --json rows.json` and pass "
              "`--rows rows.json`.\nFloors alone rank opportunities; only slack decides "
              "whether one is real.")
    return 0

def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {path}")

def main() -> int:
    parser = argparse.ArgumentParser(description="TEMPO — MiniMax-H3 profiling harness")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cell", default="production", choices=sorted(CELLS))
        p.add_argument("--reps", type=int, default=5)
        p.add_argument("--warmup", type=int, default=2)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--json", default=None)
        p.add_argument("--no-walk-sigmas", dest="walk_sigmas", action="store_false",
                       help="hold the timestep fixed across reps (the pre-2026-08-04 "
                            "behaviour; makes a table-precompute look free)")
        p.set_defaults(walk_sigmas=True)

    p = sub.add_parser("bench", help="per-eval wall time, achieved TFLOP/s, peak VRAM")
    common(p)
    p.set_defaults(fn=run_bench)

    p = sub.add_parser("rows", help="per-Target-Map-row GPU time (CUDA events)")
    common(p)
    p.set_defaults(fn=run_rows)

    p = sub.add_parser("roofline", help="measured bf16 GEMM and copy ceilings for this box")
    p.add_argument("--json", default=None)
    p.set_defaults(fn=run_roofline)

    p = sub.add_parser("phase0", help="three-floor dashboard: W/P vs bytes/BW vs measured")
    p.add_argument("--cell", default="production", choices=sorted(CELLS))
    p.add_argument("--rows", default=None, help="rows.json from `tempo rows --json`")
    p.add_argument("--ceilings", default=None, help="roofline json, to skip re-measuring")
    p.set_defaults(fn=run_phase0)

    args = parser.parse_args()
    return args.fn(args)

if __name__ == "__main__":
    sys.exit(main())
