from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/workspace/h3-model/transformer"
DST = "/workspace/h3-model-nvfp4/transformer"

QUANT_SUFFIXES = (
    ".attn.to_q.weight",
    ".attn.to_k.weight",
    ".attn.to_v.weight",
    ".attn.to_out.0.weight",
    ".ff.net.0.proj.weight",
    ".ff.net.2.weight",
)

def is_quantizable(key: str) -> bool:
    return key.startswith("transformer_blocks.") and key.endswith(QUANT_SUFFIXES)

@torch.no_grad()
def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 `(N, K)` -> (packed uint8 `(N, K/2)`, swizzled scale as uint8)."""
    from h3_inference.nvfp4 import _quantize, _swizzle

    qdata, scale = _quantize(w.to(torch.bfloat16).cuda().contiguous())
    return qdata.cpu(), _swizzle(scale).view(torch.uint8).cpu()

@torch.no_grad()
def run(src: str, dst: str, verify: bool) -> int:
    src_dir, dst_dir = Path(src), Path(dst)
    dst_dir.mkdir(parents=True, exist_ok=True)

    index_path = next(src_dir.glob("*.safetensors.index.json"), None)
    shards = ([src_dir / n for n in sorted(set(json.loads(index_path.read_text())["weight_map"].values()))]
              if index_path else sorted(src_dir.glob("*.safetensors")))

    weight_map: dict[str, str] = {}
    n_quant = n_copy = 0
    bytes_in = bytes_out = 0
    worst_cos = 1.0
    t0 = time.perf_counter()

    for i, shard in enumerate(shards, 1):
        out: dict[str, torch.Tensor] = {}
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                bytes_in += t.numel() * t.element_size()
                if is_quantizable(key):
                    qdata, scale = quantize_weight(t)
                    out[key + "_nvfp4"] = qdata
                    out[key + "_nvfp4_scale"] = scale
                    n_quant += 1
                    if verify:
                        worst_cos = min(worst_cos, _verify_one(t, qdata, scale))
                else:
                    out[key] = t
                    n_copy += 1
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        for key, t in out.items():
            weight_map[key] = name
            bytes_out += t.numel() * t.element_size()
        save_file(out, str(dst_dir / name), metadata={"format": "pt", "h3_quant": "nvfp4"})
        print(f"  [{i}/{len(shards)}] {name}  {sum(t.numel() * t.element_size() for t in out.values()) / 2**30:.2f} GiB",
              flush=True)
        del out

    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": bytes_out, "h3_quant": "nvfp4"},
                    "weight_map": weight_map}, indent=2)
    )
    shutil.copy(src_dir / "config.json", dst_dir / "config.json")

    print(f"\n{n_quant} weights quantized, {n_copy} copied through unchanged")
    print(f"{bytes_in / 2**30:.2f} GiB -> {bytes_out / 2**30:.2f} GiB "
          f"({bytes_in / bytes_out:.2f}x smaller) in {time.perf_counter() - t0:.0f} s")
    if verify:
        print(f"worst per-weight cos vs bf16: {worst_cos:.5f}")
        if worst_cos < 0.98:
            print("  FAILED")
            return 1
    print(f"-> {dst_dir}")
    return 0


@torch.no_grad()
def _verify_one(w: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor) -> float:
    """One real GEMM through the stored operands, against bf16. Returns cos."""
    from h3_inference.nvfp4 import M_ALIGN, NVFP4Weight

    wb = w.to(torch.bfloat16).cuda()
    x = torch.randn(M_ALIGN, wb.shape[1], device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, wb)

    stored = NVFP4Weight.__new__(NVFP4Weight)
    stored.out_features, stored.in_features = wb.shape
    stored.dtype = torch.bfloat16
    stored.qdata_t = qdata.cuda().view(torch.float4_e2m1fn_x2).t()
    stored.scale = scale.cuda().view(torch.float8_e4m3fn)

    from h3_inference.nvfp4 import QuantizedActivation

    got = QuantizedActivation(x).mm(stored)
    return float(torch.nn.functional.cosine_similarity(
        ref.float().flatten(), got.float().flatten(), dim=0))

def main() -> int:
    p = argparse.ArgumentParser(prog="h3_inference.quantize_checkpoint", description=__doc__)
    p.add_argument("--src", default=SRC)
    p.add_argument("--dst", default=DST)
    p.add_argument("--verify", action="store_true",
                   help="run a real _scaled_mm per quantized weight and report the worst cos")
    args = p.parse_args()
    print(f"quantizing {args.src} -> {args.dst}", flush=True)
    return run(args.src, args.dst, args.verify)

if __name__ == "__main__":
    sys.exit(main())
