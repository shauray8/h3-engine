from __future__ import annotations

import os
from typing import Iterable

import torch
from torch import nn

FP4_BLOCK = 16
FP4_MAX = 6.0
# `_scaled_mm`'s swizzled scale buffer rounds M to this; the activation is padded to match.
M_ALIGN = 128

def enabled() -> bool:
    return os.environ.get("H3_NVFP4") == "1" or os.environ.get("H3_KERNELS") == "1"

def _pad_m(m: int) -> int:
    return ((m + M_ALIGN - 1) // M_ALIGN) * M_ALIGN

def _quantize(x: torch.Tensor):
    # bf16 `(M, K)` -> (packed fp4 `(M, K//2)` uint8, e4m3 scales `(M, K//16)`).
    from maiku.kernels.quant.nvfp4 import quantize_nvfp4
    return quantize_nvfp4(x)

@torch.no_grad()
def _quantize_padded(x: torch.Tensor, mpad: int):
    from math import ceil

    import cuda.tile as ct
    from maiku.kernels.quant.nvfp4 import _TK, _TM, _nvfp4_quant_fwd

    k = x.shape[-1]
    xf = x.reshape(-1, k).contiguous()
    m = xf.shape[0]
    out = _PAD.data(mpad, k // 2, xf.device, torch.uint8)
    scale = _PAD.scale(mpad, k // FP4_BLOCK, xf.device, torch.float8_e4m3fn)

    nsm = torch.cuda.get_device_properties(xf.device).multi_processor_count
    grid = (min(nsm * 4, ceil(m / _TM)),)
    ct.launch(torch.cuda.current_stream(), grid, _nvfp4_quant_fwd,
              (xf, out, scale, _TM, _TK, k))
    return out, scale

def _swizzle(scale: torch.Tensor) -> torch.Tensor:
    from torchao.prototype.mx_formats.utils import to_blocked
    return to_blocked(scale)

class NVFP4Weight:
    __slots__ = ("qdata_t", "scale", "in_features", "out_features", "dtype")

    def __init__(self, weight: torch.Tensor):
        self.out_features, self.in_features = weight.shape
        self.dtype = weight.dtype
        qdata, scale = _quantize(weight.contiguous())
        self.qdata_t = qdata.view(torch.float4_e2m1fn_x2).t()   # -> (K, N) view for b
        self.scale = _swizzle(scale)

    def nbytes(self) -> int:
        return self.qdata_t.numel() + self.scale.numel() * self.scale.element_size()

class _PadCache:
    # Per-(rows, cols) padded fp4 buffers, allocated once and reused.
    def __init__(self) -> None:
        self._data: dict[tuple, torch.Tensor] = {}
        self._scale: dict[tuple, torch.Tensor] = {}

    def data(self, rows: int, cols: int, device, dtype) -> torch.Tensor:
        key = (rows, cols, device, dtype)
        buf = self._data.get(key)
        if buf is None:
            buf = torch.zeros(rows, cols, device=device, dtype=dtype)
            self._data[key] = buf
        return buf

    def scale(self, rows: int, cols: int, device, dtype) -> torch.Tensor:
        key = (rows, cols, device, dtype)
        buf = self._scale.get(key)
        if buf is None:
            buf = torch.zeros(rows, cols, device=device, dtype=dtype)
            self._scale[key] = buf
        return buf

    def clear(self) -> None:
        self._data.clear()
        self._scale.clear()

_PAD = _PadCache()

class QuantizedActivation:
    __slots__ = ("qdata", "scale_blocked", "rows", "shape", "dtype")

    def __init__(self, x: torch.Tensor):
        self.shape = x.shape
        self.dtype = x.dtype
        x2 = x.reshape(-1, x.shape[-1])
        self.rows = x2.shape[0]
        mpad = _pad_m(self.rows)
        try:
            qdata, scale = _quantize_padded(x2, mpad)
        except Exception:
            qdata, scale = _quantize(x2)
            if mpad != self.rows:
                buf = _PAD.data(mpad, qdata.shape[1], qdata.device, qdata.dtype)
                buf[: self.rows] = qdata
                qdata = buf
                sbuf = _PAD.scale(mpad, scale.shape[1], scale.device, scale.dtype)
                sbuf[: self.rows] = scale
                scale = sbuf
        self.qdata = qdata.view(torch.float4_e2m1fn_x2)
        self.scale_blocked = _swizzle(scale)

    def mm(self, w: NVFP4Weight) -> torch.Tensor:
        out = torch._scaled_mm(self.qdata, w.qdata_t, self.scale_blocked, w.scale,
                               out_dtype=self.dtype)
        return out[: self.rows].reshape(*self.shape[:-1], w.out_features)


def nvfp4_linear(w: NVFP4Weight, x: torch.Tensor) -> torch.Tensor:
    """`x @ w.T` in NVFP4. `x` is `(..., K)` bf16; returns `(..., N)` bf16."""
    return QuantizedActivation(x).mm(w)

def _block_linears(block: nn.Module) -> Iterable[tuple[nn.Module, str, nn.Linear]]:
    attn = block.attn
    yield attn, "to_q", attn.to_q
    yield attn, "to_k", attn.to_k
    yield attn, "to_v", attn.to_v
    yield attn.to_out, "0", attn.to_out[0]
    ff = block.ff
    yield ff.net[0], "proj", ff.net[0].proj      # SwiGLU gate+up, the 2x14336 one
    yield ff.net, "2", ff.net[2]                 # down-projection

class _QuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        if linear.bias is not None:
            raise NotImplementedError("NVFP4 path does not carry a bias term")
        self.weight_nvfp4 = NVFP4Weight(linear.weight.data)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return nvfp4_linear(self.weight_nvfp4, x)

@torch.no_grad()
def quantize_blocks(model, self_check: bool = True) -> dict:
    if self_check and os.environ.get("H3_NVFP4_SELFCHECK") != "0":
        run_self_check(model)

    before = torch.cuda.memory_allocated()
    replaced = 0
    for block in model.transformer_blocks:
        for parent, attr, linear in list(_block_linears(block)):
            q = _QuantLinear(linear)
            if isinstance(parent, (nn.ModuleList, nn.Sequential)):
                parent[int(attr)] = q
            else:
                setattr(parent, attr, q)
            replaced += 1
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    return {
        "linears_quantized": replaced,
        "freed_gib": (before - after) / 1024**3,
        "resident_gib": after / 1024**3,
    }

@torch.no_grad()
def run_self_check(model, tol: float = 0.98) -> float:
    linear = model.transformer_blocks[0].attn.to_q
    w = linear.weight.data
    x = torch.randn(M_ALIGN * 2, w.shape[1], device=w.device, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, w)
    got = nvfp4_linear(NVFP4Weight(w), x)
    cos = float(torch.nn.functional.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0))
    if cos < tol:
        raise RuntimeError(
            f"NVFP4 self-check failed: cos {cos:.5f} < {tol}. "
        )
    return cos


@torch.no_grad()
def load_transformer_nvfp4(model_path: str, device: str = "cuda"):
    import json
    from pathlib import Path

    from safetensors import safe_open
    from h3_inference.transformer.model import H3Transformer3DModel

    folder = Path(model_path) / "transformer" if (Path(model_path) / "transformer").is_dir() else Path(model_path)
    config = {k: v for k, v in json.loads((folder / "config.json").read_text()).items()
              if not k.startswith("_")}

    with torch.device("meta"):
        model = H3Transformer3DModel(**config)
    model.eval()
    model.requires_grad_(False)

    keep_fp32 = H3Transformer3DModel.KEEP_IN_FP32
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(folder / shard), framework="pt", device=str(device)) as f:
            for name in f.keys():
                if name.endswith(("_nvfp4", "_nvfp4_scale")):
                    continue
                dtype = torch.float32 if any(m in name for m in keep_fp32) else torch.bfloat16
                state[name] = f.get_tensor(name).to(dtype)

    model.load_state_dict(state, strict=False, assign=True)
    del state

    rope_theta = float(config.get("rope_theta", 10000.0))
    rope_freq_dim = int(config.get("rope_freq_dim", 16))
    model.rope.register_buffer(
        "inv_freq",
        1.0 / (rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32, device=device)
                              / (2 * rope_freq_dim))),
        persistent=False,
    )

    summary = load_quantized(model, str(folder))

    stranded = [n for n, p in model.named_parameters() if p.is_meta]
    if stranded:
        raise RuntimeError(f"Parameters left on the meta device after NVFP4 load: {stranded[:8]}")
    return model, summary

@torch.no_grad()
def load_quantized(model, path: str) -> dict:
    import json
    from pathlib import Path

    from safetensors import safe_open

    folder = Path(path)
    index = next(folder.glob("*.safetensors.index.json"), None)
    if index is None:
        raise FileNotFoundError(f"no safetensors index under {folder}")
    weight_map = json.loads(index.read_text())["weight_map"]

    wanted: dict[str, tuple[str, str]] = {}
    for key in weight_map:
        if key.endswith("_nvfp4"):
            wanted[key[: -len("_nvfp4")]] = (key, key + "_scale")
    if not wanted:
        raise RuntimeError(f"{folder} carries no `_nvfp4` weights — is it the bf16 checkpoint?")

    device = next((p.device for p in model.parameters() if not p.is_meta), torch.device("cuda"))
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted({weight_map[k] for pair in wanted.values() for k in pair}):
        with safe_open(str(folder / shard), framework="pt", device=str(device)) as f:
            for name in f.keys():
                if name.endswith(("_nvfp4", "_nvfp4_scale")):
                    tensors[name] = f.get_tensor(name)

    replaced = 0
    for block_index, block in enumerate(model.transformer_blocks):
        for parent, attr, linear in list(_block_linears(block)):
            prefix = f"transformer_blocks.{block_index}." + _suffix_for(parent, attr, block)
            pair = wanted.get(prefix)
            if pair is None:
                raise KeyError(f"pre-quantized checkpoint is missing {prefix}_nvfp4")
            q = _QuantLinear.__new__(_QuantLinear)
            nn.Module.__init__(q)
            w = NVFP4Weight.__new__(NVFP4Weight)
            w.out_features, w.in_features = linear.out_features, linear.in_features
            w.dtype = torch.bfloat16
            w.qdata_t = tensors[pair[0]].view(torch.float4_e2m1fn_x2).t()
            w.scale = tensors[pair[1]].view(torch.float8_e4m3fn)
            q.weight_nvfp4 = w
            q.in_features, q.out_features = linear.in_features, linear.out_features
            q.calls = 0
            if isinstance(parent, (nn.ModuleList, nn.Sequential)):
                parent[int(attr)] = q
            else:
                setattr(parent, attr, q)
            replaced += 1
    torch.cuda.empty_cache()
    return {"linears_loaded": replaced, "resident_gib": torch.cuda.memory_allocated() / 1024**3}

def _suffix_for(parent, attr: str, block) -> str:
    if parent is block.attn:
        return f"attn.{attr}.weight"
    if parent is block.attn.to_out:
        return f"attn.to_out.{attr}.weight"
    if parent is block.ff.net[0]:
        return f"ff.net.0.{attr}.weight"
    if parent is block.ff.net:
        return f"ff.net.{attr}.weight"
    raise KeyError(f"unmapped block Linear: {parent}.{attr}")

def shared_qkv(to_q, to_k, to_v, x: torch.Tensor):
    mods = (to_q, to_k, to_v)
    if not all(isinstance(m, _QuantLinear) for m in mods):
        return None
    act = QuantizedActivation(x)
    for m in mods:
        m.calls += 1
    return tuple(act.mm(m.weight_nvfp4) for m in mods)

def report(model) -> str:
    served = sum(m.calls for m in model.modules() if isinstance(m, _QuantLinear))
    count = sum(1 for m in model.modules() if isinstance(m, _QuantLinear))
    return f"{count} NVFP4 linears, {served} calls served"
