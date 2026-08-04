"""Direct safetensors module loading. No registry, no builders, no lazy lifecycle."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from h3_inference.transformer.model import H3Transformer3DModel

def _shard_paths(folder: Path) -> list[Path]:
    index = next(folder.glob("*.safetensors.index.json"), None)
    if index is not None:
        weight_map = json.loads(index.read_text())["weight_map"]
        return [folder / name for name in sorted(set(weight_map.values()))]
    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors under {folder}")
    return shards

def _target_dtype(name: str, keep_fp32: tuple[str, ...], dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if any(marker in name for marker in keep_fp32) else dtype

def load_transformer(
    model_path: str | os.PathLike,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> H3Transformer3DModel:
    folder = Path(model_path) / "transformer"
    config = json.loads((folder / "config.json").read_text())
    config = {k: v for k, v in config.items() if not k.startswith("_")}

    t0 = time.perf_counter()
    with torch.device("meta"):
        model = H3Transformer3DModel(**config)
    model.eval()
    model.requires_grad_(False)

    keep_fp32 = H3Transformer3DModel.KEEP_IN_FP32
    state: dict[str, torch.Tensor] = {}
    for shard in _shard_paths(folder):
        with safe_open(str(shard), framework="pt", device=str(device)) as f:
            for name in f.keys():
                state[name] = f.get_tensor(name).to(_target_dtype(name, keep_fp32, dtype))

    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    expected_missing = {"rope.inv_freq"}
    if set(missing) - expected_missing:
        raise RuntimeError(f"Missing checkpoint keys: {sorted(set(missing) - expected_missing)[:8]}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {sorted(unexpected)[:8]}")

    rope_theta = float(config.get("rope_theta", 10000.0))
    rope_freq_dim = int(config.get("rope_freq_dim", 16))
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32, device=device) / (2 * rope_freq_dim))
    )
    model.rope.register_buffer("inv_freq", inv_freq, persistent=False)
    stranded = [n for n, p in model.named_parameters() if p.is_meta]
    if stranded:
        raise RuntimeError(f"Parameters left on the meta device: {stranded[:8]}")

    model._load_seconds = time.perf_counter() - t0
    return model

def transformer_config(model_path: str | os.PathLike) -> dict:
    config = json.loads((Path(model_path) / "transformer" / "config.json").read_text())
    return {k: v for k, v in config.items() if not k.startswith("_")}
