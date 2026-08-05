
from __future__ import annotations

import contextlib
import os
from typing import Iterable, Sequence

import torch

# Timestep values are float32 and come from a fixed schedule; quantizing the cache key
# keeps float formatting out of it. 1e-6 is far finer than any sampler's step spacing.
_KEY_ROUND = 1_000_000

def enabled() -> bool:
    return os.environ.get("H3_FOLD_ADALN") == "1" or os.environ.get("H3_KERNELS") == "1"

def _key(timestep: torch.Tensor) -> tuple[int, ...]:
    """A hashable identity for one evaluation's timestep vector."""
    return tuple(int(round(float(v) * _KEY_ROUND)) for v in timestep.flatten().tolist())

class ScheduleFold:
    """The precomputed AdaLN tables for one model, keyed by timestep vector."""

    def __init__(self, model) -> None:
        self.model = model
        self.blocks: dict[tuple[int, ...], list[tuple[torch.Tensor, ...]]] = {}
        self.out: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self.freed = False
        self.hits = 0
        self.misses = 0

    def _temb(self, timestep: torch.Tensor) -> torch.Tensor:
        from h3_inference.kernels import other as _other

        return _other.timestep_embedding(self.model.time_proj, self.model.time_embedder, timestep)

    @torch.no_grad()
    def compute(self, timestep: torch.Tensor) -> None:
        """Fill the cache for one timestep vector. Idempotent."""
        key = _key(timestep)
        if key in self.blocks:
            return
        if self.freed:
            raise RuntimeError(
                f"schedule_fold: timestep {key} is not in the precomputed schedule and the "
                f"adaln_proj weights have been freed. `arm()` must be given every timestep "
                f"the sampler will visit; cached: {sorted(self.blocks)}"
            )
        self.misses += 1
        temb = self._temb(timestep)
        self.blocks[key] = [
            tuple(t.clone() for t in block.adaln_proj(temb)) for block in self.model.transformer_blocks
        ]
        shift, scale = self.model.norm_out.linear(
            torch.nn.functional.silu(temb).to(self.model.norm_out.linear.weight.dtype)
        ).chunk(2, dim=-1)
        self.out[key] = (shift.clone(), scale.clone())

    @torch.no_grad()
    def precompute(self, timesteps: Iterable[torch.Tensor]) -> int:
        for t in timesteps:
            self.compute(t)
        return len(self.blocks)

    def select(self, timestep: torch.Tensor) -> None:
        """Point every block at this step's tables. Called once per evaluation."""
        key = _key(timestep)
        if key not in self.blocks:
            self.compute(timestep)
        else:
            self.hits += 1
        for block, tables in zip(self.model.transformer_blocks, self.blocks[key]):
            block.folded_modulation = tables
        self.model.norm_out.folded_modulation = self.out[key]

    def clear_selection(self) -> None:
        for block in self.model.transformer_blocks:
            block.folded_modulation = None
        self.model.norm_out.folded_modulation = None

    @contextlib.contextmanager
    def pinned(self, timestep: torch.Tensor):
        """Select one step's tables, then stop re-selecting for the duration."""

        self.select(timestep)
        wrapper = self.model.__dict__.pop("forward", None)
        try:
            yield self
        finally:
            if wrapper is not None:
                self.model.forward = wrapper

    @torch.no_grad()
    def free(self) -> float:
        released = 0
        for block in self.model.transformer_blocks:
            released += _free_linear(block.adaln_proj)
        released += _free_linear(self.model.norm_out)
        self.freed = True
        torch.cuda.empty_cache()
        return released / (1024 ** 3)

    def stats(self) -> dict:
        return {
            "cached_steps": len(self.blocks),
            "hits": self.hits,
            "misses": self.misses,
            "freed": self.freed,
            "table_bytes": sum(
                t.numel() * t.element_size() for tables in self.blocks.values() for tt in tables for t in tt
            ),
        }

class _FreedLinear(torch.nn.Module):
    """Stands in for a projection whose weights `schedule_fold.free()` released."""

    def forward(self, *_args, **_kwargs):
        raise RuntimeError(
            "schedule_fold: this adaln_proj was freed and its output was requested. The "
            "block's `folded_modulation` is unset, which means `select()` was not called "
            "for this evaluation's timestep."
        )

def _free_linear(module: torch.nn.Module) -> int:
    """Release `module.linear`'s parameters and report the bytes."""
    linear = getattr(module, "linear", None)
    if linear is None or isinstance(linear, _FreedLinear):
        return 0
    released = sum(p.numel() * p.element_size() for p in linear.parameters())
    module.linear = _FreedLinear()
    return released

def install(model) -> ScheduleFold:
    fold = getattr(model, "_schedule_fold", None)
    if fold is not None:
        return fold

    fold = ScheduleFold(model)
    model._schedule_fold = fold
    original = model.forward

    def forward(*args, **kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None:
            return original(*args, **kwargs)
        fold.select(timestep)
        return original(*args, **kwargs)

    model.forward = forward
    return fold

def uninstall(model) -> None:
    """Restore the stock path. Refuses once the projections are gone, because there is no
    stock path left to restore — a silent restore would fail inside a block instead."""
    fold = getattr(model, "_schedule_fold", None)
    if fold is None:
        return
    if fold.freed:
        raise RuntimeError(
            "schedule_fold: cannot uninstall after free() — the adaln_proj weights are "
            "gone. Arm with free=False if the same process must also run a stock arm."
        )
    fold.clear_selection()
    del model.forward          # drops the instance attribute, exposing the bound method
    del model._schedule_fold


@torch.no_grad()
def arm(model, timesteps: Sequence[torch.Tensor], free: bool = True) -> ScheduleFold:
    """Precompute the whole schedule, then free the projections. Returns the fold."""

    fold = install(model)
    fold.precompute(timesteps)
    if free:
        fold.freed_gib = fold.free()
    return fold

def schedule_timesteps(steps: int, device: str = "cuda") -> list[torch.Tensor]:
    """The timestep vector each evaluation of a `steps`-point schedule sees."""

    sigmas = torch.linspace(1.0, 0.0, steps, device=device, dtype=torch.float32)
    return [torch.stack([sigmas[i], sigmas[i] * 0.5]) for i in range(steps - 1)]

def status(model) -> str:
    fold = getattr(model, "_schedule_fold", None)
    if fold is None:
        return "not installed"
    s = fold.stats()
    return (
        f"{s['cached_steps']} steps cached, {s['hits']} hits / {s['misses']} misses, "
        f"tables {s['table_bytes'] / 2**20:.1f} MiB, projections "
        f"{'freed' if s['freed'] else 'resident'}"
    )
