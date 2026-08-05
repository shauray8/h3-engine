# maiku dispatch 
from __future__ import annotations

import os
from typing import Any


class _Miss:
    __slots__ = ()
    def __repr__(self) -> str:
        return "MISS"

    def __bool__(self) -> bool:
        return False

MISS: Any = _Miss()

_maiku_try_call = None
_maiku_state = "unloaded"

def _load_maiku():
    global _maiku_try_call, _maiku_state
    if _maiku_state != "unloaded":
        return _maiku_try_call
    if os.environ.get("H3_NO_MAIKU") == "1":
        _maiku_state = "disabled by H3_NO_MAIKU"
        return None
    try:
        from maiku.dispatch import MISS as _maiku_miss
        from maiku.dispatch import try_call as _try_call
    except Exception as exc:
        _maiku_state = f"unavailable: {type(exc).__name__}: {exc}"
        return None

    def _call(name: str, *args, **kwargs):
        out = _try_call(name, *args, **kwargs)
        return MISS if out is _maiku_miss else out

    _maiku_try_call = _call
    _maiku_state = "loaded"
    return _call

def try_call(name: str, *args, **kwargs):
    call = _load_maiku()
    if call is None:
        return MISS
    return call(name, *args, **kwargs)

def maiku_status() -> str:
    _load_maiku()
    return _maiku_state

_ARMED: dict[str, int] = {}

def armed_counts(reset: bool = False) -> dict[str, int]:
    out = dict(_ARMED)
    if reset:
        _ARMED.clear()
    return out

def enabled(flag: str) -> bool:
    on = os.environ.get(flag) == "1" or os.environ.get("H3_KERNELS") == "1"
    if on:
        _ARMED[flag] = _ARMED.get(flag, 0) + 1
    return on
