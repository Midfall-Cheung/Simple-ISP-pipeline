"""Optional C++/OpenMP backend.

The shared object is built from ``csrc/isp_accel.cpp`` with
``python -m tools.build_cpp``. It is optional: every caller falls back to the
pure NumPy/OpenCV implementation when the extension is missing, so the repo
still runs on a machine without a compiler.

Usage::

    from modules import accel
    accel.available()          # is the extension importable?
    accel.resolve("auto")      # "cpp" or "python"
    accel.hamilton_adams(...)  # raises if the extension is missing
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_EXT = None
_ERROR: Optional[str] = None


def load():
    """Import ``modules._isp_accel`` once; returns ``None`` when unavailable."""
    global _EXT, _ERROR
    if _EXT is not None or _ERROR is not None:
        return _EXT
    try:
        from . import _isp_accel as ext  # type: ignore

        _EXT = ext
    except ImportError as exc:  # pragma: no cover - depends on local build
        _ERROR = str(exc)
        _EXT = None
    return _EXT


def available() -> bool:
    return load() is not None


def error() -> Optional[str]:
    load()
    return _ERROR


def info() -> dict:
    ext = load()
    if ext is None:
        return {"available": False, "error": _ERROR}
    data = {"available": True}
    if hasattr(ext, "max_threads"):
        data["openmp_threads"] = int(ext.max_threads())
    return data


def resolve(backend: str = "auto") -> str:
    """Map ``auto``/``cpp``/``python`` to the backend that will actually run."""
    if backend == "python":
        return "python"
    if backend == "auto":
        return "cpp" if available() else "python"
    if backend == "cpp":
        if not available():
            raise RuntimeError(
                "C++ backend not built; run `python -m tools.build_cpp` first "
                f"({_ERROR})"
            )
        return "cpp"
    raise ValueError(f"unknown backend: {backend!r}")


def hamilton_adams(
    mosaic: np.ndarray, pattern: np.ndarray, color_desc: str = "RGBG"
) -> np.ndarray:
    ext = load()
    if ext is None:
        raise RuntimeError("C++ extension is not built")
    return ext.hamilton_adams(
        np.ascontiguousarray(mosaic, dtype=np.float32),
        np.ascontiguousarray(pattern, dtype=np.uint8),
        color_desc,
    )


def bilateral_mosaic(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    sigma_color: float,
    sigma_space: float = 2.0,
    diameter: int = 5,
) -> np.ndarray:
    ext = load()
    if ext is None:
        raise RuntimeError("C++ extension is not built")
    radius = max(1, diameter // 2)
    return ext.bilateral_mosaic(
        np.ascontiguousarray(mosaic, dtype=np.float32),
        np.ascontiguousarray(pattern, dtype=np.uint8),
        float(sigma_color),
        float(sigma_space),
        int(radius),
    )
