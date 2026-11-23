"""Automatic white balance (AWB).

Two modes, selectable from the pipeline:

* ``camera``: use the as-shot multipliers recorded in the RAW file (DNG
  ``AsShotNeutral`` / EXIF). This is the look the camera would have produced.
* ``gray_world``: hand-written Gray-World estimate -- the scene average is
  assumed neutral, so each channel is scaled to the shared mean. The mean is
  taken over unsaturated pixels with percentile clipping so highlights and
  the optical-black tail do not bias it.

Gains are indexed by CFA channel (``len(color_desc)`` entries) and are
normalised to green so overall exposure is preserved.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.cfa import channel_map, color_groups


def apply_gains(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    gains: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """Multiply every CFA phase by its white-balance gain."""
    mosaic = np.asarray(mosaic, dtype=np.float32)
    gains = np.asarray(gains, dtype=np.float32).ravel()
    cmap = channel_map(pattern, mosaic.shape)
    out = np.empty_like(mosaic, dtype=np.float32)
    for index in np.unique(cmap):
        out[cmap == index] = mosaic[cmap == index] * float(gains[index])
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out


def gray_world_gains(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    color_desc: str = "RGBG",
    low: float = 2.0,
    high: float = 98.0,
    min_gain: float = 0.25,
    max_gain: float = 4.0,
) -> np.ndarray:
    """Estimate Grey-World gains on the mosaic, normalised to green."""
    mosaic = np.asarray(mosaic, dtype=np.float32)
    cmap = channel_map(pattern, mosaic.shape)
    groups = color_groups(color_desc)
    n_channels = len(color_desc)

    means = {}
    for name, indices in groups.items():
        vals = mosaic[np.isin(cmap, indices)]
        lo, hi = np.percentile(vals, [low, high])
        keep = (vals > lo) & (vals < hi)
        if keep.any():
            vals = vals[keep]
        means[name] = float(vals.mean())

    target = float(np.mean(list(means.values())))
    gains = np.ones(n_channels, dtype=np.float32)
    for name, indices in groups.items():
        if means[name] <= 1e-6:
            continue
        for i in indices:
            gains[i] = target / means[name]

    green = groups["G"]
    gains = gains / float(np.mean(gains[list(green)]))
    return np.clip(gains, min_gain, max_gain).astype(np.float32)


def gains_from_frame(frame, mode: str = "camera") -> Optional[np.ndarray]:
    """Gain vector for ``mode``; ``None`` means "leave the mosaic untouched"."""
    if mode == "camera":
        return frame.wb_gains()
    if mode == "gray_world":
        return gray_world_gains(frame.mosaic, frame.pattern, frame.color_desc)
    if mode == "none":
        return None
    raise ValueError(f"unknown AWB mode: {mode!r}")
