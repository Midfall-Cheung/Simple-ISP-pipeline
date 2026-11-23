"""CFA / Bayer phase helpers shared by every module.

The 2x2 pattern stored in :class:`utils.raw_loader.RawFrame` is the single
source of truth: each entry is an index into ``color_desc`` (e.g. ``"RGBG"``).
Working with the phase *index* instead of hard-coded pixel offsets keeps BLC,
DPC, RAW-NR and demosaic aligned to the same physical layout.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def channel_map(pattern: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """(H, W) map of pixel -> channel index, derived from the 2x2 ``pattern``."""
    pattern = np.asarray(pattern, dtype=np.uint8)
    if pattern.shape != (2, 2):
        raise ValueError(f"expected a 2x2 CFA pattern, got {pattern.shape}")
    h, w = shape
    return pattern[np.ix_(np.arange(h) % 2, np.arange(w) % 2)]


def phase_of(pattern: np.ndarray, index: int) -> Tuple[int, int]:
    """Row/column phase of channel ``index`` inside the 2x2 tile."""
    pattern = np.asarray(pattern)
    hits = np.argwhere(pattern == index)
    if hits.size == 0:
        raise ValueError(f"channel index {index} not present in pattern {pattern.tolist()}")
    return int(hits[0][0]), int(hits[0][1])


def subgrid(mosaic: np.ndarray, pattern: np.ndarray, index: int) -> np.ndarray:
    """Half-resolution same-channel sub-grid (view) for one CFA phase."""
    r, c = phase_of(pattern, index)
    return mosaic[r::2, c::2]


def split_phases(mosaic: np.ndarray, pattern: np.ndarray) -> Dict[int, np.ndarray]:
    """Split the mosaic into one half-resolution view per channel index."""
    return {int(i): subgrid(mosaic, pattern, int(i)) for i in np.unique(pattern)}


def color_groups(color_desc: str) -> Dict[str, Tuple[int, ...]]:
    """``{'R': (0,), 'G': (1, 3), 'B': (2,)}`` for ``color_desc='RGBG'``."""
    groups: Dict[str, Tuple[int, ...]] = {}
    for name in "RGB":
        idx = tuple(i for i, ch in enumerate(color_desc) if ch == name)
        if idx:
            groups[name] = idx
    return groups
