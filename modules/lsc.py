"""Lens shading correction (LSC).

A purely radial gain surface compensates vignetting:

    G(r) = 1 + k1 * r^2 + k2 * r^4,      r in [0, 1]

with ``r`` the normalised distance to the optical centre. The gain is applied
on the CFA mosaic before white balance so every colour channel sees the same
optical falloff.

Two modes are supported:

* ``explicit``: the caller supplies ``k1`` / ``k2`` from a flat-field
  calibration (the correct way for production).
* ``auto``: coefficients are fitted from the image's own smooth radial
  profile. That assumes a roughly uniform scene (sky, wall) and is only a
  first-order estimate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def radial_gain_map(
    shape: Tuple[int, int],
    k1: float,
    k2: float,
    center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Build ``G(r) = 1 + k1 r^2 + k2 r^4`` as an ``(H, W)`` float32 map."""
    h, w = shape
    cy, cx = center if center is not None else ((h - 1) / 2.0, (w - 1) / 2.0)
    yy = (np.arange(h, dtype=np.float32) - cy)[:, None]
    xx = (np.arange(w, dtype=np.float32) - cx)[None, :]
    norm = float(np.hypot(cy, cx))
    r2 = (yy * yy + xx * xx) / (norm * norm)
    gain = 1.0 + k1 * r2 + k2 * r2 * r2
    return np.clip(gain, 0.1, 8.0).astype(np.float32)


def apply_lsc(
    mosaic: np.ndarray,
    k1: float,
    k2: float,
    center: Optional[Tuple[float, float]] = None,
    clip: bool = True,
) -> np.ndarray:
    """Apply radial vignetting compensation to the mosaic."""
    mosaic = np.asarray(mosaic, dtype=np.float32)
    gain = radial_gain_map(mosaic.shape, k1, k2, center=center)
    out = mosaic * gain
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out


def _radial_profile(
    values: np.ndarray, bins: int = 16, max_samples: int = 4_000_000
) -> Tuple[np.ndarray, np.ndarray]:
    """Median value per normalised radial bin."""
    h, w = values.shape
    stride = max(1, int(np.sqrt(values.size / max_samples)))
    sample = np.asarray(values[::stride, ::stride], dtype=np.float32)
    sh, sw = sample.shape
    cy, cx = (sh - 1) / 2.0, (sw - 1) / 2.0
    norm = float(np.hypot(cy, cx))
    yy = (np.arange(sh, dtype=np.float32) - cy)[:, None]
    xx = (np.arange(sw, dtype=np.float32) - cx)[None, :]
    radius = np.sqrt(yy * yy + xx * xx) / norm

    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, bins - 1)
    flat = sample.ravel()
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.empty(bins, dtype=np.float32)
    for b in range(bins):
        sel = which == b
        profile[b] = float(np.median(flat[sel])) if sel.any() else np.nan
    return centers, profile


def estimate_coefficients(
    mosaic: np.ndarray,
    bins: int = 16,
    upper_bound: float = 0.6,
) -> Tuple[float, float]:
    """Fit ``k1`` / ``k2`` from the image's own radial falloff.

    Solves ``1/s(r) - 1 = k1 r^2 + k2 r^4`` by least squares, where ``s`` is
    the measured radial profile normalised at the centre. Coefficients are
    clamped to ``[0, upper_bound]`` so a non-uniform scene cannot produce a
    destructive over-correction.
    """
    centers, profile = _radial_profile(np.asarray(mosaic, dtype=np.float32), bins=bins)
    valid = np.isfinite(profile) & (profile > 1e-4)
    centers, profile = centers[valid], profile[valid]
    if centers.size < 4:
        return 0.0, 0.0

    rel = profile / profile[0]
    target = 1.0 / np.clip(rel, 1e-3, None) - 1.0
    basis = np.stack([centers ** 2, centers ** 4], axis=1)
    coef, *_ = np.linalg.lstsq(basis, target, rcond=None)
    k1 = float(np.clip(coef[0], 0.0, upper_bound))
    k2 = float(np.clip(coef[1], 0.0, upper_bound))
    return k1, k2
