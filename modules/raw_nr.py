"""RAW-domain adaptive denoising.

The Bayer mosaic is split into the four half-resolution, single-colour
sub-grids (R / Gr / Gb / B) and each sub-grid is filtered on its own. Because
every filtered sample only ever sees pixels of the same colour, no
cross-channel colour noise ("blotches") can be introduced -- something an
RGB-domain filter cannot guarantee after demosaicing mixed the channels.

The bilateral filter is the edge-preserving workhorse:

    I'(p) = (1/W) Σ_q exp(-||p-q||²/2σs²) · exp(-(I(p)-I(q))²/2σr²) · I(q)

``σr`` is derived from the local noise level (robust MAD estimate) so the
correction adapts per frame instead of using one hard-coded strength.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from modules import accel
from utils.cfa import phase_of, split_phases


def estimate_noise_sigma(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    block: int = 96,
    percentile: float = 75.0,
) -> float:
    """Robust, spatially-aware noise estimate of the mosaic.

    The frame is tiled into ``block`` x ``block`` patches; inside each patch
    the difference of horizontally adjacent *same-colour* samples is turned
    into a sigma with ``1.4826 * MAD / sqrt(2)``. Because noise is
    signal-dependent (shot noise) and textures inflate a global MAD, the
    returned value is a high-ish percentile of the per-block sigmas -- the
    typical noise of textured, mid-tone regions rather than the quietest sky
    or the busiest detail.
    """
    mosaic = np.asarray(mosaic, dtype=np.float32)
    sigmas = []
    for index in np.unique(pattern):
        r, c = phase_of(pattern, int(index))
        sub = np.ascontiguousarray(mosaic[r::2, c::2], dtype=np.float32)
        h, w = sub.shape
        for y in range(0, h - block + 1, block):
            for x in range(0, w - block + 1, block):
                diff = np.abs(np.diff(sub[y : y + block, x : x + block], axis=1))
                sigmas.append(1.4826 * float(np.median(diff)) / np.sqrt(2.0))
    if not sigmas:
        return 0.0
    return float(np.percentile(sigmas, percentile))


def denoise_raw(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    sigma_color: Optional[float] = None,
    sigma_space: float = 2.0,
    strength: float = 1.5,
    diameter: int = 5,
    keep_measured: bool = True,
    backend: str = "auto",
) -> Tuple[np.ndarray, float]:
    """Bilateral denoise per CFA phase; returns the result and σr used."""
    mosaic = np.asarray(mosaic, dtype=np.float32)
    if sigma_color is None:
        sigma = estimate_noise_sigma(mosaic, pattern)
        sigma_color = float(np.clip(strength * sigma, 0.002, 0.2))

    # Measured on this machine: OpenCV's SIMD bilateral beats our scalar C++
    # kernel (≈25 ms vs ≈40 ms on 6 MP), so ``auto`` keeps the OpenCV path and
    # the C++ kernel is opt-in for experiments.
    use_cpp = backend == "cpp" and accel.resolve("cpp") == "cpp"
    if use_cpp:
        out = accel.bilateral_mosaic(
            mosaic, pattern, sigma_color=sigma_color, sigma_space=sigma_space,
            diameter=diameter,
        )
        return np.clip(out, 0.0, 1.0), float(sigma_color)

    out = mosaic.copy()
    if not keep_measured:
        out = np.empty_like(mosaic)
    for index in np.unique(pattern):
        r, c = phase_of(pattern, int(index))
        sub = np.ascontiguousarray(mosaic[r::2, c::2], dtype=np.float32)
        filtered = cv2.bilateralFilter(sub, diameter, sigma_color, sigma_space)
        out[r::2, c::2] = filtered
    return np.clip(out, 0.0, 1.0), float(sigma_color)


def denoise_rgb(
    rgb: np.ndarray,
    sigma_color: Optional[float] = None,
    sigma_space: float = 2.0,
    strength: float = 1.5,
    diameter: int = 5,
) -> Tuple[np.ndarray, float]:
    """Demosaiced-domain bilateral filter, used as the comparison baseline."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if sigma_color is None:
        # Estimate on luma so the same σr scale as the RAW path is used.
        luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        high_pass = luma - cv2.medianBlur(luma, 3)
        sigma = 1.4826 * float(np.median(np.abs(high_pass)))
        sigma_color = float(np.clip(strength * sigma, 0.002, 0.2))
    filtered = cv2.bilateralFilter(np.ascontiguousarray(rgb), diameter, sigma_color, sigma_space)
    return np.clip(filtered, 0.0, 1.0), float(sigma_color)
