"""Demosaicing (CFA interpolation).

The baseline is a phase-aware bilinear interpolator. R and B each occupy a
single CFA phase, so a 1-2-1 kernel normalised by the convolved phase mask is
enough. Green occupies *two* phases (Gr / Gb), which needs care:

* at R / B sites the four orthogonal neighbours are green -> average them;
* at green sites the pixel is already measured -> keep it (blending in the
  diagonal green phase would blur every green pixel).

Using the orthogonal-only cross kernel for green makes the mask normalisation
fall back to the measured value exactly where green is sampled.

The kernel weights are chosen so that measured pixels keep their own value,
i.e. this is an interpolator, not a smoothing filter.

Stage 3 will upgrade this to a gradient-directed (Hamilton-Adams) version.
"""

from __future__ import annotations

import cv2
import numpy as np

from utils.cfa import channel_map, color_groups

# 1-2-1 separable bilinear kernel (unnormalised; normalisation per phase is
# handled by dividing by the convolved phase mask).
_BILINEAR_KERNEL = np.array(
    [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32
)

# Orthogonal-only kernel used for the two-phase green channel.
_CROSS_KERNEL = np.array(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
)


def _filter(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(array, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _interpolate_plane(mosaic: np.ndarray, phase_mask: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a single-phase colour plane."""
    sparse = np.where(phase_mask, mosaic, 0.0).astype(np.float32)
    num = _filter(sparse, _BILINEAR_KERNEL)
    den = _filter(phase_mask.astype(np.float32), _BILINEAR_KERNEL)
    den[den < 1e-6] = 1.0
    return num / den


def _interpolate_green(mosaic: np.ndarray, phase_mask: np.ndarray) -> np.ndarray:
    """Green plane: average orthogonal neighbours only at R/B sites."""
    sparse = np.where(phase_mask, mosaic, 0.0).astype(np.float32)
    num = _filter(sparse, _CROSS_KERNEL)
    den = _filter(phase_mask.astype(np.float32), _CROSS_KERNEL)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), sparse)
    return out.astype(np.float32)


def bilinear_demosaic(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    color_desc: str = "RGBG",
) -> np.ndarray:
    """Bilinear demosaic a single-channel CFA mosaic into ``(H, W, 3)`` RGB."""
    mosaic = np.asarray(mosaic, dtype=np.float32)
    cmap = channel_map(pattern, mosaic.shape)
    groups = color_groups(color_desc)

    planes = []
    for name in "RGB":
        phase_mask = np.isin(cmap, groups[name])
        if name == "G":
            planes.append(_interpolate_green(mosaic, phase_mask))
        else:
            planes.append(_interpolate_plane(mosaic, phase_mask))
    return np.clip(np.stack(planes, axis=-1), 0.0, 1.0).astype(np.float32)


def demosaic(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    color_desc: str = "RGBG",
    method: str = "bilinear",
    **kwargs,
) -> np.ndarray:
    """Dispatch to the requested demosaic implementation."""
    if method == "bilinear":
        return bilinear_demosaic(mosaic, pattern, color_desc)
    if method == "hamilton":
        raise NotImplementedError(
            "Hamilton-Adams demosaic lands in stage 3; use method='bilinear' for now"
        )
    raise ValueError(f"unknown demosaic method: {method!r}")
