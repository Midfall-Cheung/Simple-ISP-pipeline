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

Stage 3 adds the gradient-directed **Hamilton-Adams** interpolator, which
reconstructs green along the direction with the smaller second-order gradient
and then rebuilds R/B through the constant colour-difference assumption.
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


def _shifted_planes(mosaic: np.ndarray, pattern: np.ndarray, color_desc: str):
    """Pad the sparse colour planes by 2 and return aligned shifts."""
    cmap = channel_map(pattern, mosaic.shape)
    groups = color_groups(color_desc)
    gmask = np.isin(cmap, groups["G"])
    rmask = cmap == groups["R"][0]
    bmask = cmap == groups["B"][0]

    def pad(plane):
        return cv2.copyMakeBorder(plane, 2, 2, 2, 2, cv2.BORDER_REFLECT_101)

    g = pad(np.where(gmask, mosaic, 0.0).astype(np.float32))
    c = pad(np.where(rmask | bmask, mosaic, 0.0).astype(np.float32))
    return g, c, rmask, bmask


def hamilton_adams_demosaic(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    color_desc: str = "RGBG",
) -> np.ndarray:
    """Gradient-directed demosaic (Hamilton-Adams 1997).

    1. At R/B sites, compare the horizontal vs vertical second-order gradient
       ``|2C - C(-2) - C(+2)| + |G(-1) - G(+1)|``; interpolate green along the
       smoother direction and add the matching Laplacian correction.
    2. Interpolate the colour differences ``R-G`` / ``B-G`` bilinearly.
    3. Median-filter the interpolated differences to kill false colour while
       keeping the measured sites untouched.
    """
    mosaic = np.asarray(mosaic, dtype=np.float32)
    g_pad, c_pad, rmask, bmask = _shifted_planes(mosaic, pattern, color_desc)

    g_left = g_pad[2:-2, 1:-3]
    g_right = g_pad[2:-2, 3:-1]
    g_up = g_pad[1:-3, 2:-2]
    g_down = g_pad[3:-1, 2:-2]

    c_center = c_pad[2:-2, 2:-2]
    c_left2 = c_pad[2:-2, 0:-4]
    c_right2 = c_pad[2:-2, 4:]
    c_up2 = c_pad[0:-4, 2:-2]
    c_down2 = c_pad[4:, 2:-2]

    horizontal = 2.0 * c_center - c_left2 - c_right2
    vertical = 2.0 * c_center - c_up2 - c_down2

    d_h = np.abs(g_left - g_right) + np.abs(horizontal)
    d_v = np.abs(g_up - g_down) + np.abs(vertical)

    g_horizontal = 0.5 * (g_left + g_right) + 0.25 * horizontal
    g_vertical = 0.5 * (g_up + g_down) + 0.25 * vertical

    green_est = np.where(
        d_h < d_v,
        g_horizontal,
        np.where(d_v < d_h, g_vertical, 0.5 * (g_horizontal + g_vertical)),
    )
    green = np.where(np.isin(channel_map(pattern, mosaic.shape), color_groups(color_desc)["G"]), mosaic, green_est)
    green = np.clip(green, 0.0, 1.0)

    def color_plane(mask: np.ndarray) -> np.ndarray:
        diff = np.where(mask, mosaic - green, 0.0).astype(np.float32)
        interpolated = _interpolate_plane(diff, mask)
        # Median-filter only where the difference was interpolated.
        median = cv2.medianBlur(np.ascontiguousarray(interpolated), 3)
        diff_final = np.where(mask, mosaic - green, median)
        return np.clip(green + diff_final, 0.0, 1.0)

    red = color_plane(rmask)
    blue = color_plane(bmask)
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


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
        return hamilton_adams_demosaic(mosaic, pattern, color_desc)
    raise ValueError(f"unknown demosaic method: {method!r}")
