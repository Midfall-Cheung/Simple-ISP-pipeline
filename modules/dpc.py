"""Defective pixel correction (DPC).

Dead pixels (stuck at 0) and hot pixels (stuck at saturation) are isolated
outliers of their *own* colour channel. Detection therefore runs on the four
half-resolution same-colour sub-grids: a pixel is a suspect when it is both a
5x5 local extremum and far from the same-colour median. Correction replaces
the value by that median.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from utils.cfa import phase_of, split_phases


def defect_mask(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    threshold: float = 0.05,
    ksize: int = 5,
    noise_factor: float = 10.0,
) -> Tuple[np.ndarray, int]:
    """Detect dead/hot pixels.

    Parameters
    ----------
    mosaic:
        Normalised (``[0, 1]``) float32 mosaic. ``threshold`` is expressed in
        the same units, e.g. ``0.02`` == 2 % of full scale.
    pattern:
        CFA phase pattern.
    ksize:
        Same-colour neighbourhood size, ``5`` by default (2 -> 5x5 window).
    noise_factor:
        Extra guard against texture: a pixel must also deviate more than
        ``noise_factor`` times the local high-frequency noise (estimated with
        a median of absolute deviations). Real defects are single-pixel
        outliers; textured detail is not.

    Returns
    -------
    mask, count:
        Full-resolution boolean mask and the number of flagged pixels.
    """
    mosaic = np.asarray(mosaic, dtype=np.float32)
    mask = np.zeros(mosaic.shape, dtype=bool)
    count = 0
    for index, sub in split_phases(mosaic, pattern).items():
        sub = np.ascontiguousarray(sub, dtype=np.float32)
        median = cv2.medianBlur(sub, ksize)
        diff = np.abs(sub - median)
        # Local noise floor: 1.4826 * MAD ~= sigma for Gaussian noise.
        noise = cv2.medianBlur(diff, ksize) * 1.4826
        threshold_map = np.maximum(threshold, noise_factor * noise)

        kernel = np.ones((ksize, ksize), np.uint8)
        local_max = cv2.dilate(sub, kernel)
        local_min = cv2.erode(sub, kernel)
        is_extreme = (sub >= local_max - 1e-6) | (sub <= local_min + 1e-6)

        suspect = (diff > threshold_map) & is_extreme
        r, c = phase_of(pattern, index)
        mask[r::2, c::2] = suspect
        count += int(suspect.sum())
    return mask, count


def correct_defects(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    threshold: float = 0.05,
    ksize: int = 5,
    mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace defective pixels with the same-colour 5x5 median.

    Returns the corrected mosaic and the defect mask that was applied.
    """
    mosaic = np.asarray(mosaic, dtype=np.float32)
    if mask is None:
        mask, _ = defect_mask(mosaic, pattern, threshold=threshold, ksize=ksize)

    out = mosaic.copy()
    for index in np.unique(pattern):
        r, c = phase_of(pattern, int(index))
        phase_mask = mask[r::2, c::2]
        if not phase_mask.any():
            continue
        sub = np.ascontiguousarray(mosaic[r::2, c::2], dtype=np.float32)
        median = cv2.medianBlur(sub, ksize)
        out[r::2, c::2] = np.where(phase_mask, median, sub)
    return out, mask


def inject_defects(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    fraction: float = 1e-4,
    seed: int = 0,
    values: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject synthetic dead (0) and hot (1) pixels for unit tests.

    Pixels are sampled inside their own colour phase so the ground truth mask
    is exact. Returns the corrupted mosaic and a boolean ground-truth mask.
    """
    mosaic = np.asarray(mosaic, dtype=np.float32)
    rng = np.random.default_rng(seed)
    out = mosaic.copy()
    truth = np.zeros(mosaic.shape, dtype=bool)

    for index, sub in split_phases(mosaic, pattern).items():
        n = int(round(fraction * sub.size))
        if n <= 0:
            continue
        rows = rng.integers(0, sub.shape[0], n)
        cols = rng.integers(0, sub.shape[1], n)
        r, c = phase_of(pattern, index)
        out[r + 2 * rows, c + 2 * cols] = rng.choice(values, n)
        truth[r + 2 * rows, c + 2 * cols] = True
    return out, truth
