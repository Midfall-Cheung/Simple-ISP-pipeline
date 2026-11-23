"""Colour correction matrix (CCM) and gamma encoding.

The CCM is applied on linear RGB *after* white balance. Its off-diagonal
terms are negative, so dark/oversaturated pixels can go negative; those are
clipped before gamma, otherwise ``x ** (1/2.2)`` would produce ``NaN``.

Gamma follows the outline's ``I_out = I_in ** (1/2.2)``; a piecewise sRGB
transfer curve is available as a stricter IEC 61966-2-1 option.
"""

from __future__ import annotations

import numpy as np


def apply_ccm(rgb: np.ndarray, ccm: np.ndarray, clip_negative: bool = True) -> np.ndarray:
    """Apply a 3x3 colour correction matrix to linear RGB."""
    rgb = np.asarray(rgb, dtype=np.float32)
    ccm = np.asarray(ccm, dtype=np.float32)
    if ccm.shape != (3, 3):
        raise ValueError(f"expected a 3x3 CCM, got {ccm.shape}")
    out = rgb @ ccm.T
    if clip_negative:
        np.clip(out, 0.0, None, out=out)
    return out.astype(np.float32)


def gamma_encode(image: np.ndarray, gamma: float = 2.2, mode: str = "gamma") -> np.ndarray:
    """Encode a linear image for display (``uint8``-friendly ``[0, 1]``)."""
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if mode == "gamma":
        return np.power(image, 1.0 / gamma)
    if mode == "srgb":
        low = image * 12.92
        high = 1.055 * np.power(image, 1.0 / 2.4) - 0.055
        return np.where(image <= 0.0031308, low, high).astype(np.float32)
    raise ValueError(f"unknown gamma mode: {mode!r}")


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Quantise a ``[0, 1]`` float image to ``uint8`` with rounding."""
    return np.clip(np.rint(np.clip(image, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def color_finish(
    rgb_linear: np.ndarray,
    ccm: np.ndarray,
    gamma: float = 2.2,
    gamma_mode: str = "gamma",
) -> np.ndarray:
    """CCM -> clip -> gamma -> ``uint8`` in one call."""
    corrected = apply_ccm(rgb_linear, ccm)
    np.clip(corrected, 0.0, 1.0, out=corrected)
    return to_uint8(gamma_encode(corrected, gamma=gamma, mode=gamma_mode))
