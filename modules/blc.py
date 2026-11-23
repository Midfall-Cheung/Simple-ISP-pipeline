"""Black level calibration (BLC).

Each CFA phase has its own sensor pedestal. Subtracting the wrong offset (or
none at all) lifts the shadows unequally per colour, which the later WB gains
then amplify into a magenta/green cast. All maths stays in ``float32`` and the
output is normalised to ``[0, 1]``.
"""

from __future__ import annotations

import numpy as np

from utils.cfa import channel_map, split_phases


def black_level_calibration(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    black_levels,
    white_level: float,
    clip: bool = True,
) -> np.ndarray:
    """Subtract per-channel black levels and normalise to ``[0, 1]``.

    Parameters
    ----------
    mosaic:
        Raw sensor counts, shape ``(H, W)`` (``uint16`` or ``float32``).
    pattern:
        ``2x2`` CFA phase pattern (channel index per pixel).
    black_levels:
        One offset per channel index, ordered like ``color_desc``.
    white_level:
        Sensor saturation level in the same units as ``mosaic``.
    clip:
        Clamp the result into ``[0, 1]``.
    """
    mosaic = np.asarray(mosaic, dtype=np.float32)
    black = np.asarray(black_levels, dtype=np.float32).ravel()
    ch = channel_map(pattern, mosaic.shape)

    out = np.empty_like(mosaic, dtype=np.float32)
    for index in np.unique(ch):
        mask = ch == index
        offset = float(black[index])
        span = float(white_level) - offset
        if span <= 0:
            raise ValueError(
                f"white level {white_level} must exceed black level {offset} (channel {index})"
            )
        out[mask] = (mosaic[mask] - offset) / span

    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out


def estimate_black_level(
    mosaic: np.ndarray,
    pattern: np.ndarray,
    percentile: float = 0.5,
) -> np.ndarray:
    """Estimate the pedesstal per channel from the darkest ``percentile``.

    Useful when a DNG carries no ``BlackLevel`` tag but the sensor still has
    an optical-black offset. Only the lower tail is used so the estimate is
    not dragged by scene content.
    """
    mosaic = np.asarray(mosaic, dtype=np.float32)
    ch = channel_map(pattern, mosaic.shape)
    black = np.zeros(4, dtype=np.float32)
    for index in np.unique(ch):
        black[index] = float(np.percentile(mosaic[ch == index], percentile))
    return black


def add_pedestal(mosaic: np.ndarray, pattern: np.ndarray, pedestal) -> np.ndarray:
    """Add a (simulated) per-channel pedestal to raw counts."""
    mosaic = np.asarray(mosaic, dtype=np.float32)
    ped = np.asarray(pedestal, dtype=np.float32).ravel()
    if ped.size == 1:
        return mosaic + float(ped[0])
    ch = channel_map(pattern, mosaic.shape)
    out = mosaic.copy()
    for index in np.unique(ch):
        out[ch == index] += float(ped[index])
    return out


def split_same_color(mosaic: np.ndarray, pattern: np.ndarray, index: int) -> np.ndarray:
    """Convenience re-export used by the DPC / NR modules."""
    return split_phases(mosaic, pattern)[index]
