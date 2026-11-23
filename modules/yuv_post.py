"""Local tone mapping (LTM) and YUV-domain edge enhancement.

LTM runs on linear RGB *before* gamma. A guided filter splits the luminance
into a smooth illumination base layer and a reflectance/detail layer; the base
is compressed in log space (lifting shadows, keeping white white) while the
detail layer is preserved. Re-applying the ratio to RGB keeps colours intact.

Sharpening runs in YUV *after* gamma, on the luma channel only, using an
unsharp mask with a noise-coring dead-zone so dark-area grain is not amplified.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

RGB_TO_YUV = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001],
    ],
    dtype=np.float32,
)
YUV_TO_RGB = np.linalg.inv(RGB_TO_YUV).astype(np.float32)

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Guided filter
# --------------------------------------------------------------------------- #
def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 16,
    eps: float = 1e-3,
) -> np.ndarray:
    """Edge-preserving guided filter (He et al. 2010), O(1) per pixel."""
    guide = np.asarray(guide, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)
    size = (2 * radius + 1, 2 * radius + 1)

    box = lambda x: cv2.boxFilter(x, -1, size, borderType=cv2.BORDER_REFLECT_101)
    mean_i = box(guide)
    mean_p = box(src)
    corr_i = box(guide * guide)
    corr_ip = box(guide * src)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


# --------------------------------------------------------------------------- #
# Local tone mapping
# --------------------------------------------------------------------------- #
def local_tone_map(
    rgb: np.ndarray,
    radius: int = 24,
    eps: float = 1e-3,
    compress: float = 0.6,
    exposure: float = 1.0,
    detail_gain: float = 1.0,
) -> np.ndarray:
    """Compress the illumination base layer while keeping local detail.

    Parameters
    ----------
    compress:
        Exponent applied to the log-domain base layer. ``1.0`` is a no-op,
        smaller values lift shadows more; ``0.6`` is a good starting point.
    exposure:
        Linear exposure multiplier applied before the decomposition.
    detail_gain:
        Multiplier on the detail (reflectance) layer; ``>1`` adds local
        contrast/pop, ``<1`` smooths it.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    luma = rgb @ LUMA_WEIGHTS
    luma_exposed = np.maximum(luma * float(exposure), 1e-6)

    log_luma = np.log(luma_exposed)
    base = guided_filter(log_luma, log_luma, radius=radius, eps=eps)
    detail = log_luma - base
    out_luma = np.exp(base * float(compress) + detail * float(detail_gain))

    ratio = out_luma / np.maximum(luma, 1e-6)
    return np.clip(rgb * ratio[..., None], 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# YUV domain sharpening
# --------------------------------------------------------------------------- #
def rgb_to_yuv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BT.601 full-range YUV, values stay in ``[0, 1]``-ish scale."""
    rgb = np.asarray(rgb, dtype=np.float32)
    yuv = rgb @ RGB_TO_YUV.T
    return yuv[..., 0], yuv[..., 1], yuv[..., 2]


def yuv_to_rgb(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    yuv = np.stack([y, u, v], axis=-1)
    return (yuv @ YUV_TO_RGB.T).astype(np.float32)


def unsharp_mask_yuv(
    rgb: np.ndarray,
    amount: float = 0.6,
    radius: float = 1.2,
    coring: float = 0.01,
) -> np.ndarray:
    """Unsharp-mask the luma channel with soft thresholding (coring).

    ``coring`` is a dead-zone in ``[0, 1]`` units: details smaller than that
    are treated as noise and left untouched, so shadow grain is not amplified.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    y, u, v = rgb_to_yuv(rgb)

    blurred = cv2.GaussianBlur(y, (0, 0), float(radius))
    detail = y - blurred
    if coring > 0:
        magnitude = np.abs(detail)
        shrink = np.maximum(magnitude - float(coring), 0.0) / np.maximum(magnitude, 1e-6)
        detail = detail * shrink
    y_sharp = np.clip(y + float(amount) * detail, 0.0, 1.0)
    return np.clip(yuv_to_rgb(y_sharp, u, v), 0.0, 1.0).astype(np.float32)
