"""Objective image-quality metrics: PSNR, SSIM and CIEDE2000.

These quantify what each stage-3 upgrade (RAW-domain NR, gradient-directed
demosaic, LTM) is worth against the libraw rendering produced by
``rawpy.postprocess``.

Conventions
-----------
* images are ``(H, W, 3)`` RGB or ``(H, W)`` grayscale,
* inputs may be ``uint8`` / ``uint16`` / ``float`` in ``[0, 1]``,
* everything is computed **row-block by row-block** so a 24 MP frame never
  needs more than a few hundred MB (the reference machine has 8 GB),
* optional *exposure alignment* fits one global gain in linear light before
  scoring, so the different tone curve of libraw does not dominate PSNR/SSIM;
  the fitted gain is reported alongside the score.

The CIEDE2000 implementation is validated against the Sharma et al. (2005)
reference data set -- run ``python -m utils.metrics`` to execute the checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import cv2
import numpy as np

# sRGB (IEC 61966-2-1) -> XYZ, D65.
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
XYZ_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_25_POW_7 = 25.0**7

IMG_LIKE = np.ndarray


# --------------------------------------------------------------------------- #
# Chunked helpers
# --------------------------------------------------------------------------- #
def _to_float_chunk(chunk: np.ndarray) -> np.ndarray:
    if chunk.dtype == np.uint8:
        return chunk.astype(np.float32) / 255.0
    if chunk.dtype == np.uint16:
        return chunk.astype(np.float32) / 65535.0
    return np.clip(chunk.astype(np.float32), 0.0, 1.0)


def to_float01(image: np.ndarray) -> np.ndarray:
    """Normalise a (small) image to ``float32`` in ``[0, 1]``."""
    return _to_float_chunk(np.asarray(image))


def _row_blocks(height: int, chunk_rows: int) -> Iterator[Tuple[int, int]]:
    for y0 in range(0, height, chunk_rows):
        yield y0, min(height, y0 + chunk_rows)


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0).astype(np.float32)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1.0 / 2.4) - 0.055
    )


def _shapes_match(reference: np.ndarray, test: np.ndarray) -> None:
    if reference.shape[:2] != test.shape[:2]:
        raise ValueError(
            f"shape mismatch: reference {reference.shape[:2]} vs test {test.shape[:2]}"
        )


def align_pair(reference: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Centre-crop two images to a common size (rawpy and our pipeline agree)."""
    if reference.shape[:2] == test.shape[:2]:
        return reference, test
    h = min(reference.shape[0], test.shape[0])
    w = min(reference.shape[1], test.shape[1])
    ratio = (h / w) / (reference.shape[0] / reference.shape[1])
    if abs(ratio - 1.0) > 0.02:
        raise ValueError(f"aspect mismatch: {reference.shape[:2]} vs {test.shape[:2]}")

    def crop(img):
        y0 = (img.shape[0] - h) // 2
        x0 = (img.shape[1] - w) // 2
        return img[y0 : y0 + h, x0 : x0 + w]

    return crop(reference), crop(test)


# --------------------------------------------------------------------------- #
# PSNR / SSIM
# --------------------------------------------------------------------------- #
def psnr(
    reference: np.ndarray,
    test: np.ndarray,
    data_range: float = 1.0,
    chunk_rows: int = 256,
) -> float:
    """Peak signal-to-noise ratio in dB (``inf`` for identical images)."""
    _shapes_match(reference, test)
    squared_sum = 0.0
    count = 0
    for y0, y1 in _row_blocks(reference.shape[0], chunk_rows):
        diff = _to_float_chunk(reference[y0:y1]).astype(np.float64) - _to_float_chunk(
            test[y0:y1]
        ).astype(np.float64)
        squared_sum += float(np.sum(diff * diff))
        count += diff.size
    mse = squared_sum / max(count, 1)
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / mse))


def _ssim_channel(
    channel_a: np.ndarray,
    channel_b: np.ndarray,
    win_size: int = 11,
    sigma: float = 1.5,
    chunk_rows: int = 256,
) -> float:
    c1 = (0.01 * 1.0) ** 2
    c2 = (0.03 * 1.0) ** 2
    pad = win_size
    height = channel_a.shape[0]
    total = 0.0
    count = 0
    for y0, y1 in _row_blocks(height, chunk_rows):
        a0, a1 = max(0, y0 - pad), min(height, y1 + pad)
        a = _to_float_chunk(channel_a[a0:a1])
        b = _to_float_chunk(channel_b[a0:a1])
        blur = lambda x: cv2.GaussianBlur(x, (win_size, win_size), sigma, borderType=cv2.BORDER_REFLECT_101)
        mu_a, mu_b = blur(a), blur(b)
        mu_a2, mu_b2 = mu_a * mu_a, mu_b * mu_b
        sigma_a2 = blur(a * a) - mu_a2
        sigma_b2 = blur(b * b) - mu_b2
        sigma_ab = blur(a * b) - mu_a * mu_b
        ssim_map = ((2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)) / (
            (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
        )
        valid = ssim_map[y0 - a0 : y1 - a0]
        total += float(np.sum(valid.astype(np.float64)))
        count += valid.size
    return total / max(count, 1)


def ssim(
    reference: np.ndarray,
    test: np.ndarray,
    win_size: int = 11,
    sigma: float = 1.5,
    chunk_rows: int = 256,
) -> float:
    """Mean structural similarity (averaged over channels), in ``[0, 1]``."""
    _shapes_match(reference, test)
    if reference.ndim == 2:
        return _ssim_channel(reference, test, win_size, sigma, chunk_rows)
    channels = [
        _ssim_channel(reference[..., c], test[..., c], win_size, sigma, chunk_rows)
        for c in range(reference.shape[2])
    ]
    return float(np.mean(channels))


# --------------------------------------------------------------------------- #
# Colour difference (sRGB -> Lab -> CIEDE2000)
# --------------------------------------------------------------------------- #
def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB (``[0, 1]`` float or ``uint8``) to CIELAB (D65)."""
    rgb = to_float01(rgb).astype(np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ SRGB_TO_XYZ.T
    xyz = xyz / XYZ_WHITE_D65
    delta = 6.0 / 29.0
    f = np.where(xyz > delta**3, np.cbrt(np.maximum(xyz, 0.0)), xyz / (3 * delta**2) + 4.0 / 29.0)
    return np.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        axis=-1,
    )


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 colour difference between two Lab arrays."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    Cbar7 = Cbar**7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + _25_POW_7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.mod(np.degrees(np.arctan2(b1, a1p)), 360.0)
    h2p = np.mod(np.degrees(np.arctan2(b2, a2p)), 360.0)

    zero_chroma = (C1p * C2p) == 0.0
    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(zero_chroma, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lbarp = 0.5 * (L1 + L2)
    Cbarp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hbarp = np.where(
        np.abs(h1p - h2p) <= 180.0,
        hsum / 2.0,
        np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0),
    )
    hbarp = np.where(zero_chroma, hsum, hbarp)

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hbarp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hbarp))
        + 0.32 * np.cos(np.radians(3.0 * hbarp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hbarp - 63.0))
    )
    d_theta = 30.0 * np.exp(-(((hbarp - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp**7
    R_C = 2.0 * np.sqrt(Cbarp7 / (Cbarp7 + _25_POW_7))
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    S_L = 1.0 + (0.015 * (Lbarp - 50.0) ** 2) / np.sqrt(20.0 + (Lbarp - 50.0) ** 2)
    S_C = 1.0 + 0.045 * Cbarp
    S_H = 1.0 + 0.015 * Cbarp * T

    term_l = dLp / S_L
    term_c = dCp / S_C
    term_h = dHp / S_H
    return np.sqrt(term_l**2 + term_c**2 + term_h**2 + R_T * term_c * term_h)


def delta_e_2000_stats(
    reference: np.ndarray,
    test: np.ndarray,
    chunk_rows: int = 256,
) -> Tuple[float, float, float]:
    """Mean / 95th percentile / max ΔE₂₀₀₀, accumulated in a histogram."""
    _shapes_match(reference, test)
    edges = np.arange(0.0, 101.0, 0.1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist = np.zeros(centers.size, dtype=np.float64)
    total = 0.0
    count = 0
    max_value = 0.0
    for y0, y1 in _row_blocks(reference.shape[0], chunk_rows):
        diff = delta_e_2000(srgb_to_lab(reference[y0:y1]), srgb_to_lab(test[y0:y1]))
        total += float(np.sum(diff))
        count += diff.size
        max_value = max(max_value, float(np.max(diff)))
        clipped = np.clip(diff, 0.0, edges[-1] - 1e-6)
        hist += np.histogram(clipped, bins=edges)[0]
    if count == 0:
        return 0.0, 0.0, 0.0
    cdf = np.cumsum(hist) / count
    p95 = float(np.interp(0.95, cdf, centers))
    return total / count, p95, max_value


# --------------------------------------------------------------------------- #
# Exposure alignment
# --------------------------------------------------------------------------- #
def exposure_gain(reference: np.ndarray, test: np.ndarray, chunk_rows: int = 256) -> float:
    """Least-squares global gain in linear light that maps ``test`` to ``reference``."""
    _shapes_match(reference, test)
    numerator = 0.0
    denominator = 0.0
    for y0, y1 in _row_blocks(reference.shape[0], chunk_rows):
        ref_lin = srgb_to_linear(_to_float_chunk(reference[y0:y1])).astype(np.float64)
        test_lin = srgb_to_linear(_to_float_chunk(test[y0:y1])).astype(np.float64)
        numerator += float(np.sum(ref_lin * test_lin))
        denominator += float(np.sum(test_lin * test_lin))
    if denominator <= 0.0:
        return 1.0
    return numerator / denominator


def align_exposure(
    reference: np.ndarray,
    test: np.ndarray,
    gain: Optional[float] = None,
    chunk_rows: int = 256,
) -> Tuple[np.ndarray, float]:
    """Apply the least-squares exposure gain to ``test`` and return ``uint8``."""
    _shapes_match(reference, test)
    if gain is None:
        gain = exposure_gain(reference, test, chunk_rows=chunk_rows)
    out = np.empty(test.shape[:2] + ((3,) if test.ndim == 3 else ()), dtype=np.uint8)
    for y0, y1 in _row_blocks(test.shape[0], chunk_rows):
        linear = srgb_to_linear(_to_float_chunk(test[y0:y1])) * float(gain)
        out[y0:y1] = np.clip(np.rint(linear_to_srgb(linear) * 255.0), 0, 255).astype(np.uint8)
    return out, gain


# --------------------------------------------------------------------------- #
# High level helpers
# --------------------------------------------------------------------------- #
def reference_from_raw(path: str | Path, gamma: Tuple[float, float] = (2.2, 2.2)) -> np.ndarray:
    """Render the libraw reference for a RAW file (camera WB, no auto bright)."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=8,
            gamma=gamma,
        )


def evaluate(
    reference: np.ndarray,
    test: np.ndarray,
    name: str = "",
    match_exposure: bool = True,
    chunk_rows: int = 256,
) -> Dict[str, object]:
    """Metric bundle used in the README tables."""
    reference, test = align_pair(reference, test)
    gain = 1.0
    if match_exposure:
        test, gain = align_exposure(reference, test, chunk_rows=chunk_rows)
    de_mean, de_p95, de_max = delta_e_2000_stats(reference, test, chunk_rows=chunk_rows)
    return {
        "name": name,
        "psnr_db": psnr(reference, test, chunk_rows=chunk_rows),
        "ssim": ssim(reference, test, chunk_rows=chunk_rows),
        "delta_e_mean": de_mean,
        "delta_e_p95": de_p95,
        "delta_e_max": de_max,
        "exposure_gain": float(gain),
        "shape": tuple(int(v) for v in reference.shape),
    }


def format_metrics(result: Dict[str, object]) -> str:
    gain = result.get("exposure_gain", 1.0)
    gain_txt = f" | gain {gain:.2f}" if abs(float(gain) - 1.0) > 1e-3 else ""
    return (
        f"{result['psnr_db']:7.2f} dB | SSIM {result['ssim']:.4f} | "
        f"ΔE00 mean {result['delta_e_mean']:6.2f} (p95 {result['delta_e_p95']:.2f}){gain_txt}"
    )


# --------------------------------------------------------------------------- #
# Self-test: Sharma et al. (2005) CIEDE2000 reference pairs
# --------------------------------------------------------------------------- #
_SHARMA_PAIRS = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -1.1848, -84.8006), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, -0.9009, -85.5211), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0009), 7.1792),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0011), 7.2195),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (61.0, -5.0, 29.0), 22.8977),
]


def _selftest() -> bool:
    ok = True
    for i, (lab1, lab2, expected) in enumerate(_SHARMA_PAIRS, start=1):
        got = float(delta_e_2000(np.array(lab1), np.array(lab2)))
        passed = abs(got - expected) < 1e-4
        ok &= passed
        print(f"  Sharma pair {i:2d}: expected {expected:.4f}, got {got:.4f}  {'OK' if passed else 'FAIL'}")

    white = srgb_to_lab(np.ones((1, 1, 3), dtype=np.float32))
    black = srgb_to_lab(np.zeros((1, 1, 3), dtype=np.float32))
    mid = srgb_to_lab(np.full((1, 1, 3), 128.0 / 255.0, dtype=np.float32))
    lab_ok = (
        np.allclose(white[0, 0], [100.0, 0.0, 0.0], atol=1e-2)
        and np.allclose(black[0, 0], [0.0, 0.0, 0.0], atol=1e-2)
        and abs(mid[0, 0, 0] - 53.585) < 0.05
    )
    ok &= lab_ok
    print(
        f"  sRGB->Lab: white={white[0, 0].round(3)}, black={black[0, 0].round(3)}, "
        f"L(sRGB 128)={mid[0, 0, 0]:.3f} (expect ~53.585)  {'OK' if lab_ok else 'FAIL'}"
    )

    identity = np.random.default_rng(0).uniform(0.2, 0.8, (64, 64, 3)).astype(np.float32)
    ident_ok = (
        np.isclose(psnr(identity, identity), float("inf"))
        and abs(ssim(identity, identity) - 1.0) < 1e-6
        and np.isclose(delta_e_2000_stats(identity, identity)[0], 0.0, atol=1e-6)
    )
    ok &= bool(ident_ok)
    print(f"  identity metrics: {'OK' if ident_ok else 'FAIL'}")

    # Chunked SSIM must equal whole-image SSIM.
    ref = np.random.default_rng(1).uniform(0, 1, (300, 260, 3)).astype(np.float32)
    tst = np.clip(ref + np.random.default_rng(2).normal(0, 0.02, ref.shape), 0, 1).astype(np.float32)
    chunk_ok = abs(ssim(ref, tst, chunk_rows=32) - ssim(ref, tst, chunk_rows=10000)) < 1e-6
    ok &= bool(chunk_ok)
    print(f"  chunked SSIM consistency: {'OK' if chunk_ok else 'FAIL'}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return bool(ok)


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
