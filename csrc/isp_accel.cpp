// C++ / OpenMP acceleration for the Simple-ISP pipeline.
//
// Two hotspots are implemented here:
//   * bilateral_mosaic : RAW-domain bilateral filter, applied per CFA phase so
//     only same-colour samples (a 3x3 sub-lattice inside every 5x5 window) are
//     ever mixed. Range weights come from a lookup table to avoid exp() per tap.
//   * hamilton_adams   : fused gradient-directed demosaic (green pass, colour
//     difference interpolation, 3x3 median of the colour differences).
//
// Both kernels parallelise rows with `#pragma omp parallel for` and use
// replicate border handling.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clamp01(float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); }

// ---------------------------------------------------------------------------
// Bilateral filter (per CFA phase)
// ---------------------------------------------------------------------------
py::array_t<float> bilateral_mosaic(
    py::array_t<float, py::array::c_style | py::array::forcecast> mosaic,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> pattern,
    float sigma_color,
    float sigma_space,
    int radius) {
  auto buf = mosaic.request();
  if (buf.ndim != 2) throw std::runtime_error("mosaic must be 2-D");
  const int h = static_cast<int>(buf.shape[0]);
  const int w = static_cast<int>(buf.shape[1]);
  const float* src = static_cast<const float*>(buf.ptr);

  auto pbuf = pattern.request();
  if (pbuf.size != 4) throw std::runtime_error("pattern must be 2x2");
  const uint8_t* pat = static_cast<const uint8_t*>(pbuf.ptr);

  if (radius < 1) radius = 1;
  if (sigma_color <= 0.f) sigma_color = 1e-3f;
  if (sigma_space <= 0.f) sigma_space = 1.f;

  /* Spatial weights only for the same-colour sub-lattice: offsets with an
     even dy and dx, i.e. (2a, 2b) with a, b in [-radius/2, radius/2]. */
  const int half = radius / 2;
  const int side = 2 * half + 1;
  std::vector<float> spatial(side * side);
  const float inv_2ss = 1.f / (2.f * sigma_space * sigma_space);
  for (int b = -half; b <= half; ++b)
    for (int a = -half; a <= half; ++a) {
      const float dy = 2.f * a, dx = 2.f * b;
      spatial[(b + half) * side + (a + half)] =
          std::exp(-(dy * dy + dx * dx) * inv_2ss);
    }

  /* LUT over |difference| in [0, 4 * sigma_color] */
  constexpr int LUT_SIZE = 2048;
  const float range_max = 4.f * sigma_color;
  std::vector<float> lut(LUT_SIZE);
  for (int i = 0; i < LUT_SIZE; ++i) {
    const float d = range_max * static_cast<float>(i) / (LUT_SIZE - 1);
    lut[i] = std::exp(-(d * d) / (2.f * sigma_color * sigma_color));
  }
  const float lut_scale = (LUT_SIZE - 1) / range_max;

  py::array_t<float> out({h, w});
  float* dst = static_cast<float*>(out.request().ptr);

#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const int py = y & 1;
    for (int x = 0; x < w; ++x) {
      const int cx = x & 1;
      const uint8_t phase = pat[py * 2 + cx];
      const float center = src[y * w + x];
      float wsum = 0.f;
      float vsum = 0.f;
      for (int b = -half; b <= half; ++b) {
        const int yy = clamp_int(y + 2 * b, 0, h - 1);
        const int row = (yy & 1) * 2;
        const float* srow = src + static_cast<size_t>(yy) * w;
        for (int a = -half; a <= half; ++a) {
          const int xx = clamp_int(x + 2 * a, 0, w - 1);
          if (pat[row + (xx & 1)] != phase) continue; /* replicate border */
          const float diff = std::fabs(srow[xx] - center);
          if (diff >= range_max) continue;
          const float rw = lut[static_cast<int>(diff * lut_scale)];
          const float wgt = rw * spatial[(b + half) * side + (a + half)];
          vsum += wgt * srow[xx];
          wsum += wgt;
        }
      }
      dst[y * w + x] = wsum > 0.f ? clamp01(vsum / wsum) : center;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Hamilton-Adams demosaic
// ---------------------------------------------------------------------------
inline float median9(float v[9]) {
  /* insertion sort on 9 elements - branch predictor friendly and no allocs */
  for (int i = 1; i < 9; ++i) {
    float key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) {
      v[j + 1] = v[j];
      --j;
    }
    v[j + 1] = key;
  }
  return v[4];
}

py::array_t<float> hamilton_adams(
    py::array_t<float, py::array::c_style | py::array::forcecast> mosaic,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> pattern,
    const std::string& color_desc) {
  auto buf = mosaic.request();
  if (buf.ndim != 2) throw std::runtime_error("mosaic must be 2-D");
  const int h = static_cast<int>(buf.shape[0]);
  const int w = static_cast<int>(buf.shape[1]);
  const float* m = static_cast<const float*>(buf.ptr);

  auto pbuf = pattern.request();
  if (pbuf.size != 4) throw std::runtime_error("pattern must be 2x2");
  const uint8_t* pat = static_cast<const uint8_t*>(pbuf.ptr);

  int r_idx = -1, b_idx = -1;
  for (int i = 0; i < static_cast<int>(color_desc.size()); ++i) {
    if (color_desc[i] == 'R') r_idx = i;
    if (color_desc[i] == 'B') b_idx = i;
  }
  if (r_idx < 0 || b_idx < 0) throw std::runtime_error("color_desc needs R and B");

  int r_row = 0, r_col = 0, b_row = 0, b_col = 0;
  for (int i = 0; i < 4; ++i) {
    if (pat[i] == r_idx) { r_row = i / 2; r_col = i % 2; }
    if (pat[i] == b_idx) { b_row = i / 2; b_col = i % 2; }
  }

  std::vector<float> g(static_cast<size_t>(h) * w);
  std::vector<float> diff_r(static_cast<size_t>(h) * w, 0.f);
  std::vector<float> diff_b(static_cast<size_t>(h) * w, 0.f);
  std::vector<float> interp_r(static_cast<size_t>(h) * w, 0.f);
  std::vector<float> interp_b(static_cast<size_t>(h) * w, 0.f);

  /* ---- pass 1: green ---- */
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const uint8_t phase = pat[(y & 1) * 2 + (x & 1)];
      const float center = m[static_cast<size_t>(y) * w + x];
      const bool is_green = (color_desc[phase] == 'G');
      if (is_green) {
        g[static_cast<size_t>(y) * w + x] = center;
        continue;
      }
      const int xl = clamp_int(x - 1, 0, w - 1), xr = clamp_int(x + 1, 0, w - 1);
      const int yu = clamp_int(y - 1, 0, h - 1), yd = clamp_int(y + 1, 0, h - 1);
      const int xl2 = clamp_int(x - 2, 0, w - 1), xr2 = clamp_int(x + 2, 0, w - 1);
      const int yu2 = clamp_int(y - 2, 0, h - 1), yd2 = clamp_int(y + 2, 0, h - 1);

      const float gl = m[static_cast<size_t>(y) * w + xl];
      const float gr = m[static_cast<size_t>(y) * w + xr];
      const float gu = m[static_cast<size_t>(yu) * w + x];
      const float gd = m[static_cast<size_t>(yd) * w + x];
      const float cl2 = m[static_cast<size_t>(y) * w + xl2];
      const float cr2 = m[static_cast<size_t>(y) * w + xr2];
      const float cu2 = m[static_cast<size_t>(yu2) * w + x];
      const float cd2 = m[static_cast<size_t>(yd2) * w + x];

      const float horiz = 2.f * center - cl2 - cr2;
      const float vert = 2.f * center - cu2 - cd2;
      const float dh = std::fabs(gl - gr) + std::fabs(horiz);
      const float dv = std::fabs(gu - gd) + std::fabs(vert);
      const float gh = 0.5f * (gl + gr) + 0.25f * horiz;
      const float gv = 0.5f * (gu + gd) + 0.25f * vert;
      float value;
      if (dh < dv) value = gh;
      else if (dv < dh) value = gv;
      else value = 0.5f * (gh + gv);
      g[static_cast<size_t>(y) * w + x] = clamp01(value);
    }
  }

  /* ---- pass 2: measured colour differences ---- */
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      if ((y & 1) == r_row && (x & 1) == r_col) diff_r[i] = m[i] - g[i];
      if ((y & 1) == b_row && (x & 1) == b_col) diff_b[i] = m[i] - g[i];
    }
  }

  /* ---- pass 3: bilinear interpolation of the colour differences ---- */
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    const int yu = clamp_int(y - 1, 0, h - 1);
    const int yd = clamp_int(y + 1, 0, h - 1);
    for (int x = 0; x < w; ++x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      const int xl = clamp_int(x - 1, 0, w - 1);
      const int xr = clamp_int(x + 1, 0, w - 1);
      const size_t i_l = static_cast<size_t>(y) * w + xl;
      const size_t i_r = static_cast<size_t>(y) * w + xr;
      const size_t i_u = static_cast<size_t>(yu) * w + x;
      const size_t i_d = static_cast<size_t>(yd) * w + x;
      const size_t i_ul = static_cast<size_t>(yu) * w + xl;
      const size_t i_ur = static_cast<size_t>(yu) * w + xr;
      const size_t i_dl = static_cast<size_t>(yd) * w + xl;
      const size_t i_dr = static_cast<size_t>(yd) * w + xr;

      const bool same_r = ((y & 1) == r_row);
      const bool same_b = ((y & 1) == b_row);
      const bool col_r = ((x & 1) == r_col);
      const bool col_b = ((x & 1) == b_col);

      float vr, vb;
      if (same_r && col_r) {
        vr = diff_r[i];
      } else if (same_r) {
        vr = 0.5f * (diff_r[i_l] + diff_r[i_r]);
      } else if (col_r) {
        vr = 0.5f * (diff_r[i_u] + diff_r[i_d]);
      } else {
        vr = 0.25f * (diff_r[i_ul] + diff_r[i_ur] + diff_r[i_dl] + diff_r[i_dr]);
      }
      if (same_b && col_b) {
        vb = diff_b[i];
      } else if (same_b) {
        vb = 0.5f * (diff_b[i_l] + diff_b[i_r]);
      } else if (col_b) {
        vb = 0.5f * (diff_b[i_u] + diff_b[i_d]);
      } else {
        vb = 0.25f * (diff_b[i_ul] + diff_b[i_ur] + diff_b[i_dl] + diff_b[i_dr]);
      }
      interp_r[i] = vr;
      interp_b[i] = vb;
    }
  }

  /* ---- pass 4: 3x3 median of interpolated differences (keep measured) ---- */
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      float vals_r[9], vals_b[9];
      int n = 0;
      for (int dy = -1; dy <= 1; ++dy) {
        const int yy = clamp_int(y + dy, 0, h - 1);
        for (int dx = -1; dx <= 1; ++dx) {
          const int xx = clamp_int(x + dx, 0, w - 1);
          const size_t j = static_cast<size_t>(yy) * w + xx;
          vals_r[n] = interp_r[j];
          vals_b[n] = interp_b[j];
          ++n;
        }
      }
      const bool measured_r = ((y & 1) == r_row && (x & 1) == r_col);
      const bool measured_b = ((y & 1) == b_row && (x & 1) == b_col);
      if (!measured_r) interp_r[i] = median9(vals_r);
      if (!measured_b) interp_b[i] = median9(vals_b);
    }
  }

  /* ---- pass 5: reconstruct ---- */
  py::array_t<float> out({h, w, 3});
  float* rgb = static_cast<float*>(out.request().ptr);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t i = static_cast<size_t>(y) * w + x;
      rgb[3 * i + 0] = clamp01(g[i] + interp_r[i]);
      rgb[3 * i + 1] = clamp01(g[i]);
      rgb[3 * i + 2] = clamp01(g[i] + interp_b[i]);
    }
  }
  return out;
}

}  // namespace

PYBIND11_MODULE(_isp_accel, m) {
  m.doc() = "C++/OpenMP kernels for the Simple-ISP pipeline";
  m.def("bilateral_mosaic", &bilateral_mosaic,
        "RAW-domain bilateral filter on a CFA mosaic",
        py::arg("mosaic"), py::arg("pattern"),
        py::arg("sigma_color"), py::arg("sigma_space") = 2.0f, py::arg("radius") = 2);
  m.def("hamilton_adams", &hamilton_adams,
        "Gradient-directed demosaic (Hamilton-Adams)",
        py::arg("mosaic"), py::arg("pattern"), py::arg("color_desc") = "RGBG");
#ifdef _OPENMP
  m.def("max_threads", []() { return omp_get_max_threads(); });
#endif
}
