#include "ai/simd_kernels.hpp"

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace eoc {

void simd_relu(float* data, std::size_t count) noexcept {
  if (data == nullptr) {
    return;
  }
  std::size_t i = 0;
#if defined(__AVX2__)
  const __m256 z = _mm256_setzero_ps();
  for (; i + 8 <= count; i += 8) {
    const __m256 v = _mm256_loadu_ps(data + i);
    _mm256_storeu_ps(data + i, _mm256_max_ps(v, z));
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float32x4_t z = vdupq_n_f32(0.0f);
  for (; i + 4 <= count; i += 4) {
    const float32x4_t v = vld1q_f32(data + i);
    vst1q_f32(data + i, vmaxq_f32(v, z));
  }
#endif
  for (; i < count; ++i) {
    if (data[i] < 0.0f) {
      data[i] = 0.0f;
    }
  }
}

void simd_add(const float* a, const float* b, float* c, std::size_t count) noexcept {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return;
  }
  std::size_t i = 0;
#if defined(__AVX2__)
  for (; i + 8 <= count; i += 8) {
    _mm256_storeu_ps(c + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  for (; i + 4 <= count; i += 4) {
    vst1q_f32(c + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
  }
#endif
  for (; i < count; ++i) {
    c[i] = a[i] + b[i];
  }
}

void simd_gemv(const float* weights, int rows, int cols, const float* x, const float* bias,
               float* y) noexcept {
  if (weights == nullptr || x == nullptr || y == nullptr || rows <= 0 || cols <= 0) {
    return;
  }
  for (int r = 0; r < rows; ++r) {
    const float* wr = weights + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    int c = 0;
    float sum = bias != nullptr ? bias[r] : 0.0f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; c + 8 <= cols; c += 8) {
      const __m256 wv = _mm256_loadu_ps(wr + c);
      const __m256 xv = _mm256_loadu_ps(x + c);
#if defined(__FMA__)
      acc = _mm256_fmadd_ps(wv, xv, acc);
#else
      acc = _mm256_add_ps(acc, _mm256_mul_ps(wv, xv));
#endif
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; c + 4 <= cols; c += 4) {
      //simd gemv inner loop
      acc = vmlaq_f32(acc, vld1q_f32(wr + c), vld1q_f32(x + c));
    }
#if defined(__aarch64__)
    sum += vaddvq_f32(acc);
#else
    const float32x2_t pair = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum += vget_lane_f32(vpadd_f32(pair, pair), 0);
#endif
#endif
    for (; c < cols; ++c) {
      sum += wr[c] * x[c];
    }
    y[r] = sum;
  }
}

void simd_matmul_mnk(const float* a, int m, int k, const float* b, int n, float* c) noexcept {
  if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || k <= 0 || n <= 0) {
    return;
  }
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i * n + j] = 0.0f;
    }
    for (int p = 0; p < k; ++p) {
      const float aip = a[i * k + p];
      const float* bp = b + p * n;
      float* ci = c + i * n;
      int j = 0;
#if defined(__AVX2__)
      const __m256 av = _mm256_set1_ps(aip);
      for (; j + 8 <= n; j += 8) {
        const __m256 bv = _mm256_loadu_ps(bp + j);
        const __m256 cv = _mm256_loadu_ps(ci + j);
#if defined(__FMA__)
        _mm256_storeu_ps(ci + j, _mm256_fmadd_ps(av, bv, cv));
#else
        _mm256_storeu_ps(ci + j, _mm256_add_ps(cv, _mm256_mul_ps(av, bv)));
#endif
      }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
      const float32x4_t av = vdupq_n_f32(aip);
      for (; j + 4 <= n; j += 4) {
        const float32x4_t cv = vld1q_f32(ci + j);
        vst1q_f32(ci + j, vmlaq_f32(cv, av, vld1q_f32(bp + j)));
      }
#endif
      for (; j < n; ++j) {
        ci[j] += aip * bp[j];
      }
    }
  }
}

void simd_matmul(const float* a, const float* b, float* c, int n) noexcept {
  simd_matmul_mnk(a, n, n, b, n, c);
}

}  // namespace eoc
