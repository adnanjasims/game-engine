#pragma once

#include "core/export.hpp"

#include <cstddef>

namespace eoc {

EOC_API void simd_matmul(const float* a, const float* b, float* c, int n) noexcept;
EOC_API void simd_matmul_mnk(const float* a, int m, int k, const float* b, int n, float* c) noexcept;
EOC_API void simd_gemv(const float* weights, int rows, int cols, const float* x, const float* bias,
                       float* y) noexcept;
EOC_API void simd_relu(float* data, std::size_t count) noexcept;
EOC_API void simd_add(const float* a, const float* b, float* c, std::size_t count) noexcept;

}  // namespace eoc
