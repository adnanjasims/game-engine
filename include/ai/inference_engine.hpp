#pragma once

#include "ai/tensor_buffer.hpp"
#include "core/export.hpp"

#include <cstdint>
#include <functional>
#include <future>
#include <vector>

namespace eoc {

enum class InferenceBackend : std::uint8_t {
  Auto = 0,
  SimulatedNpu = 1,
  SimdFallback = 2
};

class EOC_API InferenceEngine {
 public:
  explicit InferenceEngine(InferenceBackend backend = InferenceBackend::Auto);

  [[nodiscard]] InferenceBackend backend() const noexcept { return backend_; }
  [[nodiscard]] bool accelerator_available() const noexcept { return accelerator_; }

  std::future<std::vector<float>> submit(const TensorBuffer& input);
  void submit(const TensorBuffer& input, std::function<void(std::vector<float>)> callback);

  static void simd_matmul(const float* a, const float* b, float* c, int n) noexcept;

 private:
  std::vector<float> run(const TensorBuffer& input);

  InferenceBackend backend_;
  bool accelerator_;
};

}  // namespace eoc
