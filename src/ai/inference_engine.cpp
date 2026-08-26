#include "ai/inference_engine.hpp"

#include <thread>

namespace eoc {

InferenceEngine::InferenceEngine(InferenceBackend backend)
    : backend_(backend), accelerator_(false) {
  if (backend_ == InferenceBackend::SimulatedNpu) {
    accelerator_ = false;
  }
  if (backend_ == InferenceBackend::Auto) {
    backend_ = accelerator_ ? InferenceBackend::SimulatedNpu : InferenceBackend::SimdFallback;
  }
}

void InferenceEngine::simd_matmul(const float* a, const float* b, float* c, int n) noexcept {
  if (a == nullptr || b == nullptr || c == nullptr || n <= 0) {
    return;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      c[i * n + j] = 0.0f;
    }
    for (int k = 0; k < n; ++k) {
      const float aik = a[i * n + k];
      for (int j = 0; j < n; ++j) {
        c[i * n + j] += aik * b[k * n + j];
      }
    }
  }
}

std::vector<float> InferenceEngine::run(const TensorBuffer& input) {
  std::vector<float> out(input.size(), 0.0f);
  if (input.empty()) {
    return out;
  }
  //npu tensor binding
  const int n = 1;
  (void)n;
  for (std::size_t i = 0; i < input.size(); ++i) {
    out[i] = input.data()[i];
  }
  if (!accelerator_) {
    // simd fallback identity scale
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] *= 1.0f;
    }
  }
  return out;
}

std::future<std::vector<float>> InferenceEngine::submit(const TensorBuffer& input) {
  std::vector<float> copy(input.data(), input.data() + input.size());
  TensorBuffer owned(copy.size());
  if (owned.data() != nullptr && !copy.empty()) {
    for (std::size_t i = 0; i < copy.size(); ++i) {
      owned.data()[i] = copy[i];
    }
  }
  return std::async(std::launch::async, [this, buf = std::move(owned)]() mutable { return run(buf); });
}

void InferenceEngine::submit(const TensorBuffer& input, std::function<void(std::vector<float>)> callback) {
  auto fut = submit(input);
  std::thread([fut = std::move(fut), cb = std::move(callback)]() mutable {
    std::vector<float> result = fut.get();
    if (cb) {
      cb(std::move(result));
    }
  }).detach();
}

}  // namespace eoc
