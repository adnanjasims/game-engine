#pragma once

#include "ai/tensor_buffer.hpp"
#include "core/export.hpp"
#include "core/module_interface.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <vector>

namespace eoc {

enum class InferenceBackend : std::uint8_t {
  Auto = 0,
  OnnxRuntime = 1,
  SimulatedNpu = 2,
  SimdFallback = 3
};

class EOC_API InferenceEngine {
 public:
  explicit InferenceEngine(InferenceBackend backend = InferenceBackend::Auto,
                           std::size_t worker_count = 2);
  ~InferenceEngine();

  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;

  [[nodiscard]] InferenceBackend backend() const noexcept;
  [[nodiscard]] const char* backend_name() const noexcept;
  [[nodiscard]] bool accelerator_available() const noexcept;
  [[nodiscard]] bool onnx_runtime_available() const noexcept;

  bool load_identity(int features);
  bool load_linear(const float* weights, const float* bias, int in_features, int out_features,
                   bool relu = true);
  bool load_onnx(const char* path);

  std::future<std::vector<float>> submit(const TensorBuffer& input);
  void submit(const TensorBuffer& input, std::function<void(std::vector<float>)> callback);

  static void simd_matmul(const float* a, const float* b, float* c, int n) noexcept;

 private:
  struct Impl;
  Impl* impl_;
};

class EOC_API InferenceModule : public ModuleBase {
 public:
  InferenceModule();
  bool startup() override;
  void shutdown() override;
  [[nodiscard]] InferenceEngine& engine() noexcept;
  [[nodiscard]] const InferenceEngine& engine() const noexcept;

 private:
  InferenceEngine engine_;
};

}  // namespace eoc
