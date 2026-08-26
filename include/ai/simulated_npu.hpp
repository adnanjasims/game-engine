#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>

namespace eoc {

enum class NpuMemcpyKind : std::uint8_t {
  HostToDevice = 0,
  DeviceToHost = 1,
  DeviceToDevice = 2
};

class EOC_API SimulatedNpu {
 public:
  SimulatedNpu();
  ~SimulatedNpu();

  SimulatedNpu(const SimulatedNpu&) = delete;
  SimulatedNpu& operator=(const SimulatedNpu&) = delete;

  [[nodiscard]] bool malloc(std::size_t bytes, void** out) noexcept;
  void free(void* ptr) noexcept;

  bool memcpy(void* dst, const void* src, std::size_t bytes, NpuMemcpyKind kind) noexcept;

  bool gemv_relu(const float* weights, int rows, int cols, const float* x, const float* bias,
                 float* y) noexcept;
  bool matmul(const float* a, const float* b, float* c, int n) noexcept;
  void synchronize() noexcept;

  [[nodiscard]] std::uint64_t last_op_us() const noexcept { return last_op_us_; }
  [[nodiscard]] std::size_t device_bytes() const noexcept;

 private:
  struct Impl;
  Impl* impl_;
  std::uint64_t last_op_us_;
};

}  // namespace eoc
