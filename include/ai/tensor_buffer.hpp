#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace eoc {

class EOC_API TensorBuffer {
 public:
  TensorBuffer() = default;
  explicit TensorBuffer(std::size_t element_count);
  TensorBuffer(float* external, std::size_t element_count) noexcept;

  [[nodiscard]] float* data() noexcept { return data_; }
  [[nodiscard]] const float* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

 private:
  std::unique_ptr<float[]> owned_;
  float* data_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace eoc
