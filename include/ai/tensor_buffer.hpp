#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>

namespace eoc {

class PoolAllocator;

class EOC_API TensorBuffer {
 public:
  TensorBuffer() = default;
  explicit TensorBuffer(std::size_t element_count);
  TensorBuffer(PoolAllocator& pool, std::size_t element_count);
  TensorBuffer(float* external, std::size_t element_count) noexcept;
  TensorBuffer(TensorBuffer&& other) noexcept;
  TensorBuffer& operator=(TensorBuffer&& other) noexcept;
  ~TensorBuffer();

  TensorBuffer(const TensorBuffer&) = delete;
  TensorBuffer& operator=(const TensorBuffer&) = delete;

  [[nodiscard]] float* data() noexcept { return data_; }
  [[nodiscard]] const float* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] bool pooled() const noexcept { return pool_ != nullptr; }

  void fill(float value) noexcept;
  void copy_from(const float* src, std::size_t count) noexcept;
  void copy_to(float* dst, std::size_t count) const noexcept;

 private:
  void release() noexcept;

  float* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
  PoolAllocator* pool_ = nullptr;
  bool owns_ = false;
};

}  // namespace eoc
