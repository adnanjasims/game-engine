#include "ai/tensor_buffer.hpp"

namespace eoc {

TensorBuffer::TensorBuffer(std::size_t element_count)
    : owned_(element_count > 0 ? std::make_unique<float[]>(element_count) : nullptr),
      data_(owned_.get()),
      size_(element_count) {
  if (data_ != nullptr) {
    for (std::size_t i = 0; i < size_; ++i) {
      data_[i] = 0.0f;
    }
  }
}

TensorBuffer::TensorBuffer(float* external, std::size_t element_count) noexcept
    : data_(external), size_(element_count) {}

}  // namespace eoc
