#include "ai/tensor_buffer.hpp"

#include "core/memory_allocators.hpp"

#include <cstring>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace eoc {
namespace {

void* tensor_aligned_alloc(std::size_t bytes) {
  constexpr std::size_t kAlign = 64;
  const std::size_t size = align_up(bytes == 0 ? kAlign : bytes, kAlign);
#if defined(_WIN32)
  return _aligned_malloc(size, kAlign);
#else
  return std::aligned_alloc(kAlign, size);
#endif
}

void tensor_aligned_free(void* p) noexcept {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

}  // namespace

void TensorBuffer::release() noexcept {
  if (data_ == nullptr) {
    return;
  }
  if (pool_ != nullptr) {
    pool_->deallocate(data_);
  } else if (owns_) {
    tensor_aligned_free(data_);
  }
  data_ = nullptr;
  size_ = 0;
  capacity_ = 0;
  pool_ = nullptr;
  owns_ = false;
}

TensorBuffer::TensorBuffer(std::size_t element_count)
    : data_(nullptr), size_(element_count), capacity_(element_count), pool_(nullptr), owns_(false) {
  if (element_count == 0) {
    return;
  }
  data_ = static_cast<float*>(tensor_aligned_alloc(element_count * sizeof(float)));
  if (data_ == nullptr) {
    size_ = 0;
    capacity_ = 0;
    return;
  }
  owns_ = true;
  fill(0.0f);
}

TensorBuffer::TensorBuffer(PoolAllocator& pool, std::size_t element_count)
    : data_(nullptr), size_(0), capacity_(0), pool_(nullptr), owns_(false) {
  const std::size_t bytes = element_count * sizeof(float);
  if (element_count == 0) {
    return;
  }
  if (bytes <= pool.block_size()) {
    void* mem = pool.allocate();
    if (mem != nullptr) {
      data_ = static_cast<float*>(mem);
      size_ = element_count;
      capacity_ = pool.block_size() / sizeof(float);
      pool_ = &pool;
      fill(0.0f);
      return;
    }
  }
  data_ = static_cast<float*>(tensor_aligned_alloc(bytes));
  if (data_ != nullptr) {
    size_ = element_count;
    capacity_ = element_count;
    owns_ = true;
    fill(0.0f);
  }
}

TensorBuffer::TensorBuffer(float* external, std::size_t element_count) noexcept
    : data_(external), size_(element_count), capacity_(element_count), pool_(nullptr), owns_(false) {}

TensorBuffer::TensorBuffer(TensorBuffer&& other) noexcept
    : data_(other.data_),
      size_(other.size_),
      capacity_(other.capacity_),
      pool_(other.pool_),
      owns_(other.owns_) {
  other.data_ = nullptr;
  other.size_ = 0;
  other.capacity_ = 0;
  other.pool_ = nullptr;
  other.owns_ = false;
}

TensorBuffer& TensorBuffer::operator=(TensorBuffer&& other) noexcept {
  if (this != &other) {
    release();
    data_ = other.data_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    pool_ = other.pool_;
    owns_ = other.owns_;
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    other.pool_ = nullptr;
    other.owns_ = false;
  }
  return *this;
}

TensorBuffer::~TensorBuffer() {
  release();
}

void TensorBuffer::fill(float value) noexcept {
  if (data_ == nullptr) {
    return;
  }
  for (std::size_t i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

void TensorBuffer::copy_from(const float* src, std::size_t count) noexcept {
  if (data_ == nullptr || src == nullptr) {
    return;
  }
  const std::size_t n = count < size_ ? count : size_;
  std::memcpy(data_, src, n * sizeof(float));
}

void TensorBuffer::copy_to(float* dst, std::size_t count) const noexcept {
  if (data_ == nullptr || dst == nullptr) {
    return;
  }
  const std::size_t n = count < size_ ? count : size_;
  std::memcpy(dst, data_, n * sizeof(float));
}

}  // namespace eoc
