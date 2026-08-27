#include "multimedia/zero_copy_ring.hpp"

#include "core/memory_allocators.hpp"

#include <cstring>
#include <new>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace eoc {
namespace {

std::size_t next_pow2(std::size_t value) noexcept {
  if (value <= 1) {
    return 1;
  }
  --value;
  for (std::size_t i = 1; i < sizeof(std::size_t) * 8; i <<= 1) {
    value |= value >> i;
  }
  return value + 1;
}

void* ring_alloc(std::size_t bytes) {
  constexpr std::size_t kAlign = 64;
  const std::size_t size = align_up(bytes == 0 ? kAlign : bytes, kAlign);
#if defined(_WIN32)
  return _aligned_malloc(size, kAlign);
#else
  return std::aligned_alloc(kAlign, size);
#endif
}

void ring_free(void* p) noexcept {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

void copy_into(RingView view, const std::byte* src, std::size_t n) noexcept {
  const std::size_t n1 = n < view.first.size() ? n : view.first.size();
  if (n1 > 0) {
    std::memcpy(view.first.data(), src, n1);
  }
  if (n > n1 && !view.second.empty()) {
    std::memcpy(view.second.data(), src + n1, n - n1);
  }
}

void copy_from(RingView view, std::byte* dst, std::size_t n) noexcept {
  const std::size_t n1 = n < view.first.size() ? n : view.first.size();
  if (n1 > 0) {
    std::memcpy(dst, view.first.data(), n1);
  }
  if (n > n1 && !view.second.empty()) {
    std::memcpy(dst + n1, view.second.data(), n - n1);
  }
}

}  // namespace

ZeroCopyRing::ZeroCopyRing(std::size_t capacity_bytes)
    : buffer_(nullptr),
      capacity_(next_pow2(capacity_bytes == 0 ? 4096 : capacity_bytes)),
      mask_(capacity_ - 1),
      write_pos_(0),
      read_pos_(0) {
  buffer_ = static_cast<std::byte*>(ring_alloc(capacity_));
  if (buffer_ == nullptr) {
    throw std::bad_alloc();
  }
}

ZeroCopyRing::~ZeroCopyRing() {
  ring_free(buffer_);
}

std::size_t ZeroCopyRing::size() const noexcept {
  const std::size_t w = write_pos_.load(std::memory_order_acquire);
  const std::size_t r = read_pos_.load(std::memory_order_acquire);
  return w - r;
}

std::size_t ZeroCopyRing::free_space() const noexcept {
  return capacity_ - size();
}

RingView ZeroCopyRing::acquire_write(std::size_t bytes) noexcept {
  if (buffer_ == nullptr || bytes == 0) {
    return {};
  }
  const std::size_t w = write_pos_.load(std::memory_order_relaxed);
  const std::size_t r = read_pos_.load(std::memory_order_acquire);
  const std::size_t used = w - r;
  const std::size_t free = capacity_ - used;
  if (bytes > free) {
    bytes = free;
  }
  if (bytes == 0) {
    return {};
  }
  //zero copy wrap
  const std::size_t idx = w & mask_;
  const std::size_t first = bytes < (capacity_ - idx) ? bytes : (capacity_ - idx);
  RingView view;
  view.first = {buffer_ + idx, first};
  if (first < bytes) {
    view.second = {buffer_, bytes - first};
  }
  return view;
}

void ZeroCopyRing::commit_write(std::size_t bytes) noexcept {
  if (bytes == 0) {
    return;
  }
  const std::size_t free = free_space();
  if (bytes > free) {
    bytes = free;
  }
  write_pos_.fetch_add(bytes, std::memory_order_release);
}

RingView ZeroCopyRing::acquire_read(std::size_t bytes) noexcept {
  if (buffer_ == nullptr || bytes == 0) {
    return {};
  }
  const std::size_t r = read_pos_.load(std::memory_order_relaxed);
  const std::size_t w = write_pos_.load(std::memory_order_acquire);
  const std::size_t used = w - r;
  if (bytes > used) {
    bytes = used;
  }
  if (bytes == 0) {
    return {};
  }
  const std::size_t idx = r & mask_;
  const std::size_t first = bytes < (capacity_ - idx) ? bytes : (capacity_ - idx);
  RingView view;
  view.first = {buffer_ + idx, first};
  if (first < bytes) {
    view.second = {buffer_, bytes - first};
  }
  return view;
}

void ZeroCopyRing::commit_read(std::size_t bytes) noexcept {
  if (bytes == 0) {
    return;
  }
  const std::size_t used = size();
  if (bytes > used) {
    bytes = used;
  }
  read_pos_.fetch_add(bytes, std::memory_order_release);
}

std::size_t ZeroCopyRing::write(const std::byte* src, std::size_t bytes) noexcept {
  if (src == nullptr || bytes == 0) {
    return 0;
  }
  RingView view = acquire_write(bytes);
  const std::size_t n = view.size();
  copy_into(view, src, n);
  commit_write(n);
  return n;
}

std::size_t ZeroCopyRing::read(std::byte* dst, std::size_t bytes) noexcept {
  if (dst == nullptr || bytes == 0) {
    return 0;
  }
  RingView view = acquire_read(bytes);
  const std::size_t n = view.size();
  copy_from(view, dst, n);
  commit_read(n);
  return n;
}

std::size_t ZeroCopyRing::peek(std::byte* dst, std::size_t bytes) const noexcept {
  if (dst == nullptr || bytes == 0 || buffer_ == nullptr) {
    return 0;
  }
  const std::size_t r = read_pos_.load(std::memory_order_relaxed);
  const std::size_t w = write_pos_.load(std::memory_order_acquire);
  const std::size_t used = w - r;
  if (bytes > used) {
    bytes = used;
  }
  if (bytes == 0) {
    return 0;
  }
  const std::size_t idx = r & mask_;
  const std::size_t first = bytes < (capacity_ - idx) ? bytes : (capacity_ - idx);
  std::memcpy(dst, buffer_ + idx, first);
  if (first < bytes) {
    std::memcpy(dst + first, buffer_, bytes - first);
  }
  return bytes;
}

}  // namespace eoc
