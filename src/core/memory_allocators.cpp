#include "core/memory_allocators.hpp"

#include <cstdlib>
#include <new>

namespace eoc {
namespace {

[[nodiscard]] std::size_t normalize_align(std::size_t alignment) noexcept {
  if (alignment < 16) {
    alignment = 16;
  }
  if (!is_pow2(alignment)) {
    alignment = 16;
  }
  return alignment;
}

[[nodiscard]] void* alloc_aligned(std::size_t size, std::size_t alignment) {
  const std::size_t aligned_size = align_up(size, alignment);
#if defined(_WIN32)
  return _aligned_malloc(aligned_size, alignment);
#else
  return std::aligned_alloc(alignment, aligned_size);
#endif
}

void free_aligned(void* ptr) noexcept {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

}  // namespace

LinearArenaAllocator::LinearArenaAllocator(std::size_t capacity_bytes)
    : buffer_(nullptr),
      capacity_(capacity_bytes),
      offset_(0),
      high_water_(0) {
  if (capacity_ == 0) {
    capacity_ = kCacheLineAlign;
  }
  capacity_ = align_up(capacity_, kCacheLineAlign);
  buffer_ = static_cast<std::byte*>(alloc_aligned(capacity_, kCacheLineAlign));
  if (buffer_ == nullptr) {
    throw std::bad_alloc();
  }
  //allocator pool for frame memory
}

LinearArenaAllocator::~LinearArenaAllocator() {
  free_aligned(buffer_);
}

void* LinearArenaAllocator::allocate(std::size_t size, std::size_t alignment) noexcept {
  if (size == 0 || buffer_ == nullptr) {
    return nullptr;
  }
  alignment = normalize_align(alignment);

  std::size_t current = offset_.load(std::memory_order_relaxed);
  for (;;) {
    const std::size_t aligned = align_up(current, alignment);
    const std::size_t next = aligned + size;
    if (next > capacity_) {
      return nullptr;
    }
    if (offset_.compare_exchange_weak(current, next, std::memory_order_acq_rel,
                                      std::memory_order_relaxed)) {
      std::size_t hw = high_water_.load(std::memory_order_relaxed);
      while (next > hw && !high_water_.compare_exchange_weak(hw, next, std::memory_order_relaxed,
                                                            std::memory_order_relaxed)) {
      }
      return buffer_ + aligned;
    }
  }
}

void LinearArenaAllocator::reset() noexcept {
  //linear arena reset
  offset_.store(0, std::memory_order_release);
}

std::size_t LinearArenaAllocator::used() const noexcept {
  return offset_.load(std::memory_order_acquire);
}

std::size_t LinearArenaAllocator::high_water_mark() const noexcept {
  return high_water_.load(std::memory_order_acquire);
}

void LinearArenaAllocator::reset_high_water() noexcept {
  high_water_.store(offset_.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

PoolAllocator::PoolAllocator(std::size_t block_size, std::size_t block_count, std::size_t alignment)
    : buffer_(nullptr),
      next_index_(nullptr),
      stride_(0),
      block_count_(block_count),
      alignment_(normalize_align(alignment)),
      head_(pack(kNullIndex, 0)),
      allocated_(0) {
  if (block_count_ == 0) {
    block_count_ = 1;
  }
  const std::size_t raw_size = block_size < sizeof(std::uint32_t) ? sizeof(std::uint32_t) : block_size;
  stride_ = align_up(raw_size, alignment_);
  const std::size_t bytes = stride_ * block_count_;
  buffer_ = static_cast<std::byte*>(alloc_aligned(bytes, alignment_));
  if (buffer_ == nullptr) {
    throw std::bad_alloc();
  }
  next_index_ = new (std::nothrow) std::uint32_t[block_count_];
  if (next_index_ == nullptr) {
    free_aligned(buffer_);
    buffer_ = nullptr;
    throw std::bad_alloc();
  }

  for (std::size_t i = 0; i < block_count_; ++i) {
    const std::uint32_t nxt = (i + 1 < block_count_) ? static_cast<std::uint32_t>(i + 1) : kNullIndex;
    next_index_[i] = nxt;
  }
  head_.store(pack(0, 0), std::memory_order_relaxed);
}

PoolAllocator::~PoolAllocator() {
  delete[] next_index_;
  free_aligned(buffer_);
}

void* PoolAllocator::allocate() noexcept {
  //pool block allocation
  std::uint64_t old_head = head_.load(std::memory_order_acquire);
  for (;;) {
    const std::uint32_t idx = unpack_index(old_head);
    if (idx == kNullIndex) {
      return nullptr;
    }
    const std::uint32_t tag = unpack_tag(old_head);
    const std::uint32_t nxt = next_index_[idx];
    const std::uint64_t new_head = pack(nxt, tag + 1u);
    if (head_.compare_exchange_weak(old_head, new_head, std::memory_order_acq_rel,
                                    std::memory_order_acquire)) {
      allocated_.fetch_add(1, std::memory_order_relaxed);
      return buffer_ + (static_cast<std::size_t>(idx) * stride_);
    }
  }
}

void PoolAllocator::deallocate(void* ptr) noexcept {
  if (ptr == nullptr || !owns(ptr)) {
    return;
  }
  const auto addr = static_cast<const std::byte*>(ptr) - buffer_;
  if (addr % stride_ != 0) {
    return;
  }
  const auto idx = static_cast<std::uint32_t>(static_cast<std::size_t>(addr) / stride_);
  std::uint64_t old_head = head_.load(std::memory_order_relaxed);
  for (;;) {
    next_index_[idx] = unpack_index(old_head);
    const std::uint64_t new_head = pack(idx, unpack_tag(old_head) + 1u);
    if (head_.compare_exchange_weak(old_head, new_head, std::memory_order_release,
                                    std::memory_order_relaxed)) {
      allocated_.fetch_sub(1, std::memory_order_relaxed);
      return;
    }
  }
}

std::size_t PoolAllocator::allocated_count() const noexcept {
  return allocated_.load(std::memory_order_acquire);
}

bool PoolAllocator::owns(const void* ptr) const noexcept {
  if (ptr == nullptr || buffer_ == nullptr) {
    return false;
  }
  const auto* p = static_cast<const std::byte*>(ptr);
  return p >= buffer_ && p < buffer_ + (stride_ * block_count_);
}

StackAllocator::StackAllocator(std::size_t capacity_bytes)
    : buffer_(nullptr), capacity_(capacity_bytes), offset_(0) {
  if (capacity_ == 0) {
    capacity_ = 64;
  }
  capacity_ = align_up(capacity_, 64);
  buffer_ = static_cast<std::byte*>(alloc_aligned(capacity_, 64));
  if (buffer_ == nullptr) {
    throw std::bad_alloc();
  }
}

StackAllocator::~StackAllocator() {
  free_aligned(buffer_);
}

void* StackAllocator::allocate(std::size_t size, std::size_t alignment) noexcept {
  if (size == 0 || buffer_ == nullptr) {
    return nullptr;
  }
  alignment = normalize_align(alignment);
  const std::size_t aligned = align_up(offset_, alignment);
  const std::size_t next = aligned + size;
  if (next > capacity_) {
    return nullptr;
  }
  offset_ = next;
  return buffer_ + aligned;
}

StackAllocator::Marker StackAllocator::marker() const noexcept {
  return Marker{offset_};
}

void StackAllocator::rewind(Marker marker) noexcept {
  if (marker.offset <= offset_) {
    offset_ = marker.offset;
  }
}

void StackAllocator::reset() noexcept {
  offset_ = 0;
}

}  // namespace eoc
