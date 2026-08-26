#pragma once

#include "core/export.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace eoc {

[[nodiscard]] constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept {
  return (value + (alignment - 1u)) & ~(alignment - 1u);
}

[[nodiscard]] constexpr bool is_pow2(std::size_t value) noexcept {
  return value != 0 && (value & (value - 1u)) == 0;
}

// per-frame bump allocator. no individual free; call reset() each tick.
class EOC_API LinearArenaAllocator {
 public:
  static constexpr std::size_t kDefaultAlign = 16;
  static constexpr std::size_t kCacheLineAlign = 64;

  explicit LinearArenaAllocator(std::size_t capacity_bytes);
  ~LinearArenaAllocator();

  LinearArenaAllocator(const LinearArenaAllocator&) = delete;
  LinearArenaAllocator& operator=(const LinearArenaAllocator&) = delete;

  [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = kDefaultAlign) noexcept;
  void reset() noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t used() const noexcept;
  [[nodiscard]] std::size_t high_water_mark() const noexcept;
  void reset_high_water() noexcept;

 private:
  std::byte* buffer_;
  std::size_t capacity_;
  alignas(kCacheLineAlign) std::atomic<std::size_t> offset_;
  alignas(kCacheLineAlign) std::atomic<std::size_t> high_water_;
};

// fixed-size block pool with a lock-free treiber stack free list.
class EOC_API PoolAllocator {
 public:
  PoolAllocator(std::size_t block_size, std::size_t block_count, std::size_t alignment = 64);
  ~PoolAllocator();

  PoolAllocator(const PoolAllocator&) = delete;
  PoolAllocator& operator=(const PoolAllocator&) = delete;

  [[nodiscard]] void* allocate() noexcept;
  void deallocate(void* ptr) noexcept;

  [[nodiscard]] std::size_t block_size() const noexcept { return stride_; }
  [[nodiscard]] std::size_t block_count() const noexcept { return block_count_; }
  [[nodiscard]] std::size_t allocated_count() const noexcept;
  [[nodiscard]] bool owns(const void* ptr) const noexcept;

 private:
  static constexpr std::uint32_t kNullIndex = 0xffffffffu;

  [[nodiscard]] static std::uint64_t pack(std::uint32_t index, std::uint32_t tag) noexcept {
    return (static_cast<std::uint64_t>(tag) << 32) | static_cast<std::uint64_t>(index);
  }
  [[nodiscard]] static std::uint32_t unpack_index(std::uint64_t v) noexcept {
    return static_cast<std::uint32_t>(v);
  }
  [[nodiscard]] static std::uint32_t unpack_tag(std::uint64_t v) noexcept {
    return static_cast<std::uint32_t>(v >> 32);
  }

  std::byte* buffer_;
  std::uint32_t* next_index_;
  std::size_t stride_;
  std::size_t block_count_;
  std::size_t alignment_;
  alignas(64) std::atomic<std::uint64_t> head_;
  alignas(64) std::atomic<std::size_t> allocated_;
};

// lifo marker allocator. rewind() releases everything past a marker.
class EOC_API StackAllocator {
 public:
  struct Marker {
    std::size_t offset;
  };

  explicit StackAllocator(std::size_t capacity_bytes);
  ~StackAllocator();

  StackAllocator(const StackAllocator&) = delete;
  StackAllocator& operator=(const StackAllocator&) = delete;

  [[nodiscard]] void* allocate(std::size_t size, std::size_t alignment = 16) noexcept;
  [[nodiscard]] Marker marker() const noexcept;
  void rewind(Marker marker) noexcept;
  void reset() noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t used() const noexcept { return offset_; }

 private:
  std::byte* buffer_;
  std::size_t capacity_;
  std::size_t offset_;
};

}  // namespace eoc
