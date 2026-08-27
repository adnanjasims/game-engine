#pragma once

#include "core/export.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>

namespace eoc {

struct RingView {
  std::span<std::byte> first;
  std::span<std::byte> second;

  [[nodiscard]] std::size_t size() const noexcept { return first.size() + second.size(); }
  [[nodiscard]] bool empty() const noexcept { return first.empty() && second.empty(); }
};

class EOC_API ZeroCopyRing {
 public:
  explicit ZeroCopyRing(std::size_t capacity_bytes);
  ~ZeroCopyRing();

  ZeroCopyRing(const ZeroCopyRing&) = delete;
  ZeroCopyRing& operator=(const ZeroCopyRing&) = delete;

  [[nodiscard]] RingView acquire_write(std::size_t bytes) noexcept;
  void commit_write(std::size_t bytes) noexcept;
  [[nodiscard]] RingView acquire_read(std::size_t bytes) noexcept;
  void commit_read(std::size_t bytes) noexcept;

  std::size_t write(const std::byte* src, std::size_t bytes) noexcept;
  std::size_t read(std::byte* dst, std::size_t bytes) noexcept;
  std::size_t peek(std::byte* dst, std::size_t bytes) const noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t free_space() const noexcept;

 private:
  std::byte* buffer_;
  std::size_t capacity_;
  std::size_t mask_;
  alignas(64) std::atomic<std::size_t> write_pos_;
  alignas(64) std::atomic<std::size_t> read_pos_;
};

}  // namespace eoc
