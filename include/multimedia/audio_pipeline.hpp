#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace eoc {

class EOC_API AudioPipeline {
 public:
  explicit AudioPipeline(std::size_t capacity_bytes = 65536);
  ~AudioPipeline();

  AudioPipeline(const AudioPipeline&) = delete;
  AudioPipeline& operator=(const AudioPipeline&) = delete;

  [[nodiscard]] std::span<std::byte> acquire_write(std::size_t bytes) noexcept;
  void commit_write(std::size_t bytes) noexcept;
  [[nodiscard]] std::span<const std::byte> acquire_read(std::size_t bytes) noexcept;
  void commit_read(std::size_t bytes) noexcept;

  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
  [[nodiscard]] std::size_t size() const noexcept;

 private:
  std::byte* buffer_;
  std::size_t capacity_;
  std::size_t read_pos_;
  std::size_t write_pos_;
  std::size_t filled_;
};

}  // namespace eoc
