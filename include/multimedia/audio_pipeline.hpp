#pragma once

#include "core/export.hpp"
#include "multimedia/zero_copy_ring.hpp"

#include <cstddef>
#include <cstdint>

namespace eoc {

struct AudioFormat {
  std::uint32_t sample_rate = 48000;
  std::uint16_t channels = 2;
  std::uint16_t bytes_per_sample = 2;
};

class EOC_API AudioPipeline {
 public:
  explicit AudioPipeline(std::size_t capacity_bytes = 65536, AudioFormat format = {});
  ~AudioPipeline() = default;

  AudioPipeline(const AudioPipeline&) = delete;
  AudioPipeline& operator=(const AudioPipeline&) = delete;

  [[nodiscard]] RingView acquire_write(std::size_t bytes) noexcept;
  void commit_write(std::size_t bytes) noexcept;
  [[nodiscard]] RingView acquire_read(std::size_t bytes) noexcept;
  void commit_read(std::size_t bytes) noexcept;

  std::size_t write_bytes(const std::byte* src, std::size_t bytes) noexcept;
  std::size_t read_bytes(std::byte* dst, std::size_t bytes) noexcept;
  std::size_t write_s16(const std::int16_t* samples, std::size_t count) noexcept;
  std::size_t read_s16(std::int16_t* samples, std::size_t count) noexcept;

  [[nodiscard]] AudioFormat format() const noexcept { return format_; }
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t frames() const noexcept;

 private:
  AudioFormat format_;
  ZeroCopyRing ring_;
};

}  // namespace eoc
