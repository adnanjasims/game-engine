#pragma once

#include "core/export.hpp"
#include "multimedia/zero_copy_ring.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace eoc {

struct VideoFrame {
  const std::uint8_t* data;
  const std::uint8_t* y;
  const std::uint8_t* u;
  const std::uint8_t* v;
  std::size_t width;
  std::size_t height;
  std::size_t stride_y;
  std::size_t stride_uv;
  std::uint64_t pts;
};

class EOC_API VideoDecoderStub {
 public:
  static constexpr int kFrameSlots = 3;

  VideoDecoderStub();
  ~VideoDecoderStub() = default;

  bool open(const char* uri);
  bool open_pattern(std::size_t width, std::size_t height);
  void close();

  bool push_packet(const std::byte* data, std::size_t bytes, std::uint64_t pts);
  [[nodiscard]] bool decode_next(VideoFrame& out) noexcept;
  [[nodiscard]] bool is_open() const noexcept { return open_; }
  [[nodiscard]] std::size_t width() const noexcept { return width_; }
  [[nodiscard]] std::size_t height() const noexcept { return height_; }
  [[nodiscard]] std::size_t frame_bytes() const noexcept;
  [[nodiscard]] std::size_t queued_bytes() const noexcept;
  [[nodiscard]] std::uint64_t frames_decoded() const noexcept { return frames_decoded_; }

 private:
  void setup_buffers();
  void fill_pattern(std::uint8_t* dst) noexcept;

  bool open_;
  bool pattern_;
  std::size_t width_;
  std::size_t height_;
  int slot_;
  std::uint64_t frames_decoded_;
  std::uint64_t next_pts_;
  ZeroCopyRing packets_;
  std::vector<std::uint8_t> frames_;
};

using VideoDecoder = VideoDecoderStub;

}  // namespace eoc
