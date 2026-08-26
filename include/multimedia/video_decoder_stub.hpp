#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace eoc {

struct VideoFrame {
  const std::uint8_t* data;
  std::size_t width;
  std::size_t height;
  std::size_t stride;
};

class EOC_API VideoDecoderStub {
 public:
  VideoDecoderStub();
  bool open(const char* uri);
  void close();
  [[nodiscard]] bool decode_next(VideoFrame& out) noexcept;
  [[nodiscard]] bool is_open() const noexcept { return open_; }

 private:
  bool open_;
  std::uint8_t stub_pixel_;
};

}  // namespace eoc
