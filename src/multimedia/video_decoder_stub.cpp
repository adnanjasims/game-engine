#include "multimedia/video_decoder_stub.hpp"

namespace eoc {

VideoDecoderStub::VideoDecoderStub() : open_(false), stub_pixel_(0) {}

bool VideoDecoderStub::open(const char* uri) {
  open_ = uri != nullptr && uri[0] != '\0';
  stub_pixel_ = 0;
  return open_;
}

void VideoDecoderStub::close() {
  open_ = false;
}

bool VideoDecoderStub::decode_next(VideoFrame& out) noexcept {
  if (!open_) {
    out = VideoFrame{nullptr, 0, 0, 0};
    return false;
  }
  stub_pixel_ = static_cast<std::uint8_t>(stub_pixel_ + 1);
  out.data = &stub_pixel_;
  out.width = 1;
  out.height = 1;
  out.stride = 1;
  return true;
}

}  // namespace eoc
