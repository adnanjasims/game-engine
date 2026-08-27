#include "multimedia/video_decoder_stub.hpp"

#include <cstring>

namespace eoc {
namespace {

std::size_t i420_size(std::size_t width, std::size_t height) noexcept {
  return width * height + (width * height) / 2;
}

void drain_bytes(ZeroCopyRing& ring, std::size_t bytes) noexcept {
  std::byte tmp[64];
  while (bytes > 0) {
    const std::size_t n = bytes < sizeof(tmp) ? bytes : sizeof(tmp);
    const std::size_t got = ring.read(tmp, n);
    if (got == 0) {
      break;
    }
    bytes -= got;
  }
}

}  // namespace

VideoDecoderStub::VideoDecoderStub()
    : open_(false),
      pattern_(false),
      width_(0),
      height_(0),
      slot_(0),
      frames_decoded_(0),
      next_pts_(0),
      packets_(1u << 16) {}

std::size_t VideoDecoderStub::frame_bytes() const noexcept {
  if (width_ == 0 || height_ == 0) {
    return 0;
  }
  return i420_size(width_, height_);
}

std::size_t VideoDecoderStub::queued_bytes() const noexcept {
  return packets_.size();
}

void VideoDecoderStub::setup_buffers() {
  const std::size_t bytes = frame_bytes();
  frames_.assign(bytes * static_cast<std::size_t>(kFrameSlots), 0);
  slot_ = 0;
  frames_decoded_ = 0;
  next_pts_ = 0;
}

bool VideoDecoderStub::open_pattern(std::size_t width, std::size_t height) {
  if (width < 2 || height < 2 || (width & 1u) != 0 || (height & 1u) != 0) {
    return false;
  }
  open_ = true;
  pattern_ = true;
  width_ = width;
  height_ = height;
  setup_buffers();
  return true;
}

bool VideoDecoderStub::open(const char* uri) {
  close();
  if (uri == nullptr || uri[0] == '\0') {
    return false;
  }
  if (std::strncmp(uri, "pattern", 7) == 0) {
    return open_pattern(64, 48);
  }
  return open_pattern(16, 16);
}

void VideoDecoderStub::close() {
  open_ = false;
  pattern_ = false;
  width_ = 0;
  height_ = 0;
  frames_.clear();
}

bool VideoDecoderStub::push_packet(const std::byte* data, std::size_t bytes, std::uint64_t pts) {
  if (!open_ || data == nullptr || bytes == 0) {
    return false;
  }
  std::uint8_t header[sizeof(std::uint64_t) + sizeof(std::uint32_t)];
  std::memcpy(header, &pts, sizeof(pts));
  const auto n32 = static_cast<std::uint32_t>(bytes);
  std::memcpy(header + sizeof(pts), &n32, sizeof(n32));
  const std::size_t need = sizeof(header) + bytes;
  if (packets_.free_space() < need) {
    return false;
  }
  //packet ring slot
  if (packets_.write(reinterpret_cast<const std::byte*>(header), sizeof(header)) != sizeof(header)) {
    return false;
  }
  return packets_.write(data, bytes) == bytes;
}

void VideoDecoderStub::fill_pattern(std::uint8_t* dst) noexcept {
  const std::size_t y_size = width_ * height_;
  const std::size_t uv_size = y_size / 4;
  const auto t = static_cast<std::uint8_t>(frames_decoded_ * 3u);
  for (std::size_t y = 0; y < height_; ++y) {
    for (std::size_t x = 0; x < width_; ++x) {
      dst[y * width_ + x] = static_cast<std::uint8_t>((x + y + t) & 0xffu);
    }
  }
  std::uint8_t* u = dst + y_size;
  std::uint8_t* v = u + uv_size;
  const std::size_t cw = width_ / 2;
  const std::size_t ch = height_ / 2;
  for (std::size_t y = 0; y < ch; ++y) {
    for (std::size_t x = 0; x < cw; ++x) {
      u[y * cw + x] = 128;
      v[y * cw + x] = static_cast<std::uint8_t>(64 + ((x + t) & 63u));
    }
  }
}

bool VideoDecoderStub::decode_next(VideoFrame& out) noexcept {
  if (!open_ || frames_.empty()) {
    out = VideoFrame{};
    return false;
  }
  const std::size_t header = sizeof(std::uint64_t) + sizeof(std::uint32_t);
  if (packets_.size() >= header) {
    std::uint8_t hdr[sizeof(std::uint64_t) + sizeof(std::uint32_t)];
    if (packets_.peek(reinterpret_cast<std::byte*>(hdr), sizeof(hdr)) != sizeof(hdr)) {
      out = VideoFrame{};
      return false;
    }
    std::uint64_t pts = 0;
    std::uint32_t nbytes = 0;
    std::memcpy(&pts, hdr, sizeof(pts));
    std::memcpy(&nbytes, hdr + sizeof(pts), sizeof(nbytes));
    if (packets_.size() < header + static_cast<std::size_t>(nbytes)) {
      out = VideoFrame{};
      return false;
    }
    packets_.read(reinterpret_cast<std::byte*>(hdr), sizeof(hdr));
    const std::size_t need = frame_bytes();
    if (static_cast<std::size_t>(nbytes) < need) {
      drain_bytes(packets_, nbytes);
      out = VideoFrame{};
      return false;
    }
    const std::size_t offset = static_cast<std::size_t>(slot_) * need;
    std::uint8_t* dst = frames_.data() + offset;
    if (packets_.read(reinterpret_cast<std::byte*>(dst), need) != need) {
      out = VideoFrame{};
      return false;
    }
    if (static_cast<std::size_t>(nbytes) > need) {
      drain_bytes(packets_, static_cast<std::size_t>(nbytes) - need);
    }
    const std::size_t y_size = width_ * height_;
    const std::size_t uv_size = y_size / 4;
    out.data = dst;
    out.y = dst;
    out.u = dst + y_size;
    out.v = out.u + uv_size;
    out.width = width_;
    out.height = height_;
    out.stride_y = width_;
    out.stride_uv = width_ / 2;
    out.pts = pts;
    slot_ = (slot_ + 1) % kFrameSlots;
    ++frames_decoded_;
    return true;
  }
  if (!pattern_) {
    out = VideoFrame{};
    return false;
  }
  const std::size_t need = frame_bytes();
  const std::size_t offset = static_cast<std::size_t>(slot_) * need;
  fill_pattern(frames_.data() + offset);
  const std::size_t y_size = width_ * height_;
  const std::size_t uv_size = y_size / 4;
  std::uint8_t* dst = frames_.data() + offset;
  out.data = dst;
  out.y = dst;
  out.u = dst + y_size;
  out.v = out.u + uv_size;
  out.width = width_;
  out.height = height_;
  out.stride_y = width_;
  out.stride_uv = width_ / 2;
  out.pts = next_pts_++;
  slot_ = (slot_ + 1) % kFrameSlots;
  ++frames_decoded_;
  return true;
}

}  // namespace eoc
