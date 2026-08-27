#include "multimedia/audio_pipeline.hpp"

namespace eoc {

AudioPipeline::AudioPipeline(std::size_t capacity_bytes, AudioFormat format)
    : format_(format), ring_(capacity_bytes) {}

RingView AudioPipeline::acquire_write(std::size_t bytes) noexcept {
  return ring_.acquire_write(bytes);
}

void AudioPipeline::commit_write(std::size_t bytes) noexcept {
  ring_.commit_write(bytes);
}

RingView AudioPipeline::acquire_read(std::size_t bytes) noexcept {
  return ring_.acquire_read(bytes);
}

void AudioPipeline::commit_read(std::size_t bytes) noexcept {
  ring_.commit_read(bytes);
}

std::size_t AudioPipeline::write_bytes(const std::byte* src, std::size_t bytes) noexcept {
  return ring_.write(src, bytes);
}

std::size_t AudioPipeline::read_bytes(std::byte* dst, std::size_t bytes) noexcept {
  return ring_.read(dst, bytes);
}

std::size_t AudioPipeline::write_s16(const std::int16_t* samples, std::size_t count) noexcept {
  if (samples == nullptr || count == 0) {
    return 0;
  }
  return ring_.write(reinterpret_cast<const std::byte*>(samples), count * sizeof(std::int16_t)) /
         sizeof(std::int16_t);
}

std::size_t AudioPipeline::read_s16(std::int16_t* samples, std::size_t count) noexcept {
  if (samples == nullptr || count == 0) {
    return 0;
  }
  return ring_.read(reinterpret_cast<std::byte*>(samples), count * sizeof(std::int16_t)) /
         sizeof(std::int16_t);
}

std::size_t AudioPipeline::capacity() const noexcept {
  return ring_.capacity();
}

std::size_t AudioPipeline::size() const noexcept {
  return ring_.size();
}

std::size_t AudioPipeline::frames() const noexcept {
  const std::size_t frame_bytes =
      static_cast<std::size_t>(format_.channels) * static_cast<std::size_t>(format_.bytes_per_sample);
  if (frame_bytes == 0) {
    return 0;
  }
  return ring_.size() / frame_bytes;
}

}  // namespace eoc
