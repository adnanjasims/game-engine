#include "multimedia/audio_pipeline.hpp"

#include <new>

namespace eoc {

AudioPipeline::AudioPipeline(std::size_t capacity_bytes)
    : buffer_(nullptr),
      capacity_(capacity_bytes == 0 ? 4096 : capacity_bytes),
      read_pos_(0),
      write_pos_(0),
      filled_(0) {
  buffer_ = new (std::nothrow) std::byte[capacity_];
  if (buffer_ == nullptr) {
    throw std::bad_alloc();
  }
}

AudioPipeline::~AudioPipeline() {
  delete[] buffer_;
}

std::span<std::byte> AudioPipeline::acquire_write(std::size_t bytes) noexcept {
  if (buffer_ == nullptr || bytes == 0 || filled_ + bytes > capacity_) {
    return {};
  }
  if (write_pos_ + bytes > capacity_) {
    return {};
  }
  return {buffer_ + write_pos_, bytes};
}

void AudioPipeline::commit_write(std::size_t bytes) noexcept {
  if (bytes == 0 || filled_ + bytes > capacity_) {
    return;
  }
  write_pos_ = (write_pos_ + bytes) % capacity_;
  filled_ += bytes;
}

std::span<const std::byte> AudioPipeline::acquire_read(std::size_t bytes) noexcept {
  if (buffer_ == nullptr || bytes == 0 || bytes > filled_) {
    return {};
  }
  if (read_pos_ + bytes > capacity_) {
    return {};
  }
  return {buffer_ + read_pos_, bytes};
}

void AudioPipeline::commit_read(std::size_t bytes) noexcept {
  if (bytes == 0 || bytes > filled_) {
    return;
  }
  read_pos_ = (read_pos_ + bytes) % capacity_;
  filled_ -= bytes;
}

std::size_t AudioPipeline::size() const noexcept {
  return filled_;
}

}  // namespace eoc
