#pragma once

#include "core/export.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace eoc {

class EOC_API Profiler {
 public:
  static constexpr std::size_t kCapacity = 8192;

  static Profiler& instance() noexcept;

  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  [[nodiscard]] std::uint64_t now_us() const noexcept;
  [[nodiscard]] std::uint32_t thread_id() noexcept;

  void begin_frame() noexcept;
  void end_frame() noexcept;

  void record_span(const char* name, const char* category, std::uint64_t start_us,
                   std::uint64_t end_us) noexcept;
  void record_allocator(const char* name, std::size_t bytes) noexcept;
  void record_cache_sample(bool hit) noexcept;

  void reset() noexcept;
  [[nodiscard]] bool export_chrome_trace(const char* path) const;

  [[nodiscard]] std::uint64_t last_frame_us() const noexcept;
  [[nodiscard]] std::size_t event_count() const noexcept;
  [[nodiscard]] std::size_t dropped_count() const noexcept;
  [[nodiscard]] std::uint64_t cache_hits() const noexcept;
  [[nodiscard]] std::uint64_t cache_misses() const noexcept;

 private:
  struct Event {
    char name[48];
    char category[16];
    char phase;
    std::uint64_t ts_us;
    std::uint64_t dur_us;
    std::uint32_t tid;
    std::uint64_t counter;
  };

  Profiler();

  Event events_[kCapacity];
  alignas(64) std::atomic<std::size_t> write_;
  alignas(64) std::atomic<std::size_t> dropped_;
  std::atomic<std::uint64_t> cache_hits_;
  std::atomic<std::uint64_t> cache_misses_;
  std::atomic<std::uint64_t> frame_start_us_;
  std::atomic<std::uint64_t> last_frame_us_;
  std::chrono::steady_clock::time_point origin_;
};

class EOC_API ScopedTrace {
 public:
  ScopedTrace(const char* name, const char* category) noexcept;
  ~ScopedTrace();

  ScopedTrace(const ScopedTrace&) = delete;
  ScopedTrace& operator=(const ScopedTrace&) = delete;

 private:
  const char* name_;
  const char* category_;
  std::uint64_t start_us_;
};

}  // namespace eoc
