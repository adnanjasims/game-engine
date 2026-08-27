#include "core/profiler.hpp"

#include <fstream>
#include <ostream>
#include <string_view>

namespace eoc {
namespace {

void copy_cstr(char* dst, std::size_t dst_size, const char* src) noexcept {
  if (dst_size == 0) {
    return;
  }
  if (src == nullptr) {
    dst[0] = '\0';
    return;
  }
  std::size_t i = 0;
  for (; i + 1 < dst_size && src[i] != '\0'; ++i) {
    dst[i] = src[i];
  }
  dst[i] = '\0';
}

void json_escape(std::ostream& out, std::string_view s) {
  for (char c : s) {
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      default:
        out << c;
        break;
    }
  }
}

void emit_comma(std::ostream& out, bool& first) {
  if (!first) {
    out << ",\n";
  }
  first = false;
}

}  // namespace

Profiler& Profiler::instance() noexcept {
  static Profiler profiler;
  return profiler;
}

Profiler::Profiler()
    : events_{},
      write_(0),
      dropped_(0),
      cache_hits_(0),
      cache_misses_(0),
      frame_start_us_(0),
      last_frame_us_(0),
      origin_(std::chrono::steady_clock::now()) {}

std::uint64_t Profiler::now_us() const noexcept {
  const auto now = std::chrono::steady_clock::now();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(now - origin_).count());
}

std::uint32_t Profiler::thread_id() noexcept {
  static std::atomic<std::uint32_t> next{1};
  thread_local std::uint32_t id = next.fetch_add(1, std::memory_order_relaxed);
  return id;
}

void Profiler::begin_frame() noexcept {
  frame_start_us_.store(now_us(), std::memory_order_relaxed);
}

void Profiler::end_frame() noexcept {
  const std::uint64_t start = frame_start_us_.load(std::memory_order_relaxed);
  const std::uint64_t end = now_us();
  const std::uint64_t dur = end >= start ? end - start : 0;
  last_frame_us_.store(dur, std::memory_order_relaxed);
  record_span("frame", "cpu", start, end);
}

void Profiler::record_span(const char* name, const char* category, std::uint64_t start_us,
                           std::uint64_t end_us) noexcept {
  const std::size_t slot = write_.fetch_add(1, std::memory_order_relaxed);
  if (slot >= kCapacity) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  Event& e = events_[slot];
  copy_cstr(e.name, sizeof(e.name), name);
  copy_cstr(e.category, sizeof(e.category), category);
  e.phase = 'X';
  e.ts_us = start_us;
  e.dur_us = end_us >= start_us ? end_us - start_us : 0;
  e.tid = thread_id();
  e.counter = 0;
  //record span timestamps
}

void Profiler::record_allocator(const char* name, std::size_t bytes) noexcept {
  const std::size_t slot = write_.fetch_add(1, std::memory_order_relaxed);
  if (slot >= kCapacity) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return;
  }
  Event& e = events_[slot];
  copy_cstr(e.name, sizeof(e.name), name);
  copy_cstr(e.category, sizeof(e.category), "memory");
  e.phase = 'C';
  e.ts_us = now_us();
  e.dur_us = 0;
  e.tid = thread_id();
  e.counter = static_cast<std::uint64_t>(bytes);
}

void Profiler::record_cache_sample(bool hit) noexcept {
  if (hit) {
    cache_hits_.fetch_add(1, std::memory_order_relaxed);
  } else {
    cache_misses_.fetch_add(1, std::memory_order_relaxed);
  }
}

void Profiler::reset() noexcept {
  write_.store(0, std::memory_order_relaxed);
  dropped_.store(0, std::memory_order_relaxed);
  cache_hits_.store(0, std::memory_order_relaxed);
  cache_misses_.store(0, std::memory_order_relaxed);
  last_frame_us_.store(0, std::memory_order_relaxed);
}

bool Profiler::export_chrome_trace(const char* path) const {
  if (path == nullptr) {
    return false;
  }
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  std::size_t count = write_.load(std::memory_order_acquire);
  if (count > kCapacity) {
    count = kCapacity;
  }

  std::uint32_t tids[64];
  std::size_t ntid = 0;
  auto remember_tid = [&](std::uint32_t tid) {
    for (std::size_t i = 0; i < ntid; ++i) {
      if (tids[i] == tid) {
        return;
      }
    }
    if (ntid < 64) {
      tids[ntid++] = tid;
    }
  };
  for (std::size_t i = 0; i < count; ++i) {
    remember_tid(events_[i].tid);
  }

  const std::uint64_t end_ts = count == 0 ? now_us() : events_[count - 1].ts_us;
  const std::uint64_t hits = cache_hits_.load(std::memory_order_acquire);
  const std::uint64_t misses = cache_misses_.load(std::memory_order_acquire);
  const std::size_t dropped = dropped_.load(std::memory_order_acquire);

  out << "{\n  \"displayTimeUnit\": \"us\",\n  \"traceEvents\": [\n";
  bool first = true;

  emit_comma(out, first);
  out << "    {\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{\"name\":\"eoc\"}}";
  emit_comma(out, first);
  out << "    {\"name\":\"process_labels\",\"ph\":\"M\",\"pid\":1,\"args\":{\"labels\":\"engineoptimizationcore\"}}";

  for (std::size_t i = 0; i < ntid; ++i) {
    emit_comma(out, first);
    out << "    {\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":" << tids[i]
        << ",\"args\":{\"name\":\"worker-" << tids[i] << "\"}}";
  }

  for (std::size_t i = 0; i < count; ++i) {
    const Event& e = events_[i];
    emit_comma(out, first);
    out << "    {\"name\":\"";
    json_escape(out, e.name);
    out << "\",\"cat\":\"";
    json_escape(out, e.category);
    out << "\",\"ph\":\"" << e.phase << "\",\"ts\":" << e.ts_us << ",\"pid\":1,\"tid\":" << e.tid;
    if (e.phase == 'X') {
      out << ",\"dur\":" << e.dur_us;
    } else {
      out << ",\"args\":{\"bytes\":" << e.counter << ",\"value\":" << e.counter << "}";
    }
    out << "}";
  }

  //cache hit/miss and drop counters for chrome://tracing
  emit_comma(out, first);
  out << "    {\"name\":\"cache_hits\",\"cat\":\"cache\",\"ph\":\"C\",\"ts\":" << end_ts
      << ",\"pid\":1,\"tid\":1,\"args\":{\"value\":" << hits << "}}";
  emit_comma(out, first);
  out << "    {\"name\":\"cache_misses\",\"cat\":\"cache\",\"ph\":\"C\",\"ts\":" << end_ts
      << ",\"pid\":1,\"tid\":1,\"args\":{\"value\":" << misses << "}}";
  emit_comma(out, first);
  out << "    {\"name\":\"trace_dropped\",\"cat\":\"profiler\",\"ph\":\"C\",\"ts\":" << end_ts
      << ",\"pid\":1,\"tid\":1,\"args\":{\"value\":" << dropped << "}}";

  out << "\n  ]\n}\n";
  return static_cast<bool>(out);
}

std::uint64_t Profiler::last_frame_us() const noexcept {
  return last_frame_us_.load(std::memory_order_acquire);
}

std::size_t Profiler::event_count() const noexcept {
  const std::size_t w = write_.load(std::memory_order_acquire);
  return w < kCapacity ? w : kCapacity;
}

std::size_t Profiler::dropped_count() const noexcept {
  return dropped_.load(std::memory_order_acquire);
}

std::uint64_t Profiler::cache_hits() const noexcept {
  return cache_hits_.load(std::memory_order_acquire);
}

std::uint64_t Profiler::cache_misses() const noexcept {
  return cache_misses_.load(std::memory_order_acquire);
}

ScopedTrace::ScopedTrace(const char* name, const char* category) noexcept
    : name_(name), category_(category), start_us_(Profiler::instance().now_us()) {}

ScopedTrace::~ScopedTrace() {
  Profiler::instance().record_span(name_, category_, start_us_, Profiler::instance().now_us());
}

}  // namespace eoc
