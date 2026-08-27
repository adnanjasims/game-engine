#pragma once

#include "core/export.hpp"
#include "telemetry/metrics_collector.hpp"

#include <cstddef>
#include <cstdint>

namespace eoc {

class EOC_API NetworkTelemetry {
 public:
  explicit NetworkTelemetry(std::size_t capacity = 1024);
  ~NetworkTelemetry();

  NetworkTelemetry(const NetworkTelemetry&) = delete;
  NetworkTelemetry& operator=(const NetworkTelemetry&) = delete;

  bool record(const char* name, double value, std::uint64_t timestamp_us) noexcept;
  bool bind_loopback(std::uint16_t port) noexcept;
  void close_socket() noexcept;

  std::size_t flush_udp() noexcept;
  std::size_t flush_to(void* dst, std::size_t dst_bytes) noexcept;

  [[nodiscard]] MetricsCollector& collector() noexcept { return collector_; }
  [[nodiscard]] const MetricsCollector& collector() const noexcept { return collector_; }
  [[nodiscard]] bool socket_ready() const noexcept { return fd_ >= 0; }

 private:
  MetricsCollector collector_;
  int fd_;
  std::uint16_t port_;
};

}  // namespace eoc
