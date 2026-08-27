#include "telemetry/network_telemetry.hpp"

#include <cstring>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace eoc {

NetworkTelemetry::NetworkTelemetry(std::size_t capacity)
    : collector_(capacity), fd_(-1), port_(0) {}

NetworkTelemetry::~NetworkTelemetry() {
  close_socket();
}

bool NetworkTelemetry::record(const char* name, double value, std::uint64_t timestamp_us) noexcept {
  return collector_.record(name, value, timestamp_us);
}

bool NetworkTelemetry::bind_loopback(std::uint16_t port) noexcept {
  close_socket();
#if defined(_WIN32)
  (void)port;
  return false;
#else
  const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    return false;
  }
  fd_ = fd;
  port_ = port;
  return true;
#endif
}

void NetworkTelemetry::close_socket() noexcept {
#if !defined(_WIN32)
  if (fd_ >= 0) {
    ::close(fd_);
  }
#endif
  fd_ = -1;
  port_ = 0;
}

std::size_t NetworkTelemetry::flush_to(void* dst, std::size_t dst_bytes) noexcept {
  if (dst == nullptr || collector_.size() == 0) {
    return 0;
  }
  const std::size_t bytes = collector_.size() * sizeof(MetricSample);
  const std::size_t n = bytes < dst_bytes ? bytes : dst_bytes;
  const std::size_t count = n / sizeof(MetricSample);
  if (count == 0) {
    return 0;
  }
  std::memcpy(dst, collector_.data(), count * sizeof(MetricSample));
  collector_.clear();
  return count;
}

std::size_t NetworkTelemetry::flush_udp() noexcept {
  const std::size_t count = collector_.size();
  if (count == 0) {
    return 0;
  }
#if defined(_WIN32)
  collector_.clear();
  return count;
#else
  if (fd_ < 0) {
    collector_.clear();
    return count;
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  //udp telemetry flush
  const ssize_t sent = ::sendto(fd_, collector_.data(), count * sizeof(MetricSample), 0,
                                reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  collector_.clear();
  if (sent <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(sent) / sizeof(MetricSample);
#endif
}

}  // namespace eoc
