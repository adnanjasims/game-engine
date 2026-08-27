#include "telemetry/metrics_collector.hpp"

namespace eoc {

MetricsCollector::MetricsCollector(std::size_t capacity) : slots_(capacity == 0 ? 1 : capacity), count_(0) {
  for (auto& s : slots_) {
    s.name[0] = '\0';
    s.value = 0.0;
    s.timestamp_us = 0;
  }
}

bool MetricsCollector::record(const char* name, double value, std::uint64_t timestamp_us) noexcept {
  if (count_ >= slots_.size()) {
    return false;
  }
  MetricSample& s = slots_[count_];
  if (name == nullptr) {
    s.name[0] = '\0';
  } else {
    std::size_t i = 0;
    for (; i + 1 < sizeof(s.name) && name[i] != '\0'; ++i) {
      s.name[i] = name[i];
    }
    s.name[i] = '\0';
  }
  s.value = value;
  s.timestamp_us = timestamp_us;
  ++count_;
  return true;
}

std::size_t MetricsCollector::size() const noexcept {
  return count_;
}

std::size_t MetricsCollector::capacity() const noexcept {
  return slots_.size();
}

const MetricSample* MetricsCollector::data() const noexcept {
  return slots_.empty() ? nullptr : slots_.data();
}

void MetricsCollector::clear() noexcept {
  count_ = 0;
}

}  // namespace eoc
