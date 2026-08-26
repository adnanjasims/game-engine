#include "telemetry/metrics_collector.hpp"

namespace eoc {

void MetricsCollector::record(const char* name, double value, std::uint64_t timestamp_us) {
  names_.emplace_back(name != nullptr ? name : "");
  samples_.push_back(MetricSample{names_.back().c_str(), value, timestamp_us});
}

std::size_t MetricsCollector::size() const noexcept {
  return samples_.size();
}

const MetricSample* MetricsCollector::data() const noexcept {
  return samples_.empty() ? nullptr : samples_.data();
}

void MetricsCollector::clear() noexcept {
  samples_.clear();
  names_.clear();
}

}  // namespace eoc
