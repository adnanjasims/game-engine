#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace eoc {

struct MetricSample {
  const char* name;
  double value;
  std::uint64_t timestamp_us;
};

class EOC_API MetricsCollector {
 public:
  void record(const char* name, double value, std::uint64_t timestamp_us);
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] const MetricSample* data() const noexcept;
  void clear() noexcept;

 private:
  std::vector<MetricSample> samples_;
  std::vector<std::string> names_;
};

}  // namespace eoc
