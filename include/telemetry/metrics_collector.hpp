#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace eoc {

struct MetricSample {
  char name[32];
  double value;
  std::uint64_t timestamp_us;
};

class EOC_API MetricsCollector {
 public:
  explicit MetricsCollector(std::size_t capacity = 1024);
  ~MetricsCollector() = default;

  bool record(const char* name, double value, std::uint64_t timestamp_us) noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] const MetricSample* data() const noexcept;
  void clear() noexcept;

 private:
  std::vector<MetricSample> slots_;
  std::size_t count_;
};

}  // namespace eoc
