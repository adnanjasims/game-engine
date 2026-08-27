#include "telemetry/metrics_collector.hpp"
#include "telemetry/network_telemetry.hpp"

#include <vector>

#include <gtest/gtest.h>

using eoc::MetricSample;
using eoc::MetricsCollector;
using eoc::NetworkTelemetry;

TEST(MetricsCollector, PreallocatedNoGrowth) {
  MetricsCollector c(4);
  EXPECT_EQ(c.capacity(), 4u);
  EXPECT_TRUE(c.record("a", 1.0, 10));
  EXPECT_TRUE(c.record("b", 2.0, 20));
  EXPECT_EQ(c.size(), 2u);
  EXPECT_STREQ(c.data()[0].name, "a");
  EXPECT_DOUBLE_EQ(c.data()[1].value, 2.0);
  EXPECT_TRUE(c.record("c", 3.0, 30));
  EXPECT_TRUE(c.record("d", 4.0, 40));
  EXPECT_FALSE(c.record("e", 5.0, 50));
  EXPECT_EQ(c.size(), 4u);
  c.clear();
  EXPECT_EQ(c.size(), 0u);
  EXPECT_TRUE(c.record("f", 6.0, 60));
}

TEST(NetworkTelemetry, FlushToBuffer) {
  NetworkTelemetry tel(8);
  EXPECT_TRUE(tel.record("frame_us", 1200.0, 1));
  EXPECT_TRUE(tel.record("tasks", 12.0, 2));
  std::vector<MetricSample> out(8);
  const std::size_t n = tel.flush_to(out.data(), out.size() * sizeof(MetricSample));
  EXPECT_EQ(n, 2u);
  EXPECT_STREQ(out[0].name, "frame_us");
  EXPECT_DOUBLE_EQ(out[1].value, 12.0);
  EXPECT_EQ(tel.collector().size(), 0u);
}

TEST(NetworkTelemetry, FlushUdpWithoutBindClears) {
  NetworkTelemetry tel(4);
  EXPECT_TRUE(tel.record("x", 1.0, 1));
  const std::size_t n = tel.flush_udp();
  EXPECT_EQ(n, 1u);
  EXPECT_EQ(tel.collector().size(), 0u);
}
