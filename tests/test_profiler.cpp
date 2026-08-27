#include "core/profiler.hpp"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

using eoc::Profiler;
using eoc::ScopedTrace;

TEST(Profiler, CacheAndAllocatorCounters) {
  Profiler& p = Profiler::instance();
  p.reset();
  p.record_cache_sample(true);
  p.record_cache_sample(true);
  p.record_cache_sample(false);
  p.record_allocator("arena_high_water", 4096);
  EXPECT_EQ(p.cache_hits(), 2u);
  EXPECT_EQ(p.cache_misses(), 1u);
  EXPECT_GE(p.event_count(), 1u);
}

TEST(Profiler, ExportChromeMetadataAndCounters) {
  Profiler& p = Profiler::instance();
  p.reset();
  {
    ScopedTrace t("export_span", "test");
  }
  p.record_cache_sample(true);
  p.record_cache_sample(false);
  p.record_allocator("arena_high_water", 2048);

  const std::string path = std::string(testing::TempDir()) + "eoc_trace_test.json";
  ASSERT_TRUE(p.export_chrome_trace(path.c_str()));

  std::ifstream in(path);
  ASSERT_TRUE(static_cast<bool>(in));
  std::string body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_NE(body.find("\"displayTimeUnit\": \"us\""), std::string::npos);
  EXPECT_NE(body.find("\"name\":\"process_name\""), std::string::npos);
  EXPECT_NE(body.find("\"name\":\"cache_hits\""), std::string::npos);
  EXPECT_NE(body.find("\"name\":\"cache_misses\""), std::string::npos);
  EXPECT_NE(body.find("arena_high_water"), std::string::npos);
  EXPECT_NE(body.find("\"value\":"), std::string::npos);
  std::remove(path.c_str());
}
