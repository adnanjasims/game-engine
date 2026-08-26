#include "core/task_graph.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using eoc::TaskGraph;
using eoc::TaskLane;

TEST(TaskGraph, IndependentTasks) {
  TaskGraph graph(2, 1024);
  std::atomic<int> counter{0};
  constexpr int kCount = 64;
  for (int i = 0; i < kCount; ++i) {
    graph.create_task(TaskLane::General, "inc", [&counter]() { counter.fetch_add(1, std::memory_order_relaxed); });
  }
  graph.submit();
  graph.wait();
  EXPECT_EQ(counter.load(), kCount);
  EXPECT_EQ(graph.outstanding(), 0u);
}

TEST(TaskGraph, DagDependencies) {
  TaskGraph graph(4, 64);
  std::atomic<int> stage{0};
  std::atomic<int> seen_a{0};
  std::atomic<int> seen_b{0};

  auto a = graph.create_task(TaskLane::Physics, "a", [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    seen_a.store(1, std::memory_order_release);
    stage.fetch_add(1, std::memory_order_acq_rel);
  });
  auto b = graph.create_task(TaskLane::Audio, "b", [&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    seen_b.store(1, std::memory_order_release);
    stage.fetch_add(1, std::memory_order_acq_rel);
  });
  auto c = graph.create_task(TaskLane::Render, "c", [&]() {
    EXPECT_EQ(seen_a.load(std::memory_order_acquire), 1);
    EXPECT_EQ(seen_b.load(std::memory_order_acquire), 1);
    stage.fetch_add(10, std::memory_order_acq_rel);
  });
  ASSERT_TRUE(static_cast<bool>(a));
  ASSERT_TRUE(static_cast<bool>(b));
  ASSERT_TRUE(static_cast<bool>(c));
  EXPECT_TRUE(graph.add_dependency(a, c));
  EXPECT_TRUE(graph.add_dependency(b, c));
  graph.submit();
  graph.wait();
  EXPECT_EQ(stage.load(), 12);
}

TEST(TaskGraph, ResetReuse) {
  TaskGraph graph(2, 32);
  std::atomic<int> counter{0};
  graph.create_task(TaskLane::General, "one", [&counter]() { counter.fetch_add(1); });
  graph.submit();
  graph.wait();
  EXPECT_EQ(counter.load(), 1);
  graph.reset();
  graph.create_task(TaskLane::Inference, "two", [&counter]() { counter.fetch_add(1); });
  graph.submit();
  graph.wait();
  EXPECT_EQ(counter.load(), 2);
}

TEST(TaskGraph, Chain) {
  TaskGraph graph(2, 32);
  std::vector<int> order;
  std::mutex mu;
  auto t0 = graph.create_task(TaskLane::General, "t0", [&]() {
    std::lock_guard<std::mutex> lock(mu);
    order.push_back(0);
  });
  auto t1 = graph.create_task(TaskLane::General, "t1", [&]() {
    std::lock_guard<std::mutex> lock(mu);
    order.push_back(1);
  });
  auto t2 = graph.create_task(TaskLane::General, "t2", [&]() {
    std::lock_guard<std::mutex> lock(mu);
    order.push_back(2);
  });
  graph.add_dependency(t0, t1);
  graph.add_dependency(t1, t2);
  graph.submit();
  graph.wait();
  ASSERT_EQ(order.size(), 3u);
  EXPECT_EQ(order[0], 0);
  EXPECT_EQ(order[1], 1);
  EXPECT_EQ(order[2], 2);
}
