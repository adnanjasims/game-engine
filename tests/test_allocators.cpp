#include "core/memory_allocators.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using eoc::LinearArenaAllocator;
using eoc::PoolAllocator;
using eoc::StackAllocator;
using eoc::align_up;

TEST(AlignUp, PowersOfTwo) {
  EXPECT_EQ(align_up(0, 16), 0u);
  EXPECT_EQ(align_up(1, 16), 16u);
  EXPECT_EQ(align_up(16, 16), 16u);
  EXPECT_EQ(align_up(17, 64), 64u);
}

TEST(LinearArena, AllocAlignAndReset) {
  LinearArenaAllocator arena(4096);
  void* a = arena.allocate(1, 16);
  void* b = arena.allocate(1, 64);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % 16u, 0u);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b) % 64u, 0u);
  EXPECT_GT(arena.used(), 0u);
  const std::size_t hw = arena.high_water_mark();
  arena.reset();
  EXPECT_EQ(arena.used(), 0u);
  EXPECT_EQ(arena.high_water_mark(), hw);
  void* c = arena.allocate(32, 16);
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c, a);
}

TEST(LinearArena, Exhaustion) {
  LinearArenaAllocator arena(128);
  void* p = arena.allocate(256, 16);
  EXPECT_EQ(p, nullptr);
  void* q = arena.allocate(80, 16);
  ASSERT_NE(q, nullptr);
  void* r = arena.allocate(80, 16);
  EXPECT_EQ(r, nullptr);
}

TEST(LinearArena, ConcurrentBump) {
  LinearArenaAllocator arena(1u << 20);
  constexpr int kThreads = 4;
  constexpr int kEach = 10000;
  std::vector<std::thread> threads;
  std::atomic<int> ok{0};
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&arena, &ok]() {
      for (int i = 0; i < kEach; ++i) {
        void* p = arena.allocate(16, 16);
        if (p != nullptr) {
          ok.fetch_add(1);
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(ok.load(), kThreads * kEach);
  EXPECT_GE(arena.used(), static_cast<std::size_t>(kThreads * kEach * 16));
}

TEST(PoolAllocator, AllocFreeReuse) {
  PoolAllocator pool(64, 8, 64);
  void* a = pool.allocate();
  void* b = pool.allocate();
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a) % 64u, 0u);
  EXPECT_EQ(pool.allocated_count(), 2u);
  pool.deallocate(a);
  void* c = pool.allocate();
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c, a);
  pool.deallocate(b);
  pool.deallocate(c);
  EXPECT_EQ(pool.allocated_count(), 0u);
}

TEST(PoolAllocator, Exhaustion) {
  PoolAllocator pool(32, 2, 64);
  EXPECT_NE(pool.allocate(), nullptr);
  EXPECT_NE(pool.allocate(), nullptr);
  EXPECT_EQ(pool.allocate(), nullptr);
}

TEST(StackAllocator, MarkerRewind) {
  StackAllocator stack(1024);
  const auto mark = stack.marker();
  void* a = stack.allocate(64, 16);
  ASSERT_NE(a, nullptr);
  EXPECT_GT(stack.used(), 0u);
  stack.rewind(mark);
  EXPECT_EQ(stack.used(), 0u);
  void* b = stack.allocate(64, 16);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(a, b);
}
