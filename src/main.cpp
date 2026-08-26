#include "ai/inference_engine.hpp"
#include "ai/tensor_buffer.hpp"
#include "core/memory_allocators.hpp"
#include "core/module_interface.hpp"
#include "core/profiler.hpp"
#include "core/task_graph.hpp"
#include "multimedia/audio_pipeline.hpp"
#include "multimedia/video_decoder_stub.hpp"
#include "telemetry/metrics_collector.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

std::uint64_t ns_now() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

}  // namespace

int main() {
  using eoc::LinearArenaAllocator;
  using eoc::PoolAllocator;
  using eoc::Profiler;
  using eoc::ScopedTrace;
  using eoc::StackAllocator;
  using eoc::TaskGraph;
  using eoc::TaskLane;

  Profiler& prof = Profiler::instance();
  prof.reset();
  prof.begin_frame();

  LinearArenaAllocator arena(1u << 20);
  constexpr int kArenaIters = 200000;
  const std::uint64_t arena_t0 = ns_now();
  for (int i = 0; i < kArenaIters; ++i) {
    void* p = arena.allocate(32, 16);
    if (p == nullptr) {
      arena.reset();
      p = arena.allocate(32, 16);
    }
    (void)p;
  }
  const std::uint64_t arena_ns = ns_now() - arena_t0;
  prof.record_allocator("arena_high_water", arena.high_water_mark());

  PoolAllocator pool(64, 4096, 64);
  constexpr int kPoolIters = 100000;
  const std::uint64_t pool_t0 = ns_now();
  for (int i = 0; i < kPoolIters; ++i) {
    void* p = pool.allocate();
    pool.deallocate(p);
  }
  const std::uint64_t pool_ns = ns_now() - pool_t0;

  StackAllocator stack(1u << 16);
  void* sp = stack.allocate(128, 64);
  (void)sp;

  TaskGraph graph(0, 4096);
  std::atomic<int> counter{0};
  constexpr int kIndependent = 512;
  const std::uint64_t tg_t0 = ns_now();
  {
    ScopedTrace trace("independent_tasks", "bench");
    for (int i = 0; i < kIndependent; ++i) {
      graph.create_task(TaskLane::General, "inc", [&counter]() { counter.fetch_add(1); });
    }
    graph.submit();
    graph.wait();
  }
  const std::uint64_t tg_indep_ns = ns_now() - tg_t0;
  graph.reset();

  std::atomic<int> dag_flag{0};
  const std::uint64_t dag_t0 = ns_now();
  {
    ScopedTrace trace("dag_tasks", "bench");
    auto a = graph.create_task(TaskLane::Physics, "a", [&dag_flag]() { dag_flag.fetch_add(1); });
    auto b = graph.create_task(TaskLane::Audio, "b", [&dag_flag]() { dag_flag.fetch_add(1); });
    auto c = graph.create_task(TaskLane::Render, "c", [&dag_flag]() { dag_flag.fetch_add(10); });
    graph.add_dependency(a, c);
    graph.add_dependency(b, c);
    graph.submit();
    graph.wait();
  }
  const std::uint64_t dag_ns = ns_now() - dag_t0;

  eoc::InferenceEngine infer;
  eoc::TensorBuffer tensor(8);
  for (std::size_t i = 0; i < tensor.size(); ++i) {
    tensor.data()[i] = static_cast<float>(i);
  }
  auto fut = infer.submit(tensor);
  const std::vector<float> out = fut.get();

  eoc::AudioPipeline audio;
  eoc::VideoDecoderStub video;
  eoc::MetricsCollector metrics;
  eoc::ModuleRegistry modules;
  metrics.record("bench_complete", 1.0, prof.now_us());
  (void)audio;
  (void)video;
  (void)modules;
  (void)out;

  prof.end_frame();
  const bool exported = prof.export_chrome_trace("trace.json");

  std::printf("arena_ns_per_alloc: %.2f\n",
              static_cast<double>(arena_ns) / static_cast<double>(kArenaIters));
  std::printf("pool_ns_per_alloc: %.2f\n",
              static_cast<double>(pool_ns) / static_cast<double>(kPoolIters));
  std::printf("taskgraph_independent_us: %.2f\n", static_cast<double>(tg_indep_ns) / 1000.0);
  std::printf("taskgraph_dag_us: %.2f\n", static_cast<double>(dag_ns) / 1000.0);
  std::printf("taskgraph_counter: %d\n", counter.load());
  std::printf("taskgraph_dag_flag: %d\n", dag_flag.load());
  std::printf("taskgraph_workers: %zu\n", graph.worker_count());
  std::printf("frame_us: %llu\n", static_cast<unsigned long long>(prof.last_frame_us()));
  std::printf("trace_events: %zu\n", prof.event_count());
  std::printf("trace_exported: %d\n", exported ? 1 : 0);
  std::printf("inference_out0: %.1f\n", out.empty() ? -1.0 : static_cast<double>(out[0]));

  return counter.load() == kIndependent && dag_flag.load() == 12 ? 0 : 1;
}
