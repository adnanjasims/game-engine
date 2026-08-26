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

  eoc::InferenceEngine infer(eoc::InferenceBackend::SimulatedNpu, 2);
  eoc::TensorBuffer tensor(8);
  for (std::size_t i = 0; i < tensor.size(); ++i) {
    tensor.data()[i] = static_cast<float>(i);
  }
  const std::uint64_t inf_t0 = ns_now();
  auto fut = infer.submit(tensor);
  const std::vector<float> out = fut.get();
  const std::uint64_t inf_ns = ns_now() - inf_t0;

  eoc::AudioPipeline audio;
  eoc::VideoDecoderStub video;
  eoc::MetricsCollector metrics;

  class BenchCore final : public eoc::ModuleBase {
   public:
    BenchCore() : ModuleBase("bench_core") { props().bind("workers", workers_); }
    bool startup() override {
      set_ready(true);
      return true;
    }
    void shutdown() override { set_ready(false); }
    int workers_ = 0;
  };

  class BenchRender final : public eoc::ModuleBase {
   public:
    BenchRender() : ModuleBase("bench_render") { props().bind("enabled", enabled_); }
    bool startup() override {
      set_ready(true);
      return true;
    }
    void shutdown() override { set_ready(false); }
    bool enabled_ = true;
  };

  BenchCore core_mod;
  BenchRender render_mod;
  core_mod.workers_ = static_cast<int>(graph.worker_count());
  eoc::ModuleRegistry modules;
  const char* const render_deps[] = {"bench_core"};
  const bool core_reg = modules.register_module(&core_mod);
  const bool render_reg = modules.register_module(&render_mod, render_deps);
  const bool modules_ok = core_reg && render_reg && modules.startup_all();
  std::size_t module_props = 0;
  if (auto* p = core_mod.properties()) {
    module_props += p->size();
  }
  if (auto* p = render_mod.properties()) {
    module_props += p->size();
  }

  metrics.record("bench_complete", 1.0, prof.now_us());
  (void)audio;
  (void)video;
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
  std::printf("inference_backend: %s\n", infer.backend_name());
  std::printf("inference_us: %.2f\n", static_cast<double>(inf_ns) / 1000.0);
  std::printf("onnx_runtime: %d\n", infer.onnx_runtime_available() ? 1 : 0);
  std::printf("modules_ready: %zu\n", modules.ready_count());
  std::printf("module_properties: %zu\n", module_props);
  std::printf("modules_ok: %d\n", modules_ok ? 1 : 0);

  const int rc = counter.load() == kIndependent && dag_flag.load() == 12 && modules_ok ? 0 : 1;
  modules.shutdown_all();
  return rc;
}
