#include "ai/inference_engine.hpp"
#include "ai/simd_kernels.hpp"
#include "ai/tensor_buffer.hpp"
#include "core/memory_allocators.hpp"
#include "core/module_interface.hpp"
#include "core/profiler.hpp"
#include "core/task_graph.hpp"
#include "multimedia/audio_pipeline.hpp"
#include "multimedia/video_decoder_stub.hpp"
#include "telemetry/metrics_collector.hpp"
#include "telemetry/network_telemetry.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::uint64_t ns_now() {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now().time_since_epoch())
                                        .count());
}

struct BenchOpts {
  std::size_t workers = 0;
  std::size_t arena_bytes = 1u << 20;
  int frames = 8;
  const char* assets = nullptr;
  const char* trace = "trace.json";
};

struct PackedAssets {
  std::vector<std::int16_t> pcm;
  std::vector<std::uint8_t> i420;
  std::uint32_t sample_rate = 48000;
  std::uint16_t channels = 1;
  std::size_t width = 64;
  std::size_t height = 48;
  std::size_t frame_count = 0;
};

std::size_t parse_size(const char* s) {
  if (s == nullptr || s[0] == '\0') {
    return 0;
  }
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s, &end, 10);
  return static_cast<std::size_t>(v);
}

int parse_i32(const char* s) {
  if (s == nullptr || s[0] == '\0') {
    return 0;
  }
  return static_cast<int>(std::strtol(s, nullptr, 10));
}

void print_usage() {
  std::fprintf(stderr,
               "eoc_bench [--workers N] [--arena-bytes N] [--frames N] [--assets DIR] "
               "[--trace PATH]\n");
}

bool parse_opts(int argc, char** argv, BenchOpts& opts) {
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strcmp(a, "--help") == 0 || std::strcmp(a, "-h") == 0) {
      print_usage();
      return false;
    }
    if (i + 1 >= argc) {
      print_usage();
      return false;
    }
    const char* v = argv[++i];
    if (std::strcmp(a, "--workers") == 0) {
      opts.workers = parse_size(v);
    } else if (std::strcmp(a, "--arena-bytes") == 0) {
      opts.arena_bytes = parse_size(v);
      if (opts.arena_bytes == 0) {
        opts.arena_bytes = 1u << 20;
      }
    } else if (std::strcmp(a, "--frames") == 0) {
      opts.frames = parse_i32(v);
    } else if (std::strcmp(a, "--assets") == 0) {
      opts.assets = v;
    } else if (std::strcmp(a, "--trace") == 0) {
      opts.trace = v;
    } else {
      print_usage();
      return false;
    }
  }
  return true;
}

bool read_all(const std::string& path, std::vector<std::uint8_t>& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff n = in.tellg();
  if (n <= 0) {
    out.clear();
    return true;
  }
  in.seekg(0, std::ios::beg);
  out.resize(static_cast<std::size_t>(n));
  in.read(reinterpret_cast<char*>(out.data()), n);
  return static_cast<bool>(in);
}

bool load_assets(const char* dir, PackedAssets& assets) {
  if (dir == nullptr || dir[0] == '\0') {
    return false;
  }
  const std::string root = std::string(dir) + "/";
  std::ifstream man(root + "manifest.txt");
  if (!man) {
    return false;
  }
  std::string audio_name = "tone.s16le";
  std::string video_name = "clip.i420";
  std::string line;
  while (std::getline(man, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) {
      continue;
    }
    const std::string key = line.substr(0, eq);
    const std::string val = line.substr(eq + 1);
    if (key == "sample_rate") {
      assets.sample_rate = static_cast<std::uint32_t>(parse_size(val.c_str()));
    } else if (key == "channels") {
      assets.channels = static_cast<std::uint16_t>(parse_size(val.c_str()));
    } else if (key == "width") {
      assets.width = parse_size(val.c_str());
    } else if (key == "height") {
      assets.height = parse_size(val.c_str());
    } else if (key == "frames") {
      assets.frame_count = parse_size(val.c_str());
    } else if (key == "audio") {
      audio_name = val;
    } else if (key == "video") {
      video_name = val;
    }
  }

  std::vector<std::uint8_t> pcm_bytes;
  if (!read_all(root + audio_name, pcm_bytes) || (pcm_bytes.size() % 2) != 0) {
    return false;
  }
  assets.pcm.resize(pcm_bytes.size() / 2);
  std::memcpy(assets.pcm.data(), pcm_bytes.data(), pcm_bytes.size());

  if (!read_all(root + video_name, assets.i420)) {
    return false;
  }
  const std::size_t fb = assets.width * assets.height + (assets.width * assets.height) / 2;
  if (fb == 0 || assets.i420.size() < fb) {
    return false;
  }
  if (assets.frame_count == 0) {
    assets.frame_count = assets.i420.size() / fb;
  }
  return true;
}

struct TickState {
  float x;
  float v;
  int audio_n;
  int video_n;
};

int run_ticks(const BenchOpts& opts, eoc::LinearArenaAllocator& arena, eoc::TaskGraph& graph,
              eoc::InferenceEngine& infer, eoc::AudioPipeline& audio, eoc::VideoDecoderStub& video,
              const PackedAssets* packed, std::uint64_t& tick_ns_out) {
  eoc::Profiler& prof = eoc::Profiler::instance();
  eoc::TensorBuffer tensor(8);
  for (std::size_t i = 0; i < tensor.size(); ++i) {
    tensor.data()[i] = static_cast<float>(i) * 0.25f;
  }

  const std::int16_t* pcm = packed != nullptr && !packed->pcm.empty() ? packed->pcm.data() : nullptr;
  const std::size_t pcm_n = packed != nullptr ? packed->pcm.size() : 0;
  const std::uint8_t* i420 = packed != nullptr && !packed->i420.empty() ? packed->i420.data() : nullptr;
  const std::size_t i420_n = packed != nullptr ? packed->i420.size() : 0;
  const std::size_t frame_bytes = video.frame_bytes();

  int ok_frames = 0;
  const std::uint64_t t0 = ns_now();
  for (int f = 0; f < opts.frames; ++f) {
    eoc::ScopedTrace tick("frame_tick", "tick");
    arena.reset();
    auto* st = static_cast<TickState*>(arena.allocate(sizeof(TickState), 16));
    if (st == nullptr) {
      continue;
    }
    st->x = 0.0f;
    st->v = 1.0f;
    st->audio_n = 0;
    st->video_n = 0;

    constexpr int kN = 4;
    auto* x = static_cast<float*>(arena.allocate(sizeof(float) * static_cast<std::size_t>(kN), 64));
    auto* y = static_cast<float*>(arena.allocate(sizeof(float) * static_cast<std::size_t>(kN), 64));
    auto* w = static_cast<float*>(
        arena.allocate(sizeof(float) * static_cast<std::size_t>(kN * kN), 64));
    if (x == nullptr || y == nullptr || w == nullptr) {
      continue;
    }
    for (int i = 0; i < kN; ++i) {
      x[i] = static_cast<float>(i);
      y[i] = 0.0f;
    }
    for (int i = 0; i < kN * kN; ++i) {
      w[i] = (i % (kN + 1) == 0) ? 1.0f : 0.0f;
    }

    graph.reset();
    auto phys = graph.create_task(eoc::TaskLane::Physics, "phys", [st]() {
      st->v += 0.05f;
      st->x += st->v;
    });
    auto aud = graph.create_task(eoc::TaskLane::Audio, "aud", [&audio, pcm, pcm_n, st, f]() {
      std::int16_t tmp[32];
      const int n = 32;
      if (pcm != nullptr && pcm_n > 0) {
        for (int i = 0; i < n; ++i) {
          tmp[i] = pcm[(static_cast<std::size_t>(f) * static_cast<std::size_t>(n) +
                        static_cast<std::size_t>(i)) %
                       pcm_n];
        }
      } else {
        for (int i = 0; i < n; ++i) {
          tmp[i] = static_cast<std::int16_t>((f + i) & 0x7fff);
        }
      }
      st->audio_n = static_cast<int>(audio.write_s16(tmp, static_cast<std::size_t>(n)));
      (void)audio.read_s16(tmp, static_cast<std::size_t>(n));
    });
    auto inf = graph.create_task(eoc::TaskLane::Inference, "inf", [w, x, y]() {
      eoc::simd_gemv(w, kN, kN, x, nullptr, y);
      eoc::simd_relu(y, static_cast<std::size_t>(kN));
    });
    auto rend = graph.create_task(eoc::TaskLane::Render, "rend", [st]() { (void)st->x; });
    graph.add_dependency(phys, rend);
    graph.add_dependency(aud, rend);
    graph.add_dependency(inf, rend);

    auto fut = infer.submit(tensor);
    graph.submit();
    graph.wait();
    const std::vector<float> inf_out = fut.get();

    if (i420 != nullptr && frame_bytes > 0 && i420_n >= frame_bytes) {
      const std::size_t off = (static_cast<std::size_t>(f) * frame_bytes) % (i420_n - frame_bytes + 1);
      (void)video.push_packet(reinterpret_cast<const std::byte*>(i420 + off), frame_bytes,
                              static_cast<std::uint64_t>(f));
    }
    eoc::VideoFrame frame{};
    if (video.decode_next(frame)) {
      st->video_n = 1;
    }

    prof.record_allocator("arena_high_water", arena.high_water_mark());
    if (st->audio_n > 0 && st->video_n > 0 && !inf_out.empty() && st->x != 0.0f) {
      ++ok_frames;
    }
  }
  tick_ns_out = ns_now() - t0;
  return ok_frames;
}

}  // namespace

int main(int argc, char** argv) {
  using eoc::LinearArenaAllocator;
  using eoc::PoolAllocator;
  using eoc::Profiler;
  using eoc::ScopedTrace;
  using eoc::StackAllocator;
  using eoc::TaskGraph;
  using eoc::TaskLane;

  BenchOpts opts;
  if (!parse_opts(argc, argv, opts)) {
    return 2;
  }

  Profiler& prof = Profiler::instance();
  prof.reset();
  prof.begin_frame();

  LinearArenaAllocator arena(opts.arena_bytes);
  constexpr int kArenaIters = 200000;
  const std::uint64_t arena_t0 = ns_now();
  for (int i = 0; i < kArenaIters; ++i) {
    void* p = arena.allocate(32, 16);
    if (p == nullptr) {
      prof.record_cache_sample(false);
      arena.reset();
      p = arena.allocate(32, 16);
    } else {
      prof.record_cache_sample(true);
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
    prof.record_cache_sample(p != nullptr);
    pool.deallocate(p);
  }
  const std::uint64_t pool_ns = ns_now() - pool_t0;

  StackAllocator stack(1u << 16);
  void* sp = stack.allocate(128, 64);
  (void)sp;

  TaskGraph graph(opts.workers, 4096);
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

  PackedAssets packed;
  const bool have_assets = load_assets(opts.assets, packed);
  eoc::AudioFormat afmt;
  if (have_assets) {
    afmt.sample_rate = packed.sample_rate;
    afmt.channels = packed.channels == 0 ? 1 : packed.channels;
  }
  eoc::AudioPipeline audio(4096, afmt);
  std::int16_t tone[256];
  for (int i = 0; i < 256; ++i) {
    tone[i] = static_cast<std::int16_t>(i);
  }
  if (have_assets && packed.pcm.size() >= 256) {
    for (int i = 0; i < 256; ++i) {
      tone[i] = packed.pcm[static_cast<std::size_t>(i)];
    }
  }
  const std::size_t audio_written = audio.write_s16(tone, 256);
  std::int16_t tone_out[256];
  const std::size_t audio_read = audio.read_s16(tone_out, 256);

  eoc::VideoDecoderStub video;
  bool video_ok = false;
  if (have_assets) {
    video_ok = video.open_pattern(packed.width, packed.height);
  } else {
    video_ok = video.open("pattern");
  }
  eoc::VideoFrame frame{};
  int video_frames = 0;
  for (int i = 0; i < 8 && video_ok && video.decode_next(frame); ++i) {
    ++video_frames;
  }
  if (have_assets && video_ok && packed.i420.size() >= video.frame_bytes()) {
    const bool packet_ok =
        video.push_packet(reinterpret_cast<const std::byte*>(packed.i420.data()), video.frame_bytes(),
                          42);
    if (packet_ok && video.decode_next(frame)) {
      ++video_frames;
    }
  } else {
    std::vector<std::byte> pkt(video.frame_bytes(), std::byte{7});
    const bool packet_ok = video_ok && video.push_packet(pkt.data(), pkt.size(), 42);
    if (packet_ok && video.decode_next(frame)) {
      ++video_frames;
    }
  }

  eoc::NetworkTelemetry telemetry(64);
  telemetry.record("bench_complete", 1.0, prof.now_us());
  telemetry.record("video_frames", static_cast<double>(video_frames), prof.now_us());
  std::vector<eoc::MetricSample> flushed(8);
  const std::size_t tel_n =
      telemetry.flush_to(flushed.data(), flushed.size() * sizeof(eoc::MetricSample));

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

  std::uint64_t tick_ns = 0;
  int tick_ok = 0;
  if (opts.frames > 0) {
    tick_ok = run_ticks(opts, arena, graph, infer, audio, video, have_assets ? &packed : nullptr,
                        tick_ns);
  }

  (void)out;
  (void)tone_out;
  (void)audio_read;

  prof.end_frame();
  const bool exported = prof.export_chrome_trace(opts.trace);

  std::printf("arena_ns_per_alloc: %.2f\n",
              static_cast<double>(arena_ns) / static_cast<double>(kArenaIters));
  std::printf("pool_ns_per_alloc: %.2f\n",
              static_cast<double>(pool_ns) / static_cast<double>(kPoolIters));
  std::printf("taskgraph_independent_us: %.2f\n", static_cast<double>(tg_indep_ns) / 1000.0);
  std::printf("taskgraph_dag_us: %.2f\n", static_cast<double>(dag_ns) / 1000.0);
  std::printf("taskgraph_counter: %d\n", counter.load());
  std::printf("taskgraph_dag_flag: %d\n", dag_flag.load());
  std::printf("taskgraph_workers: %zu\n", graph.worker_count());
  std::printf("arena_bytes: %zu\n", opts.arena_bytes);
  std::printf("frame_us: %llu\n", static_cast<unsigned long long>(prof.last_frame_us()));
  std::printf("trace_events: %zu\n", prof.event_count());
  std::printf("trace_exported: %d\n", exported ? 1 : 0);
  std::printf("cache_hits: %llu\n", static_cast<unsigned long long>(prof.cache_hits()));
  std::printf("cache_misses: %llu\n", static_cast<unsigned long long>(prof.cache_misses()));
  std::printf("inference_out0: %.1f\n", out.empty() ? -1.0 : static_cast<double>(out[0]));
  std::printf("inference_backend: %s\n", infer.backend_name());
  std::printf("inference_us: %.2f\n", static_cast<double>(inf_ns) / 1000.0);
  std::printf("onnx_runtime: %d\n", infer.onnx_runtime_available() ? 1 : 0);
  std::printf("modules_ready: %zu\n", modules.ready_count());
  std::printf("module_properties: %zu\n", module_props);
  std::printf("modules_ok: %d\n", modules_ok ? 1 : 0);
  std::printf("audio_samples: %zu\n", audio_written);
  std::printf("video_frames: %d\n", video_frames);
  std::printf("telemetry_flushed: %zu\n", tel_n);
  std::printf("tick_frames: %d\n", opts.frames);
  std::printf("tick_ok: %d\n", tick_ok);
  std::printf("tick_us_avg: %.2f\n",
              opts.frames > 0 ? (static_cast<double>(tick_ns) / 1000.0) / static_cast<double>(opts.frames)
                              : 0.0);
  std::printf("assets_loaded: %d\n", have_assets ? 1 : 0);

  const bool ticks_pass = opts.frames <= 0 || tick_ok == opts.frames;
  const int rc = counter.load() == kIndependent && dag_flag.load() == 12 && modules_ok &&
                         audio_written == 256 && video_frames >= 8 && tel_n == 2 && ticks_pass
                     ? 0
                     : 1;
  std::fflush(stdout);
  modules.shutdown_all();
  return rc;
}
