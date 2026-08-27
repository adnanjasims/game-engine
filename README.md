# C++ Game Engine

C++20 engine subsystem: includes a module registry, work-stealing task graph, frame allocators, async inference (simulated npu / simd / optional onnx runtime), multimedia ring buffers, and a chrome-tracing profiler.

## Architecture

```
tick
  -> ModuleRegistry (pre_init, init, post_init)
  -> LinearArenaAllocator.reset
  -> TaskGraph.submit
       workers (render | physics | audio | inference | general)
       work stealing + lane affinity
       profiler span start/end
  -> PoolAllocator task nodes / tensors
  -> InferenceEngine (onnx | simulated npu | simd) on inference lane
  -> AudioPipeline / VideoDecoder zero-copy rings
  -> NetworkTelemetry flush
  -> chrome tracing json (spans, allocator high-water, cache hit/miss)
```

Namespace: `eoc`. Shared library: `eoc_core`. Benchmark binary: `eoc_bench`.

Module lifecycle: `pre_init` -> `startup` -> `post_init`, shutdown reverse, dependency order, failed modules are isolated. Properties bind `bool`, `int32`, `int64`, `float`, `double`, `string`, `vec3` for editor get/set. Shared plugins export `eoc_module_create` / `eoc_module_destroy`.

Inference: `InferenceEngine` submits GEMV/ReLU (or identity) onto the task-graph `Inference` lane. Backends are simulated NPU (host-visible device malloc + dma), SIMD (AVX2/NEON), and ONNX Runtime when `find_package`/`ONNXRUNTIME_ROOT` succeeds; tensors allocate from a 64-byte pool.

Multimedia: SPSC wrap-around `ZeroCopyRing` for audio samples and video packets. `VideoDecoderStub` decodes I420 packets into a 3-slot frame pool or generates a pattern. `NetworkTelemetry` records into a preallocated `MetricsCollector` and flushes samples without extra heap growth.

Tick: `eoc_bench` runs a small frame loop (render + physics + audio + inference) that resets the arena each frame. Chrome traces include process/thread metadata plus `cache_hits` / `cache_misses` counters (work-stealing local pop vs steal).

## Dependencies

- CMake 3.20+
- C++20 compiler (Clang or GCC)
- pthread
- GoogleTest v1.15.2 (fetched by CMake)
- ONNX Runtime (optional, `ONNXRUNTIME_ROOT` or system paths)
- Python 3 (optional, tools/)



## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/eoc_bench
python3 tools/asset_pack.py --out assets
./build/eoc_bench --assets assets --frames 8
python3 tools/benchmark_runner.py --bin ./build/eoc_bench --sweep-workers 2,4 --sweep-arena 64k,256k,1M
python3 tools/trace_exporter.py --input trace.json --output /tmp/eoc_trace.json
```

Docker:

```bash
docker build -t eoc .
```



## Sample benchmark output

```
arena_ns_per_alloc: 6.89
pool_ns_per_alloc: 17.47
taskgraph_independent_us: 295.96
taskgraph_dag_us: 36.08
taskgraph_counter: 512
taskgraph_dag_flag: 12
taskgraph_workers: 8
arena_bytes: 1048576
frame_us: 4658
trace_events: 591
trace_exported: 1
cache_hits: 300049
cache_misses: 507
inference_out0: 0.0
inference_backend: simulated_npu
inference_us: 25.21
onnx_runtime: 0
modules_ready: 2
module_properties: 2
modules_ok: 1
audio_samples: 256
video_frames: 9
telemetry_flushed: 2
tick_frames: 8
tick_ok: 8
tick_us_avg: 43.78
assets_loaded: 1
```

