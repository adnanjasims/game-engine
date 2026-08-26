# C++ Game Engine

C++20 engine subsystem: includes a module registry, work-stealing task graph, frame allocators, async inference stub, multimedia ring buffers, and a chrome-tracing profiler.

## Architecture

```
tick
  -> LinearArenaAllocator.reset
  -> TaskGraph.submit
       workers (render | physics | audio | inference | general)
       work stealing + lane affinity
       profiler span start/end
  -> PoolAllocator task nodes
  -> chrome tracing json
```

Namespace: `eoc`. Shared library: `eoc_core`. Benchmark binary: `eoc_bench`.

## Dependencies

- CMake 3.20+
- C++20 compiler (Clang or GCC)
- pthread
- GoogleTest v1.15.2 (fetched by CMake)
- Python 3 (optional, tools/)



## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/eoc_bench
python3 tools/benchmark_runner.py --bin ./build/eoc_bench
python3 tools/trace_exporter.py --input trace.json --output /tmp/eoc_trace.json
```

Docker:

```bash
docker build -t eoc .
```



## Sample benchmark output

```
arena_ns_per_alloc: 2.56
pool_ns_per_alloc: 7.24
taskgraph_independent_us: 76.25
taskgraph_dag_us: 9.00
taskgraph_counter: 512
taskgraph_dag_flag: 12
taskgraph_workers: 8
frame_us: 1429
trace_events: 519
trace_exported: 1
inference_out0: 0.0
```

