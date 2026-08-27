#include "ai/inference_engine.hpp"

#include "ai/onnx_session.hpp"
#include "ai/simd_kernels.hpp"
#include "ai/simulated_npu.hpp"
#include "core/memory_allocators.hpp"
#include "core/profiler.hpp"
#include "core/task_graph.hpp"

#include <new>
#include <utility>

namespace eoc {

namespace {

const char* name_of(InferenceBackend backend) noexcept {
  switch (backend) {
    case InferenceBackend::OnnxRuntime:
      return "onnxruntime";
    case InferenceBackend::SimulatedNpu:
      return "simulated_npu";
    case InferenceBackend::SimdFallback:
      return "simd";
    case InferenceBackend::Auto:
    default:
      return "auto";
  }
}

InferenceBackend resolve_backend(InferenceBackend requested) noexcept {
  if (requested == InferenceBackend::OnnxRuntime && !OnnxSession::runtime_available()) {
    return InferenceBackend::SimulatedNpu;
  }
  if (requested == InferenceBackend::Auto) {
    if (OnnxSession::runtime_available()) {
      return InferenceBackend::OnnxRuntime;
    }
    return InferenceBackend::SimulatedNpu;
  }
  return requested;
}

}  // namespace

struct InferJob {
  TensorBuffer input;
  std::promise<std::vector<float>> promise;
  std::function<void(std::vector<float>)> callback;
  bool use_callback = false;
};

struct InferenceEngine::Impl {
  static constexpr std::size_t kMaxTensorFloats = 4096;

  explicit Impl(InferenceBackend backend, std::size_t worker_count)
      : backend(resolve_backend(backend)),
        graph(worker_count == 0 ? 2 : worker_count, 1024, TaskLane::Inference),
        tensor_pool(kMaxTensorFloats * sizeof(float), 64, 64),
        job_pool(sizeof(InferJob), 256, 64) {
    session.load_identity(0);
  }

  InferenceBackend backend;
  TaskGraph graph;
  PoolAllocator tensor_pool;
  PoolAllocator job_pool;
  OnnxSession session;
  SimulatedNpu npu;
  bool accelerator = false;

  TensorBuffer make_tensor(std::size_t n) { return TensorBuffer(tensor_pool, n); }

  std::vector<float> execute(const TensorBuffer& input) {
    ScopedTrace trace("infer_execute", "inference");
    const std::size_t in_n = input.size();
    const std::size_t out_n = session.in_features() == 0 && session.out_features() == 0
                                  ? in_n
                                  : static_cast<std::size_t>(session.out_features());
    std::vector<float> out(out_n, 0.0f);
    if (input.data() == nullptr || in_n == 0) {
      return out;
    }

    if (backend == InferenceBackend::SimulatedNpu) {
      void* dx = nullptr;
      void* dy = nullptr;
      const std::size_t in_bytes = in_n * sizeof(float);
      const std::size_t out_bytes = out_n * sizeof(float);
      if (npu.malloc(in_bytes, &dx) && npu.malloc(out_bytes, &dy)) {
        npu.memcpy(dx, input.data(), in_bytes, NpuMemcpyKind::HostToDevice);
        session.run(static_cast<const float*>(dx), in_n, static_cast<float*>(dy), out_n);
        npu.memcpy(out.data(), dy, out_bytes, NpuMemcpyKind::DeviceToHost);
        npu.synchronize();
        accelerator = true;
      } else {
        session.run(input.data(), in_n, out.data(), out_n);
      }
      npu.free(dx);
      npu.free(dy);
      return out;
    }

    session.run(input.data(), in_n, out.data(), out_n);
    return out;
  }

  InferJob* alloc_job() {
    void* mem = job_pool.allocate();
    if (mem == nullptr) {
      return new InferJob();
    }
    return new (mem) InferJob();
  }

  void free_job(InferJob* job) {
    if (job == nullptr) {
      return;
    }
    if (job_pool.owns(job)) {
      job->~InferJob();
      job_pool.deallocate(job);
    } else {
      delete job;
    }
  }

  std::future<std::vector<float>> enqueue(const TensorBuffer& input,
                                          std::function<void(std::vector<float>)> cb) {
    InferJob* job = alloc_job();
    job->input = make_tensor(input.size());
    if (job->input.data() != nullptr && input.data() != nullptr) {
      job->input.copy_from(input.data(), input.size());
    }
    job->use_callback = static_cast<bool>(cb);
    job->callback = std::move(cb);
    auto fut = job->promise.get_future();

    //dispatch task to inference lane
    graph.create_task(TaskLane::Inference, "infer", [this, job]() {
      std::vector<float> result = execute(job->input);
      if (job->use_callback && job->callback) {
        job->callback(result);
        job->promise.set_value(std::vector<float>{});
      } else {
        job->promise.set_value(std::move(result));
      }
      free_job(job);
    });
    graph.submit();
    return fut;
  }
};

InferenceEngine::InferenceEngine(InferenceBackend backend, std::size_t worker_count)
    : impl_(new Impl(backend, worker_count)) {
  impl_->accelerator = impl_->backend == InferenceBackend::SimulatedNpu ||
                       impl_->backend == InferenceBackend::OnnxRuntime;
}

InferenceEngine::~InferenceEngine() {
  if (impl_ != nullptr) {
    delete impl_;
  }
}

InferenceBackend InferenceEngine::backend() const noexcept {
  return impl_->backend;
}

const char* InferenceEngine::backend_name() const noexcept {
  return name_of(impl_->backend);
}

bool InferenceEngine::accelerator_available() const noexcept {
  return impl_->accelerator;
}

bool InferenceEngine::onnx_runtime_available() const noexcept {
  return OnnxSession::runtime_available();
}

bool InferenceEngine::load_identity(int features) {
  return impl_->session.load_identity(features);
}

bool InferenceEngine::load_linear(const float* weights, const float* bias, int in_features,
                                  int out_features, bool relu) {
  return impl_->session.load_linear(weights, bias, in_features, out_features, relu);
}

bool InferenceEngine::load_onnx(const char* path) {
  if (!impl_->session.load_file(path)) {
    return false;
  }
  impl_->backend = InferenceBackend::OnnxRuntime;
  impl_->accelerator = true;
  return true;
}

std::future<std::vector<float>> InferenceEngine::submit(const TensorBuffer& input) {
  return impl_->enqueue(input, nullptr);
}

void InferenceEngine::submit(const TensorBuffer& input, std::function<void(std::vector<float>)> callback) {
  (void)impl_->enqueue(input, std::move(callback));
}

void InferenceEngine::simd_matmul(const float* a, const float* b, float* c, int n) noexcept {
  eoc::simd_matmul(a, b, c, n);
}

InferenceModule::InferenceModule() : ModuleBase("inference"), engine_() {}

bool InferenceModule::startup() {
  engine_.load_identity(0);
  set_ready(true);
  return true;
}

void InferenceModule::shutdown() {
  set_ready(false);
}

InferenceEngine& InferenceModule::engine() noexcept {
  return engine_;
}

const InferenceEngine& InferenceModule::engine() const noexcept {
  return engine_;
}

}  // namespace eoc
