#include "ai/simulated_npu.hpp"

#include "ai/simd_kernels.hpp"
#include "core/profiler.hpp"

#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace eoc {
namespace {

void* host_aligned_alloc(std::size_t bytes) {
  const std::size_t align = 64;
  const std::size_t size = (bytes + (align - 1u)) & ~(align - 1u);
#if defined(_WIN32)
  return _aligned_malloc(size == 0 ? align : size, align);
#else
  return std::aligned_alloc(align, size == 0 ? align : size);
#endif
}

void host_aligned_free(void* p) noexcept {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

}  // namespace

struct SimulatedNpu::Impl {
  std::mutex mu;
  std::unordered_map<void*, std::size_t> blocks;
  std::size_t total = 0;
};

SimulatedNpu::SimulatedNpu() : impl_(new Impl()), last_op_us_(0) {}

SimulatedNpu::~SimulatedNpu() {
  if (impl_ != nullptr) {
    for (auto& kv : impl_->blocks) {
      host_aligned_free(kv.first);
    }
    delete impl_;
  }
}

bool SimulatedNpu::malloc(std::size_t bytes, void** out) noexcept {
  if (out == nullptr || impl_ == nullptr) {
    return false;
  }
  void* p = host_aligned_alloc(bytes == 0 ? 64 : bytes);
  if (p == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->blocks[p] = bytes;
  impl_->total += bytes;
  *out = p;
  return true;
}

void SimulatedNpu::free(void* ptr) noexcept {
  if (ptr == nullptr || impl_ == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->mu);
  auto it = impl_->blocks.find(ptr);
  if (it == impl_->blocks.end()) {
    return;
  }
  if (impl_->total >= it->second) {
    impl_->total -= it->second;
  }
  impl_->blocks.erase(it);
  host_aligned_free(ptr);
}

bool SimulatedNpu::memcpy(void* dst, const void* src, std::size_t bytes, NpuMemcpyKind kind) noexcept {
  if (dst == nullptr || src == nullptr) {
    return false;
  }
  (void)kind;
  //device dma copy
  const std::uint64_t t0 = Profiler::instance().now_us();
  std::memcpy(dst, src, bytes);
  last_op_us_ = Profiler::instance().now_us() - t0;
  return true;
}

bool SimulatedNpu::gemv_relu(const float* weights, int rows, int cols, const float* x,
                             const float* bias, float* y) noexcept {
  if (weights == nullptr || x == nullptr || y == nullptr) {
    return false;
  }
  //npu tensor binding
  const std::uint64_t t0 = Profiler::instance().now_us();
  simd_gemv(weights, rows, cols, x, bias, y);
  simd_relu(y, static_cast<std::size_t>(rows));
  last_op_us_ = Profiler::instance().now_us() - t0;
  return true;
}

bool SimulatedNpu::matmul(const float* a, const float* b, float* c, int n) noexcept {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return false;
  }
  const std::uint64_t t0 = Profiler::instance().now_us();
  simd_matmul(a, b, c, n);
  last_op_us_ = Profiler::instance().now_us() - t0;
  return true;
}

void SimulatedNpu::synchronize() noexcept {}

std::size_t SimulatedNpu::device_bytes() const noexcept {
  return impl_ != nullptr ? impl_->total : 0;
}

}  // namespace eoc
