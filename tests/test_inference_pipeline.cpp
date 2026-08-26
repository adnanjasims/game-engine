#include "ai/inference_engine.hpp"
#include "ai/simd_kernels.hpp"
#include "ai/simulated_npu.hpp"
#include "ai/tensor_buffer.hpp"
#include "core/memory_allocators.hpp"
#include "core/module_interface.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using eoc::InferenceBackend;
using eoc::InferenceEngine;
using eoc::InferenceModule;
using eoc::ModuleRegistry;
using eoc::PoolAllocator;
using eoc::SimulatedNpu;
using eoc::TensorBuffer;

TEST(InferencePipeline, SubmitIdentity) {
  InferenceEngine engine;
  TensorBuffer in(4);
  in.data()[0] = 1.0f;
  in.data()[1] = 2.0f;
  in.data()[2] = 3.0f;
  in.data()[3] = 4.0f;
  auto fut = engine.submit(in);
  const std::vector<float> out = fut.get();
  ASSERT_EQ(out.size(), 4u);
  EXPECT_FLOAT_EQ(out[0], 1.0f);
  EXPECT_FLOAT_EQ(out[3], 4.0f);
}

TEST(InferencePipeline, SimdMatmul2x2) {
  const float a[4] = {1, 2, 3, 4};
  const float b[4] = {5, 6, 7, 8};
  float c[4] = {0, 0, 0, 0};
  InferenceEngine::simd_matmul(a, b, c, 2);
  EXPECT_FLOAT_EQ(c[0], 19.0f);
  EXPECT_FLOAT_EQ(c[1], 22.0f);
  EXPECT_FLOAT_EQ(c[2], 43.0f);
  EXPECT_FLOAT_EQ(c[3], 50.0f);
}

TEST(InferencePipeline, SimdGemvRelu) {
  const float w[4] = {1, 2, 3, 4};
  const float x[2] = {1, 1};
  const float b[2] = {-10, 0};
  float y[2] = {0, 0};
  eoc::simd_gemv(w, 2, 2, x, b, y);
  EXPECT_FLOAT_EQ(y[0], -7.0f);
  EXPECT_FLOAT_EQ(y[1], 7.0f);
  eoc::simd_relu(y, 2);
  EXPECT_FLOAT_EQ(y[0], 0.0f);
  EXPECT_FLOAT_EQ(y[1], 7.0f);
}

TEST(InferencePipeline, TensorPoolAligned) {
  PoolAllocator pool(4096, 8, 64);
  TensorBuffer t(pool, 16);
  ASSERT_NE(t.data(), nullptr);
  EXPECT_TRUE(t.pooled());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(t.data()) % 64u, 0u);
  EXPECT_EQ(pool.allocated_count(), 1u);
  t = TensorBuffer();
  EXPECT_EQ(pool.allocated_count(), 0u);
}

TEST(InferencePipeline, SimulatedNpuMatchesSimd) {
  const float w[6] = {1, 0, 0, 1, 0.5f, 0.5f};
  const float x[2] = {2, 4};
  const float bias[3] = {0, 0, 0};
  float simd_y[3] = {0, 0, 0};
  eoc::simd_gemv(w, 3, 2, x, bias, simd_y);
  eoc::simd_relu(simd_y, 3);

  SimulatedNpu npu;
  void* dx = nullptr;
  void* dy = nullptr;
  ASSERT_TRUE(npu.malloc(2 * sizeof(float), &dx));
  ASSERT_TRUE(npu.malloc(3 * sizeof(float), &dy));
  ASSERT_TRUE(npu.memcpy(dx, x, 2 * sizeof(float), eoc::NpuMemcpyKind::HostToDevice));
  ASSERT_TRUE(npu.gemv_relu(w, 3, 2, static_cast<float*>(dx), bias, static_cast<float*>(dy)));
  float npu_y[3] = {0, 0, 0};
  ASSERT_TRUE(npu.memcpy(npu_y, dy, 3 * sizeof(float), eoc::NpuMemcpyKind::DeviceToHost));
  EXPECT_FLOAT_EQ(npu_y[0], simd_y[0]);
  EXPECT_FLOAT_EQ(npu_y[1], simd_y[1]);
  EXPECT_FLOAT_EQ(npu_y[2], simd_y[2]);
  npu.free(dx);
  npu.free(dy);
}

TEST(InferencePipeline, LinearAsyncAndCallback) {
  InferenceEngine engine(InferenceBackend::SimdFallback, 2);
  const float w[4] = {1, 0, 0, 1};
  const float b[2] = {0.5f, -0.5f};
  ASSERT_TRUE(engine.load_linear(w, b, 2, 2, true));
  TensorBuffer in(2);
  in.data()[0] = 1.0f;
  in.data()[1] = 2.0f;
  auto fut = engine.submit(in);
  const std::vector<float> out = fut.get();
  ASSERT_EQ(out.size(), 2u);
  EXPECT_FLOAT_EQ(out[0], 1.5f);
  EXPECT_FLOAT_EQ(out[1], 1.5f);

  std::atomic<int> done{0};
  std::vector<float> cb_out;
  engine.submit(in, [&](std::vector<float> v) {
    cb_out = std::move(v);
    done.store(1);
  });
  for (int i = 0; i < 2000 && done.load() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  EXPECT_EQ(done.load(), 1);
  ASSERT_EQ(cb_out.size(), 2u);
  EXPECT_FLOAT_EQ(cb_out[0], 1.5f);
}

TEST(InferencePipeline, NpuAndSimdBackends) {
  const float w[2] = {2, 3};
  const float b[1] = {1};
  TensorBuffer in(2);
  in.data()[0] = 1.0f;
  in.data()[1] = 1.0f;

  InferenceEngine simd(InferenceBackend::SimdFallback, 1);
  ASSERT_TRUE(simd.load_linear(w, b, 2, 1, false));
  const std::vector<float> a = simd.submit(in).get();

  InferenceEngine npu(InferenceBackend::SimulatedNpu, 1);
  ASSERT_TRUE(npu.load_linear(w, b, 2, 1, false));
  const std::vector<float> c = npu.submit(in).get();
  ASSERT_EQ(a.size(), 1u);
  ASSERT_EQ(c.size(), 1u);
  EXPECT_FLOAT_EQ(a[0], 6.0f);
  EXPECT_FLOAT_EQ(c[0], 6.0f);
  EXPECT_STREQ(npu.backend_name(), "simulated_npu");
}

TEST(InferencePipeline, InferenceModule) {
  InferenceModule mod;
  ModuleRegistry reg;
  ASSERT_TRUE(reg.register_module(&mod));
  ASSERT_TRUE(reg.startup_all());
  TensorBuffer in(3);
  in.fill(2.0f);
  const std::vector<float> out = mod.engine().submit(in).get();
  ASSERT_EQ(out.size(), 3u);
  EXPECT_FLOAT_EQ(out[1], 2.0f);
  reg.shutdown_all();
}
