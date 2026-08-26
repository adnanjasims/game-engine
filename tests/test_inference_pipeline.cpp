#include "ai/inference_engine.hpp"
#include "ai/tensor_buffer.hpp"

#include <vector>

#include <gtest/gtest.h>

using eoc::InferenceEngine;
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
