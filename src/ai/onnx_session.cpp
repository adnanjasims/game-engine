#include "ai/onnx_session.hpp"

#include "ai/simd_kernels.hpp"

#include <cstring>
#include <memory>
#include <vector>

#if defined(EOC_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#endif

namespace eoc {

struct OnnxSession::Impl {
  enum class Kind { None, Identity, Linear, Ort };

  Kind kind = Kind::None;
  int in_features = 0;
  int out_features = 0;
  bool relu = false;
  std::vector<float> weights;
  std::vector<float> bias;
#if defined(EOC_HAS_ONNXRUNTIME)
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "eoc"};
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
#endif
};

OnnxSession::OnnxSession() : impl_(new Impl()) {}

OnnxSession::~OnnxSession() {
  delete impl_;
}

bool OnnxSession::runtime_available() noexcept {
#if defined(EOC_HAS_ONNXRUNTIME)
  return true;
#else
  return false;
#endif
}

bool OnnxSession::load_file(const char* path) {
  clear();
  if (path == nullptr || path[0] == '\0') {
    return false;
  }
#if defined(EOC_HAS_ONNXRUNTIME)
  try {
    impl_->options.SetIntraOpNumThreads(1);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, path, impl_->options);
    impl_->kind = Impl::Kind::Ort;
    return true;
  } catch (...) {
    impl_->session.reset();
    impl_->kind = Impl::Kind::None;
    return false;
  }
#else
  return false;
#endif
}

bool OnnxSession::load_linear(const float* weights, const float* bias, int in_features,
                              int out_features, bool relu) {
  clear();
  if (weights == nullptr || in_features <= 0 || out_features <= 0) {
    return false;
  }
  const std::size_t wcount =
      static_cast<std::size_t>(in_features) * static_cast<std::size_t>(out_features);
  impl_->weights.assign(weights, weights + wcount);
  if (bias != nullptr) {
    impl_->bias.assign(bias, bias + out_features);
  } else {
    impl_->bias.assign(static_cast<std::size_t>(out_features), 0.0f);
  }
  impl_->in_features = in_features;
  impl_->out_features = out_features;
  impl_->relu = relu;
  impl_->kind = Impl::Kind::Linear;
  return true;
}

bool OnnxSession::load_identity(int features) {
  clear();
  if (features < 0) {
    return false;
  }
  impl_->in_features = features;
  impl_->out_features = features;
  impl_->kind = Impl::Kind::Identity;
  return true;
}

void OnnxSession::clear() {
  impl_->kind = Impl::Kind::None;
  impl_->in_features = 0;
  impl_->out_features = 0;
  impl_->relu = false;
  impl_->weights.clear();
  impl_->bias.clear();
#if defined(EOC_HAS_ONNXRUNTIME)
  impl_->session.reset();
#endif
}

int OnnxSession::in_features() const noexcept {
  return impl_->in_features;
}

int OnnxSession::out_features() const noexcept {
  return impl_->out_features;
}

bool OnnxSession::ready() const noexcept {
  return impl_->kind != Impl::Kind::None;
}

bool OnnxSession::using_ort() const noexcept {
  return impl_->kind == Impl::Kind::Ort;
}

bool OnnxSession::run(const float* input, std::size_t in_count, float* output,
                      std::size_t out_count) const {
  if (input == nullptr || output == nullptr || impl_->kind == Impl::Kind::None) {
    return false;
  }
  if (impl_->kind == Impl::Kind::Identity) {
    if (in_count != out_count) {
      return false;
    }
    if (impl_->in_features > 0 && static_cast<int>(in_count) != impl_->in_features) {
      return false;
    }
    std::memcpy(output, input, in_count * sizeof(float));
    return true;
  }
  if (impl_->kind == Impl::Kind::Linear) {
    if (static_cast<int>(in_count) != impl_->in_features ||
        static_cast<int>(out_count) != impl_->out_features) {
      return false;
    }
    simd_gemv(impl_->weights.data(), impl_->out_features, impl_->in_features, input,
              impl_->bias.data(), output);
    if (impl_->relu) {
      simd_relu(output, out_count);
    }
    return true;
  }
#if defined(EOC_HAS_ONNXRUNTIME)
  if (impl_->kind == Impl::Kind::Ort && impl_->session) {
    (void)in_count;
    (void)out_count;
    return false;
  }
#endif
  return false;
}

}  // namespace eoc
