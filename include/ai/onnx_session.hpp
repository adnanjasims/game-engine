#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>

namespace eoc {

class EOC_API OnnxSession {
 public:
  OnnxSession();
  ~OnnxSession();

  OnnxSession(const OnnxSession&) = delete;
  OnnxSession& operator=(const OnnxSession&) = delete;

  [[nodiscard]] static bool runtime_available() noexcept;

  bool load_file(const char* path);
  bool load_linear(const float* weights, const float* bias, int in_features, int out_features,
                   bool relu);
  bool load_identity(int features);
  void clear();

  [[nodiscard]] int in_features() const noexcept;
  [[nodiscard]] int out_features() const noexcept;
  [[nodiscard]] bool ready() const noexcept;
  [[nodiscard]] bool using_ort() const noexcept;

  bool run(const float* input, std::size_t in_count, float* output, std::size_t out_count) const;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace eoc
