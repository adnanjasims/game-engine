#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace eoc {

class EOC_API IModule {
 public:
  virtual ~IModule() = default;
  [[nodiscard]] virtual const char* name() const noexcept = 0;
  virtual bool startup() = 0;
  virtual void shutdown() = 0;
  [[nodiscard]] virtual bool is_ready() const noexcept = 0;
};

struct PropertyBinding {
  const char* name;
  void* address;
  const char* type_name;
};

class EOC_API PropertyRegistry {
 public:
  void bind(const char* name, void* address, const char* type_name);
  [[nodiscard]] const PropertyBinding* find(std::string_view name) const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;

 private:
  std::vector<PropertyBinding> bindings_;
};

class EOC_API ModuleRegistry {
 public:
  void register_module(IModule* module);
  bool startup_all();
  void shutdown_all();
  [[nodiscard]] IModule* find(std::string_view name) const noexcept;
  [[nodiscard]] PropertyRegistry& properties() noexcept { return properties_; }
  [[nodiscard]] const PropertyRegistry& properties() const noexcept { return properties_; }

 private:
  std::vector<IModule*> modules_;
  PropertyRegistry properties_;
};

}  // namespace eoc
