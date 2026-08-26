#include "core/module_interface.hpp"

namespace eoc {

void PropertyRegistry::bind(const char* name, void* address, const char* type_name) {
  bindings_.push_back(PropertyBinding{name, address, type_name});
}

const PropertyBinding* PropertyRegistry::find(std::string_view name) const noexcept {
  for (const auto& b : bindings_) {
    if (b.name != nullptr && name == b.name) {
      return &b;
    }
  }
  return nullptr;
}

std::size_t PropertyRegistry::size() const noexcept {
  return bindings_.size();
}

void ModuleRegistry::register_module(IModule* module) {
  if (module != nullptr) {
    modules_.push_back(module);
  }
}

bool ModuleRegistry::startup_all() {
  bool ok = true;
  for (IModule* m : modules_) {
    if (m != nullptr && !m->startup()) {
      ok = false;
    }
  }
  return ok;
}

void ModuleRegistry::shutdown_all() {
  for (auto it = modules_.rbegin(); it != modules_.rend(); ++it) {
    if (*it != nullptr) {
      (*it)->shutdown();
    }
  }
}

IModule* ModuleRegistry::find(std::string_view name) const noexcept {
  for (IModule* m : modules_) {
    if (m != nullptr && name == m->name()) {
      return m;
    }
  }
  return nullptr;
}

}  // namespace eoc
