#include "core/module_interface.hpp"

#include "core/profiler.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <sstream>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace eoc {
namespace {

void close_dylib(void* handle) noexcept {
  if (handle == nullptr) {
    return;
  }
#if defined(_WIN32)
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

void* open_dylib(const char* path) noexcept {
  if (path == nullptr || path[0] == '\0') {
    return nullptr;
  }
#if defined(_WIN32)
  return static_cast<void*>(LoadLibraryA(path));
#else
  return dlopen(path, RTLD_NOW);
#endif
}

void* dylib_symbol(void* handle, const char* name) noexcept {
  if (handle == nullptr || name == nullptr) {
    return nullptr;
  }
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
  return dlsym(handle, name);
#endif
}

bool iequals(std::string_view a, std::string_view b) noexcept {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    char ca = a[i];
    char cb = b[i];
    if (ca >= 'A' && ca <= 'Z') {
      ca = static_cast<char>(ca - 'A' + 'a');
    }
    if (cb >= 'A' && cb <= 'Z') {
      cb = static_cast<char>(cb - 'A' + 'a');
    }
    if (ca != cb) {
      return false;
    }
  }
  return true;
}

bool parse_bool(std::string_view text, bool& out) noexcept {
  if (iequals(text, "true") || text == "1") {
    out = true;
    return true;
  }
  if (iequals(text, "false") || text == "0") {
    out = false;
    return true;
  }
  return false;
}

template <typename T>
bool parse_int(std::string_view text, T& out) noexcept {
  const char* begin = text.data();
  const char* end = begin + text.size();
  T value{};
  const auto r = std::from_chars(begin, end, value);
  if (r.ec != std::errc() || r.ptr != end) {
    return false;
  }
  out = value;
  return true;
}

bool parse_float(std::string_view text, float& out) noexcept {
  if (text.empty()) {
    return false;
  }
  const std::string tmp(text);
  char* end = nullptr;
  const float v = std::strtof(tmp.c_str(), &end);
  if (end == tmp.c_str() || *end != '\0') {
    return false;
  }
  out = v;
  return true;
}

bool parse_double(std::string_view text, double& out) noexcept {
  if (text.empty()) {
    return false;
  }
  const std::string tmp(text);
  char* end = nullptr;
  const double v = std::strtod(tmp.c_str(), &end);
  if (end == tmp.c_str() || *end != '\0') {
    return false;
  }
  out = v;
  return true;
}

bool parse_vec3(std::string_view text, Vec3f& out) noexcept {
  float vals[3] = {0.0f, 0.0f, 0.0f};
  std::size_t index = 0;
  std::size_t start = 0;
  for (std::size_t i = 0; i <= text.size(); ++i) {
    if (i == text.size() || text[i] == ',') {
      if (index >= 3) {
        return false;
      }
      if (!parse_float(text.substr(start, i - start), vals[index])) {
        return false;
      }
      ++index;
      start = i + 1;
    }
  }
  if (index != 3) {
    return false;
  }
  out.x = vals[0];
  out.y = vals[1];
  out.z = vals[2];
  return true;
}

const char* phase_span_name(ModulePhase phase) noexcept {
  switch (phase) {
    case ModulePhase::PreInit:
      return "module_pre_init";
    case ModulePhase::Init:
      return "module_init";
    case ModulePhase::PostInit:
      return "module_post_init";
    default:
      return "module_phase";
  }
}

}  // namespace

struct ModuleRegistry::Record {
  IModule* module = nullptr;
  std::vector<std::string> deps;
  ModulePhase phase = ModulePhase::Registered;
  std::string error;
  bool owned = false;
  void* dylib = nullptr;
  void (*destroy_fn)(IModule*) = nullptr;

  void destroy_instance() noexcept {
    if (module == nullptr || !owned) {
      module = nullptr;
      return;
    }
    if (destroy_fn != nullptr) {
      destroy_fn(module);
    } else {
      delete module;
    }
    module = nullptr;
  }

  void close_lib() noexcept {
    close_dylib(dylib);
    dylib = nullptr;
  }
};

bool PropertyRegistry::add_desc(PropertyDesc desc) {
  if (desc.name.empty() || desc.address == nullptr) {
    return false;
  }
  if (find(desc.name) != nullptr) {
    return false;
  }
  bindings_.push_back(std::move(desc));
  return true;
}

bool PropertyRegistry::bind_raw(const char* name, void* address, const char* type_name,
                                bool editable) {
  PropertyDesc desc;
  desc.name = name != nullptr ? name : "";
  desc.type = PropertyType::Raw;
  desc.type_name = type_name != nullptr ? type_name : "raw";
  desc.address = address;
  desc.editable = editable;
  return add_desc(std::move(desc));
}

const PropertyDesc* PropertyRegistry::find(std::string_view name) const noexcept {
  for (const auto& b : bindings_) {
    if (b.name == name) {
      return &b;
    }
  }
  return nullptr;
}

PropertyDesc* PropertyRegistry::find(std::string_view name) noexcept {
  return const_cast<PropertyDesc*>(
      static_cast<const PropertyRegistry*>(this)->find(name));
}

std::span<const PropertyDesc> PropertyRegistry::list() const noexcept {
  return std::span<const PropertyDesc>(bindings_.data(), bindings_.size());
}

std::size_t PropertyRegistry::size() const noexcept {
  return bindings_.size();
}

bool PropertyRegistry::set_from_string(std::string_view name, std::string_view value) {
  //property type dispatch
  PropertyDesc* desc = find(name);
  if (desc == nullptr || desc->address == nullptr || !desc->editable) {
    return false;
  }
  switch (desc->type) {
    case PropertyType::Bool: {
      bool v = false;
      if (!parse_bool(value, v)) {
        return false;
      }
      *static_cast<bool*>(desc->address) = v;
      return true;
    }
    case PropertyType::Int32: {
      std::int32_t v = 0;
      if (!parse_int(value, v)) {
        return false;
      }
      *static_cast<std::int32_t*>(desc->address) = v;
      return true;
    }
    case PropertyType::Int64: {
      std::int64_t v = 0;
      if (!parse_int(value, v)) {
        return false;
      }
      *static_cast<std::int64_t*>(desc->address) = v;
      return true;
    }
    case PropertyType::Float: {
      float v = 0.0f;
      if (!parse_float(value, v)) {
        return false;
      }
      *static_cast<float*>(desc->address) = v;
      return true;
    }
    case PropertyType::Double: {
      double v = 0.0;
      if (!parse_double(value, v)) {
        return false;
      }
      *static_cast<double*>(desc->address) = v;
      return true;
    }
    case PropertyType::String:
      *static_cast<std::string*>(desc->address) = std::string(value);
      return true;
    case PropertyType::Vec3: {
      Vec3f v;
      if (!parse_vec3(value, v)) {
        return false;
      }
      *static_cast<Vec3f*>(desc->address) = v;
      return true;
    }
    case PropertyType::Raw:
    default:
      return false;
  }
}

bool PropertyRegistry::get_as_string(std::string_view name, std::string& out) const {
  const PropertyDesc* desc = find(name);
  if (desc == nullptr || desc->address == nullptr) {
    return false;
  }
  std::ostringstream oss;
  switch (desc->type) {
    case PropertyType::Bool:
      oss << (*static_cast<const bool*>(desc->address) ? "true" : "false");
      break;
    case PropertyType::Int32:
      oss << *static_cast<const std::int32_t*>(desc->address);
      break;
    case PropertyType::Int64:
      oss << *static_cast<const std::int64_t*>(desc->address);
      break;
    case PropertyType::Float:
      oss << *static_cast<const float*>(desc->address);
      break;
    case PropertyType::Double:
      oss << *static_cast<const double*>(desc->address);
      break;
    case PropertyType::String:
      out = *static_cast<const std::string*>(desc->address);
      return true;
    case PropertyType::Vec3: {
      const auto* v = static_cast<const Vec3f*>(desc->address);
      oss << v->x << "," << v->y << "," << v->z;
      break;
    }
    case PropertyType::Raw:
    default:
      return false;
  }
  out = oss.str();
  return true;
}

void PropertyRegistry::clear() noexcept {
  bindings_.clear();
}

ModuleBase::ModuleBase(const char* module_name)
    : name_(module_name != nullptr ? module_name : ""), ready_(false) {}

ModuleBase::~ModuleBase() = default;

const char* ModuleBase::name() const noexcept {
  return name_.c_str();
}

bool ModuleBase::is_ready() const noexcept {
  return ready_;
}

PropertyRegistry* ModuleBase::properties() noexcept {
  return &properties_;
}

const PropertyRegistry* ModuleBase::properties() const noexcept {
  return &properties_;
}

void ModuleBase::set_ready(bool ready) noexcept {
  ready_ = ready;
}

PropertyRegistry& ModuleBase::props() noexcept {
  return properties_;
}

const PropertyRegistry& ModuleBase::props() const noexcept {
  return properties_;
}

ModuleRegistry::ModuleRegistry() = default;

ModuleRegistry::~ModuleRegistry() {
  shutdown_all();
  for (auto& rec : records_) {
    rec.destroy_instance();
    rec.close_lib();
  }
  records_.clear();
}

ModuleRegistry::Record* ModuleRegistry::find_record(std::string_view name) noexcept {
  for (auto& rec : records_) {
    if (rec.module != nullptr && rec.module->name() != nullptr && name == rec.module->name()) {
      return &rec;
    }
  }
  return nullptr;
}

const ModuleRegistry::Record* ModuleRegistry::find_record(std::string_view name) const noexcept {
  for (const auto& rec : records_) {
    if (rec.module != nullptr && rec.module->name() != nullptr && name == rec.module->name()) {
      return &rec;
    }
  }
  return nullptr;
}

bool ModuleRegistry::insert_record(IModule* module, std::span<const char* const> dependencies,
                                   bool owned, void* dylib, void (*destroy_fn)(IModule*)) {
  if (module == nullptr || module->name() == nullptr || module->name()[0] == '\0') {
    if (owned && module != nullptr) {
      if (destroy_fn != nullptr) {
        destroy_fn(module);
      } else {
        delete module;
      }
    }
    close_dylib(dylib);
    return false;
  }
  if (find_record(module->name()) != nullptr) {
    if (owned) {
      if (destroy_fn != nullptr) {
        destroy_fn(module);
      } else {
        delete module;
      }
    }
    close_dylib(dylib);
    return false;
  }
  Record rec;
  rec.module = module;
  rec.phase = ModulePhase::Registered;
  rec.owned = owned;
  rec.dylib = dylib;
  rec.destroy_fn = destroy_fn;
  rec.deps.reserve(dependencies.size());
  for (const char* dep : dependencies) {
    if (dep != nullptr && dep[0] != '\0') {
      rec.deps.emplace_back(dep);
    }
  }
  records_.push_back(std::move(rec));
  return true;
}

bool ModuleRegistry::register_module(IModule* module) {
  return insert_record(module, {}, false, nullptr, nullptr);
}

bool ModuleRegistry::register_module(IModule* module, std::span<const char* const> dependencies) {
  return insert_record(module, dependencies, false, nullptr, nullptr);
}

bool ModuleRegistry::register_module(std::unique_ptr<IModule> module) {
  IModule* raw = module.release();
  return insert_record(raw, {}, true, nullptr, nullptr);
}

bool ModuleRegistry::register_module(std::unique_ptr<IModule> module,
                                     std::span<const char* const> dependencies) {
  IModule* raw = module.release();
  return insert_record(raw, dependencies, true, nullptr, nullptr);
}

bool ModuleRegistry::load_shared(const char* path) {
  void* handle = open_dylib(path);
  if (handle == nullptr) {
    return false;
  }
  using CreateFn = IModule* (*)();
  using DestroyFn = void (*)(IModule*);
  auto create = reinterpret_cast<CreateFn>(dylib_symbol(handle, "eoc_module_create"));
  auto destroy = reinterpret_cast<DestroyFn>(dylib_symbol(handle, "eoc_module_destroy"));
  if (create == nullptr) {
    close_dylib(handle);
    return false;
  }
  IModule* module = create();
  if (module == nullptr) {
    close_dylib(handle);
    return false;
  }
  return insert_record(module, {}, true, handle, destroy);
}

bool ModuleRegistry::unregister(std::string_view name) {
  for (auto it = records_.begin(); it != records_.end(); ++it) {
    if (it->module == nullptr || it->module->name() == nullptr || name != it->module->name()) {
      continue;
    }
    if (it->phase == ModulePhase::Ready || it->phase == ModulePhase::Init ||
        it->phase == ModulePhase::PostInit || it->phase == ModulePhase::PreInit) {
      it->module->shutdown();
      it->phase = ModulePhase::Shutdown;
    }
    it->destroy_instance();
    it->close_lib();
    records_.erase(it);
    return true;
  }
  return false;
}

bool ModuleRegistry::compute_order(std::vector<std::size_t>& order) {
  const std::size_t n = records_.size();
  std::vector<int> indeg(n, 0);
  std::vector<std::vector<std::size_t>> adj(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (records_[i].module == nullptr) {
      continue;
    }
    for (const auto& dep_name : records_[i].deps) {
      const Record* dep = find_record(dep_name);
      if (dep == nullptr) {
        records_[i].phase = ModulePhase::Failed;
        records_[i].error = "missing dependency";
        continue;
      }
      const std::size_t dep_index = static_cast<std::size_t>(dep - records_.data());
      adj[dep_index].push_back(i);
      indeg[i] += 1;
    }
  }

  std::vector<std::size_t> ready;
  ready.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (records_[i].module != nullptr && indeg[i] == 0 &&
        records_[i].phase != ModulePhase::Failed) {
      ready.push_back(i);
    }
  }

  order.clear();
  std::size_t cursor = 0;
  while (cursor < ready.size()) {
    const std::size_t i = ready[cursor++];
    order.push_back(i);
    for (std::size_t nxt : adj[i]) {
      indeg[nxt] -= 1;
      if (indeg[nxt] == 0 && records_[nxt].phase != ModulePhase::Failed) {
        ready.push_back(nxt);
      }
    }
  }

  bool acyclic = true;
  for (std::size_t i = 0; i < n; ++i) {
    if (records_[i].module == nullptr || records_[i].phase == ModulePhase::Failed) {
      continue;
    }
    if (std::find(order.begin(), order.end(), i) == order.end()) {
      records_[i].phase = ModulePhase::Failed;
      records_[i].error = "dependency cycle";
      acyclic = false;
    }
  }
  return acyclic;
}

bool ModuleRegistry::deps_ready(const Record& rec) const noexcept {
  for (const auto& dep_name : rec.deps) {
    const Record* dep = find_record(dep_name);
    if (dep == nullptr || dep->phase != ModulePhase::Ready) {
      return false;
    }
  }
  return true;
}

bool ModuleRegistry::startup_all() {
  std::vector<std::size_t> order;
  compute_order(order);
  bool all_ok = true;

  for (std::size_t idx : order) {
    Record& rec = records_[idx];
    if (rec.module == nullptr || rec.phase == ModulePhase::Failed) {
      all_ok = false;
      continue;
    }
    if (rec.phase == ModulePhase::Ready) {
      continue;
    }

    if (!deps_ready(rec)) {
      //skip failed dependency
      rec.phase = ModulePhase::Failed;
      rec.error = "dependency not ready";
      all_ok = false;
      continue;
    }

    const char* mod_name = rec.module->name();
    rec.error.clear();

    rec.phase = ModulePhase::PreInit;
    {
      ScopedTrace trace(phase_span_name(ModulePhase::PreInit), mod_name);
      if (!rec.module->pre_init()) {
        rec.phase = ModulePhase::Failed;
        rec.error = "pre_init failed";
        all_ok = false;
        continue;
      }
    }

    //module init phase
    rec.phase = ModulePhase::Init;
    {
      ScopedTrace trace(phase_span_name(ModulePhase::Init), mod_name);
      if (!rec.module->startup()) {
        rec.module->shutdown();
        rec.phase = ModulePhase::Failed;
        rec.error = "startup failed";
        all_ok = false;
        continue;
      }
    }

    rec.phase = ModulePhase::PostInit;
    {
      ScopedTrace trace(phase_span_name(ModulePhase::PostInit), mod_name);
      if (!rec.module->post_init()) {
        rec.module->shutdown();
        rec.phase = ModulePhase::Failed;
        rec.error = "post_init failed";
        all_ok = false;
        continue;
      }
    }

    rec.phase = ModulePhase::Ready;
  }

  for (const auto& rec : records_) {
    if (rec.module != nullptr && rec.phase != ModulePhase::Ready) {
      all_ok = false;
    }
  }
  return all_ok;
}

void ModuleRegistry::shutdown_all() {
  for (auto it = records_.rbegin(); it != records_.rend(); ++it) {
    if (it->module == nullptr) {
      continue;
    }
    if (it->phase == ModulePhase::Ready || it->phase == ModulePhase::PostInit ||
        it->phase == ModulePhase::Init) {
      it->module->shutdown();
      it->phase = ModulePhase::Shutdown;
    } else if (it->phase == ModulePhase::PreInit) {
      it->phase = ModulePhase::Shutdown;
    }
  }
}

IModule* ModuleRegistry::find(std::string_view name) const noexcept {
  const Record* rec = find_record(name);
  return rec != nullptr ? rec->module : nullptr;
}

ModulePhase ModuleRegistry::phase(std::string_view name) const noexcept {
  const Record* rec = find_record(name);
  return rec != nullptr ? rec->phase : ModulePhase::Unregistered;
}

std::string_view ModuleRegistry::last_error(std::string_view name) const noexcept {
  const Record* rec = find_record(name);
  if (rec == nullptr) {
    return {};
  }
  return rec->error;
}

std::size_t ModuleRegistry::count() const noexcept {
  return records_.size();
}

std::size_t ModuleRegistry::ready_count() const noexcept {
  std::size_t n = 0;
  for (const auto& rec : records_) {
    if (rec.phase == ModulePhase::Ready) {
      ++n;
    }
  }
  return n;
}

}  // namespace eoc
