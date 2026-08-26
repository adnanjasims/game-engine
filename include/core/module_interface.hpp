#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace eoc {

enum class ModulePhase : std::uint8_t {
  Unregistered = 0,
  Registered,
  PreInit,
  Init,
  PostInit,
  Ready,
  Failed,
  Shutdown
};

enum class PropertyType : std::uint8_t {
  Bool = 0,
  Int32,
  Int64,
  Float,
  Double,
  String,
  Vec3,
  Raw
};

struct Vec3f {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
};

struct PropertyDesc {
  std::string name;
  PropertyType type = PropertyType::Raw;
  const char* type_name = "raw";
  void* address = nullptr;
  bool editable = true;
};

class EOC_API PropertyRegistry {
 public:
  template <typename T>
  bool bind(const char* name, T& value, bool editable = true);

  bool bind_raw(const char* name, void* address, const char* type_name, bool editable = true);

  [[nodiscard]] const PropertyDesc* find(std::string_view name) const noexcept;
  [[nodiscard]] PropertyDesc* find(std::string_view name) noexcept;
  [[nodiscard]] std::span<const PropertyDesc> list() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;

  template <typename T>
  [[nodiscard]] T* get(std::string_view name) noexcept;

  template <typename T>
  [[nodiscard]] const T* get(std::string_view name) const noexcept;

  bool set_from_string(std::string_view name, std::string_view value);
  bool get_as_string(std::string_view name, std::string& out) const;
  void clear() noexcept;

 private:
  bool add_desc(PropertyDesc desc);
  std::vector<PropertyDesc> bindings_;
};

class EOC_API IModule {
 public:
  virtual ~IModule() = default;
  [[nodiscard]] virtual const char* name() const noexcept = 0;
  virtual bool pre_init() { return true; }
  virtual bool startup() = 0;
  virtual bool post_init() { return true; }
  virtual void shutdown() = 0;
  [[nodiscard]] virtual bool is_ready() const noexcept = 0;
  [[nodiscard]] virtual bool supports_dynamic_reload() const noexcept { return false; }
  [[nodiscard]] virtual PropertyRegistry* properties() noexcept { return nullptr; }
  [[nodiscard]] virtual const PropertyRegistry* properties() const noexcept { return nullptr; }
};

class EOC_API ModuleBase : public IModule {
 public:
  explicit ModuleBase(const char* module_name);
  ~ModuleBase() override;

  [[nodiscard]] const char* name() const noexcept override;
  [[nodiscard]] bool is_ready() const noexcept override;
  [[nodiscard]] PropertyRegistry* properties() noexcept override;
  [[nodiscard]] const PropertyRegistry* properties() const noexcept override;

 protected:
  void set_ready(bool ready) noexcept;
  [[nodiscard]] PropertyRegistry& props() noexcept;
  [[nodiscard]] const PropertyRegistry& props() const noexcept;

 private:
  std::string name_;
  PropertyRegistry properties_;
  bool ready_;
};

class EOC_API ModuleRegistry {
 public:
  ModuleRegistry();
  ~ModuleRegistry();

  ModuleRegistry(const ModuleRegistry&) = delete;
  ModuleRegistry& operator=(const ModuleRegistry&) = delete;

  bool register_module(IModule* module);
  bool register_module(IModule* module, std::span<const char* const> dependencies);
  bool register_module(std::unique_ptr<IModule> module);
  bool register_module(std::unique_ptr<IModule> module, std::span<const char* const> dependencies);
  bool load_shared(const char* path);
  bool unregister(std::string_view name);

  bool startup_all();
  void shutdown_all();

  [[nodiscard]] IModule* find(std::string_view name) const noexcept;
  [[nodiscard]] ModulePhase phase(std::string_view name) const noexcept;
  [[nodiscard]] std::string_view last_error(std::string_view name) const noexcept;
  [[nodiscard]] std::size_t count() const noexcept;
  [[nodiscard]] std::size_t ready_count() const noexcept;

  [[nodiscard]] PropertyRegistry& properties() noexcept { return properties_; }
  [[nodiscard]] const PropertyRegistry& properties() const noexcept { return properties_; }

 private:
  struct Record;
  std::vector<Record> records_;
  PropertyRegistry properties_;

  Record* find_record(std::string_view name) noexcept;
  const Record* find_record(std::string_view name) const noexcept;
  bool insert_record(IModule* module, std::span<const char* const> dependencies, bool owned,
                     void* dylib, void (*destroy_fn)(IModule*));
  bool compute_order(std::vector<std::size_t>& order);
  bool deps_ready(const Record& rec) const noexcept;
};

[[nodiscard]] constexpr const char* property_type_name(PropertyType type) noexcept {
  switch (type) {
    case PropertyType::Bool:
      return "bool";
    case PropertyType::Int32:
      return "int32";
    case PropertyType::Int64:
      return "int64";
    case PropertyType::Float:
      return "float";
    case PropertyType::Double:
      return "double";
    case PropertyType::String:
      return "string";
    case PropertyType::Vec3:
      return "vec3";
    case PropertyType::Raw:
    default:
      return "raw";
  }
}

template <typename T>
constexpr PropertyType property_type_of() noexcept {
  if constexpr (std::is_same_v<T, bool>) {
    return PropertyType::Bool;
  } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, std::int32_t>) {
    return PropertyType::Int32;
  } else if constexpr (std::is_same_v<T, long long> || std::is_same_v<T, std::int64_t>) {
    return PropertyType::Int64;
  } else if constexpr (std::is_same_v<T, float>) {
    return PropertyType::Float;
  } else if constexpr (std::is_same_v<T, double>) {
    return PropertyType::Double;
  } else if constexpr (std::is_same_v<T, std::string>) {
    return PropertyType::String;
  } else if constexpr (std::is_same_v<T, Vec3f>) {
    return PropertyType::Vec3;
  } else {
    return PropertyType::Raw;
  }
}

template <typename T>
bool PropertyRegistry::bind(const char* name, T& value, bool editable) {
  PropertyDesc desc;
  desc.name = name != nullptr ? name : "";
  desc.type = property_type_of<T>();
  desc.type_name = property_type_name(desc.type);
  desc.address = &value;
  desc.editable = editable;
  return add_desc(std::move(desc));
}

template <typename T>
T* PropertyRegistry::get(std::string_view name) noexcept {
  PropertyDesc* desc = find(name);
  if (desc == nullptr || desc->address == nullptr) {
    return nullptr;
  }
  if (desc->type != property_type_of<T>()) {
    return nullptr;
  }
  return static_cast<T*>(desc->address);
}

template <typename T>
const T* PropertyRegistry::get(std::string_view name) const noexcept {
  return const_cast<PropertyRegistry*>(this)->get<T>(name);
}

}  // namespace eoc
