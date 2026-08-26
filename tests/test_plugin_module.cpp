#include "core/module_interface.hpp"

namespace {

class TestPluginModule final : public eoc::ModuleBase {
 public:
  TestPluginModule() : ModuleBase("test_plugin") {
    props().bind("enabled", enabled_);
  }

  bool startup() override {
    set_ready(true);
    return true;
  }

  void shutdown() override { set_ready(false); }

 private:
  bool enabled_ = true;
};

}  // namespace

#if defined(_WIN32)
#define EOC_PLUGIN_API extern "C" __declspec(dllexport)
#else
#define EOC_PLUGIN_API extern "C" __attribute__((visibility("default")))
#endif

EOC_PLUGIN_API eoc::IModule* eoc_module_create() {
  return new TestPluginModule();
}

EOC_PLUGIN_API void eoc_module_destroy(eoc::IModule* module) {
  delete module;
}
