#include "core/module_interface.hpp"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using eoc::IModule;
using eoc::ModuleBase;
using eoc::ModulePhase;
using eoc::ModuleRegistry;
using eoc::PropertyRegistry;
using eoc::PropertyType;
using eoc::Vec3f;

namespace {

class RecModule final : public ModuleBase {
 public:
  RecModule(const char* module_name, std::vector<std::string>* log)
      : ModuleBase(module_name), log_(log) {}

  bool fail_pre = false;
  bool fail_start = false;
  bool fail_post = false;

  bool pre_init() override {
    push("pre");
    return !fail_pre;
  }

  bool startup() override {
    push("start");
    if (fail_start) {
      return false;
    }
    set_ready(true);
    return true;
  }

  bool post_init() override {
    push("post");
    return !fail_post;
  }

  void shutdown() override {
    push("stop");
    set_ready(false);
  }

 private:
  void push(const char* phase) {
    if (log_ != nullptr) {
      log_->push_back(std::string(name()) + ":" + phase);
    }
  }

  std::vector<std::string>* log_;
};

}  // namespace

TEST(PropertyRegistry, BindGetSet) {
  PropertyRegistry props;
  bool flag = false;
  int count = 0;
  float scale = 1.0f;
  std::string label = "idle";
  Vec3f pos{1.0f, 2.0f, 3.0f};

  EXPECT_TRUE(props.bind("flag", flag));
  EXPECT_TRUE(props.bind("count", count));
  EXPECT_TRUE(props.bind("scale", scale));
  EXPECT_TRUE(props.bind("label", label));
  EXPECT_TRUE(props.bind("pos", pos));
  EXPECT_FALSE(props.bind("flag", flag));
  EXPECT_EQ(props.size(), 5u);

  ASSERT_NE(props.get<bool>("flag"), nullptr);
  EXPECT_EQ(props.find("count")->type, PropertyType::Int32);

  EXPECT_TRUE(props.set_from_string("flag", "true"));
  EXPECT_TRUE(flag);
  EXPECT_TRUE(props.set_from_string("count", "42"));
  EXPECT_EQ(count, 42);
  EXPECT_TRUE(props.set_from_string("scale", "2.5"));
  EXPECT_FLOAT_EQ(scale, 2.5f);
  EXPECT_TRUE(props.set_from_string("label", "run"));
  EXPECT_EQ(label, "run");
  EXPECT_TRUE(props.set_from_string("pos", "4,5,6"));
  EXPECT_FLOAT_EQ(pos.x, 4.0f);
  EXPECT_FLOAT_EQ(pos.y, 5.0f);
  EXPECT_FLOAT_EQ(pos.z, 6.0f);

  std::string text;
  EXPECT_TRUE(props.get_as_string("flag", text));
  EXPECT_EQ(text, "true");
  EXPECT_TRUE(props.get_as_string("label", text));
  EXPECT_EQ(text, "run");
  EXPECT_FALSE(props.set_from_string("missing", "1"));
  EXPECT_FALSE(props.set_from_string("count", "nope"));
}

TEST(ModuleRegistry, PhaseOrderAndShutdownReverse) {
  std::vector<std::string> log;
  RecModule a("a", &log);
  RecModule b("b", &log);
  ModuleRegistry reg;
  ASSERT_TRUE(reg.register_module(&a));
  ASSERT_TRUE(reg.register_module(&b));
  EXPECT_TRUE(reg.startup_all());
  EXPECT_EQ(reg.ready_count(), 2u);
  EXPECT_EQ(reg.phase("a"), ModulePhase::Ready);
  EXPECT_TRUE(a.is_ready());

  const std::vector<std::string> started = {"a:pre", "a:start", "a:post", "b:pre", "b:start",
                                            "b:post"};
  EXPECT_EQ(log, started);

  log.clear();
  reg.shutdown_all();
  EXPECT_EQ(reg.phase("a"), ModulePhase::Shutdown);
  const std::vector<std::string> stopped = {"b:stop", "a:stop"};
  EXPECT_EQ(log, stopped);
}

TEST(ModuleRegistry, DependencyOrder) {
  std::vector<std::string> log;
  RecModule core("core", &log);
  RecModule render("render", &log);
  ModuleRegistry reg;
  const char* const deps[] = {"core"};
  ASSERT_TRUE(reg.register_module(&render, deps));
  ASSERT_TRUE(reg.register_module(&core));
  EXPECT_EQ(reg.count(), 2u);
  EXPECT_TRUE(reg.startup_all());
  ASSERT_GE(log.size(), 6u);
  EXPECT_EQ(log[0], "core:pre");
  EXPECT_EQ(log[3], "render:pre");
}

TEST(ModuleRegistry, FailureIsolation) {
  std::vector<std::string> log;
  RecModule core("core", &log);
  RecModule render("render", &log);
  RecModule audio("audio", &log);
  core.fail_start = true;

  ModuleRegistry reg;
  ASSERT_TRUE(reg.register_module(&core));
  const char* const deps[] = {"core"};
  ASSERT_TRUE(reg.register_module(&render, deps));
  ASSERT_TRUE(reg.register_module(&audio));
  EXPECT_FALSE(reg.startup_all());
  EXPECT_EQ(reg.phase("core"), ModulePhase::Failed);
  EXPECT_EQ(reg.phase("render"), ModulePhase::Failed);
  EXPECT_EQ(reg.phase("audio"), ModulePhase::Ready);
  EXPECT_EQ(reg.last_error("core"), "startup failed");
  EXPECT_EQ(reg.last_error("render"), "dependency not ready");
  EXPECT_TRUE(audio.is_ready());
  EXPECT_FALSE(render.is_ready());
}

TEST(ModuleRegistry, DependencyCycle) {
  RecModule a("a", nullptr);
  RecModule b("b", nullptr);
  ModuleRegistry reg;
  const char* const a_deps[] = {"b"};
  const char* const b_deps[] = {"a"};
  ASSERT_TRUE(reg.register_module(&a, a_deps));
  ASSERT_TRUE(reg.register_module(&b, b_deps));
  EXPECT_FALSE(reg.startup_all());
  EXPECT_EQ(reg.phase("a"), ModulePhase::Failed);
  EXPECT_EQ(reg.phase("b"), ModulePhase::Failed);
  EXPECT_EQ(reg.last_error("a"), "dependency cycle");
}

TEST(ModuleRegistry, DuplicateAndMissing) {
  RecModule a("dup", nullptr);
  RecModule b("dup", nullptr);
  ModuleRegistry reg;
  EXPECT_TRUE(reg.register_module(&a));
  EXPECT_FALSE(reg.register_module(&b));
  EXPECT_EQ(reg.find("missing"), nullptr);
  EXPECT_EQ(reg.phase("missing"), ModulePhase::Unregistered);
}

TEST(ModuleRegistry, OwnedAndUnregister) {
  auto owned = std::make_unique<RecModule>("owned", nullptr);
  ModuleRegistry reg;
  ASSERT_TRUE(reg.register_module(std::unique_ptr<IModule>(std::move(owned))));
  EXPECT_TRUE(reg.startup_all());
  EXPECT_TRUE(reg.unregister("owned"));
  EXPECT_EQ(reg.find("owned"), nullptr);
  EXPECT_EQ(reg.count(), 0u);
}

TEST(ModuleRegistry, LoadSharedPlugin) {
  const char* path = EOC_TEST_PLUGIN_PATH;
  if (path == nullptr || path[0] == '\0') {
    GTEST_SKIP() << "plugin path not set";
  }
  ModuleRegistry reg;
  ASSERT_TRUE(reg.load_shared(path)) << "failed to load " << path;
  ASSERT_NE(reg.find("test_plugin"), nullptr);
  EXPECT_TRUE(reg.startup_all());
  EXPECT_EQ(reg.phase("test_plugin"), ModulePhase::Ready);
  auto* props = reg.find("test_plugin")->properties();
  ASSERT_NE(props, nullptr);
  ASSERT_NE(props->get<bool>("enabled"), nullptr);
  EXPECT_TRUE(*props->get<bool>("enabled"));
}
