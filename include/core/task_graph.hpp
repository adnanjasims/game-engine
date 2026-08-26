#pragma once

#include "core/export.hpp"

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace eoc {

struct TaskNode;

enum class TaskLane : std::uint8_t {
  Render = 0,
  Physics = 1,
  Audio = 2,
  Inference = 3,
  General = 4
};

inline constexpr std::size_t kTaskLaneCount = 5;

class EOC_API TaskHandle {
 public:
  TaskHandle() = default;
  [[nodiscard]] explicit operator bool() const noexcept { return node_ != nullptr; }

 private:
  friend class TaskGraph;
  TaskNode* node_ = nullptr;
};

class EOC_API TaskGraph {
 public:
  explicit TaskGraph(std::size_t worker_count = 0, std::size_t max_tasks = 4096);
  TaskGraph(std::size_t worker_count, std::size_t max_tasks, TaskLane pin_lane);
  ~TaskGraph();

  TaskGraph(const TaskGraph&) = delete;
  TaskGraph& operator=(const TaskGraph&) = delete;

  template <typename Fn>
  TaskHandle create_task(TaskLane lane, const char* name, Fn&& fn);

  bool add_dependency(TaskHandle prerequisite, TaskHandle dependent) noexcept;
  void submit();
  void wait() noexcept;
  void reset();

  [[nodiscard]] std::size_t worker_count() const noexcept;
  [[nodiscard]] std::size_t outstanding() const noexcept;

 private:
  TaskHandle create_task_impl(TaskLane lane, const char* name, void (*invoke)(void*),
                              void (*destroy)(void*), void (*construct)(void*, void*),
                              void* callable, std::size_t callable_size,
                              std::size_t callable_align);

  struct Impl;
  Impl* impl_;
};

template <typename Fn>
TaskHandle TaskGraph::create_task(TaskLane lane, const char* name, Fn&& fn) {
  using Decayed = std::decay_t<Fn>;
  Decayed local(std::forward<Fn>(fn));
  auto invoke = [](void* p) { (*static_cast<Decayed*>(p))(); };
  auto destroy = [](void* p) { static_cast<Decayed*>(p)->~Decayed(); };
  auto construct = [](void* dst, void* src) {
    new (dst) Decayed(std::move(*static_cast<Decayed*>(src)));
  };
  return create_task_impl(lane, name, invoke, destroy, construct, &local, sizeof(Decayed),
                         alignof(Decayed));
}

}  // namespace eoc
