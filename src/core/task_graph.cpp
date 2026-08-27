#include "core/task_graph.hpp"

#include "core/memory_allocators.hpp"
#include "core/profiler.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(__APPLE__)
#include <mach/mach_init.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace eoc {
namespace {

constexpr std::size_t kCallableBytes = 96;
constexpr std::size_t kMaxSuccessors = 16;
constexpr std::size_t kDequeCap = 1024;
constexpr std::size_t kQueueCap = 4096;

void cpu_pause() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

void bind_affinity(std::size_t worker_index) noexcept {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  const int ncpu = static_cast<int>(std::thread::hardware_concurrency());
  const int cpu = ncpu > 0 ? static_cast<int>(worker_index % static_cast<std::size_t>(ncpu)) : 0;
  CPU_SET(cpu, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);
#elif defined(__APPLE__)
  (void)worker_index;
#else
  (void)worker_index;
#endif
}

const char* lane_name(TaskLane lane) noexcept {
  switch (lane) {
    case TaskLane::Render:
      return "render";
    case TaskLane::Physics:
      return "physics";
    case TaskLane::Audio:
      return "audio";
    case TaskLane::Inference:
      return "inference";
    case TaskLane::General:
    default:
      return "general";
  }
}

}  // namespace

struct TaskNode {
  alignas(16) std::byte callable[kCallableBytes];
  std::atomic<void (*)(void*)> invoke{nullptr};
  void (*destroy)(void*);
  std::atomic<std::uint32_t> pending;
  std::uint16_t successor_count;
  TaskLane lane;
  char name[32];
  std::array<TaskNode*, kMaxSuccessors> successors;
};

class ChaseLevDeque {
 public:
  ChaseLevDeque() {
    for (std::size_t i = 0; i < kDequeCap; ++i) {
      buffer_[i].store(nullptr, std::memory_order_relaxed);
    }
  }

  bool push(TaskNode* node) noexcept {
    const std::int64_t b = bottom_.load(std::memory_order_relaxed);
    const std::int64_t t = top_.load(std::memory_order_acquire);
    if (b - t >= static_cast<std::int64_t>(kDequeCap)) {
      return false;
    }
    buffer_[static_cast<std::size_t>(b) & (kDequeCap - 1)].store(node, std::memory_order_relaxed);
    bottom_.store(b + 1, std::memory_order_release);
    return true;
  }

  TaskNode* pop() noexcept {
    std::int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
    bottom_.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    std::int64_t t = top_.load(std::memory_order_relaxed);
    if (t > b) {
      bottom_.store(t, std::memory_order_relaxed);
      return nullptr;
    }
    TaskNode* node =
        buffer_[static_cast<std::size_t>(b) & (kDequeCap - 1)].load(std::memory_order_relaxed);
    if (t != b) {
      return node;
    }
    if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
      node = nullptr;
    }
    bottom_.store(t + 1, std::memory_order_relaxed);
    return node;
  }

  TaskNode* steal() noexcept {
    std::int64_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const std::int64_t b = bottom_.load(std::memory_order_acquire);
    if (t >= b) {
      return nullptr;
    }
    TaskNode* node =
        buffer_[static_cast<std::size_t>(t) & (kDequeCap - 1)].load(std::memory_order_relaxed);
    if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
      return nullptr;
    }
    return node;
  }

 private:
  std::atomic<TaskNode*> buffer_[kDequeCap];
  alignas(64) std::atomic<std::int64_t> top_{0};
  alignas(64) std::atomic<std::int64_t> bottom_{0};
};

class MpmcQueue {
 public:
  MpmcQueue() {
    for (std::size_t i = 0; i < kQueueCap; ++i) {
      slots_[i].seq.store(i, std::memory_order_relaxed);
      slots_[i].item = nullptr;
    }
  }

  bool enqueue(TaskNode* node) noexcept {
    std::uint64_t pos = tail_.load(std::memory_order_relaxed);
    for (;;) {
      Slot& slot = slots_[pos & (kQueueCap - 1)];
      const std::intptr_t dif =
          static_cast<std::intptr_t>(slot.seq.load(std::memory_order_acquire)) - static_cast<std::intptr_t>(pos);
      if (dif == 0) {
        if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
          slot.item = node;
          slot.seq.store(pos + 1, std::memory_order_release);
          return true;
        }
      } else if (dif < 0) {
        return false;
      } else {
        pos = tail_.load(std::memory_order_relaxed);
      }
    }
  }

  TaskNode* dequeue() noexcept {
    std::uint64_t pos = head_.load(std::memory_order_relaxed);
    for (;;) {
      Slot& slot = slots_[pos & (kQueueCap - 1)];
      const std::intptr_t dif = static_cast<std::intptr_t>(slot.seq.load(std::memory_order_acquire)) -
                                static_cast<std::intptr_t>(pos + 1);
      if (dif == 0) {
        if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
          TaskNode* node = slot.item;
          slot.seq.store(pos + kQueueCap, std::memory_order_release);
          return node;
        }
      } else if (dif < 0) {
        return nullptr;
      } else {
        pos = head_.load(std::memory_order_relaxed);
      }
    }
  }

 private:
  struct Slot {
    std::atomic<std::uint64_t> seq;
    TaskNode* item;
  };

  alignas(64) Slot slots_[kQueueCap];
  alignas(64) std::atomic<std::uint64_t> head_{0};
  alignas(64) std::atomic<std::uint64_t> tail_{0};
};

struct alignas(64) Worker {
  ChaseLevDeque deque;
  TaskLane preferred{TaskLane::General};
  std::thread thread;
};

struct TaskGraph::Impl {
  explicit Impl(std::size_t worker_count, std::size_t max_tasks, const TaskLane* pin_lane)
      : node_pool_(sizeof(TaskNode), max_tasks, 64),
        tasks_(max_tasks, nullptr),
        max_tasks_(max_tasks) {
    if (worker_count == 0) {
      worker_count = std::thread::hardware_concurrency();
    }
    if (worker_count == 0) {
      worker_count = 1;
    }
    workers_.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) {
      auto worker = std::make_unique<Worker>();
      if (pin_lane != nullptr) {
        worker->preferred = *pin_lane;
      } else {
        worker->preferred = static_cast<TaskLane>(i % kTaskLaneCount);
      }
      workers_.push_back(std::move(worker));
    }
    for (std::size_t i = 0; i < worker_count; ++i) {
      workers_[i]->thread = std::thread([this, i]() { worker_loop(i); });
    }
  }

  ~Impl() {
    stop_.store(true, std::memory_order_release);
    wake_all();
    for (auto& w : workers_) {
      if (w->thread.joinable()) {
        w->thread.join();
      }
    }
    workers_.clear();
    reset_nodes();
  }

  void wake_all() noexcept {
    signal_.fetch_add(1, std::memory_order_release);
    signal_.notify_all();
  }

  std::size_t pick_worker(TaskLane lane) noexcept {
    const std::size_t n = workers_.size();
    const std::size_t lane_i = static_cast<std::size_t>(lane);
    const std::size_t start = lane_rr_[lane_i].fetch_add(1, std::memory_order_relaxed);
    for (std::size_t k = 0; k < n; ++k) {
      const std::size_t idx = (start + k) % n;
      if (workers_[idx]->preferred == lane) {
        return idx;
      }
    }
    return start % n;
  }

  void dispatch(TaskNode* node) noexcept {
    //dispatch task to worker pool
    const std::size_t idx = pick_worker(node->lane);
    if (workers_[idx]->deque.push(node)) {
      wake_all();
      return;
    }
    if (!inbox_.enqueue(node)) {
      execute(node);
      return;
    }
    wake_all();
  }

  TaskNode* take_task(std::size_t worker_index) noexcept {
    TaskNode* node = workers_[worker_index]->deque.pop();
    if (node != nullptr) {
      Profiler::instance().record_cache_sample(true);
      return node;
    }
    node = inbox_.dequeue();
    if (node != nullptr) {
      Profiler::instance().record_cache_sample(true);
      return node;
    }
    const std::size_t n = workers_.size();
    for (std::size_t k = 1; k < n; ++k) {
      const std::size_t victim = (worker_index + k) % n;
      node = workers_[victim]->deque.steal();
      if (node != nullptr) {
        Profiler::instance().record_cache_sample(false);
        return node;
      }
    }
    return nullptr;
  }

  TaskNode* help() noexcept {
    TaskNode* node = inbox_.dequeue();
    if (node != nullptr) {
      Profiler::instance().record_cache_sample(false);
      return node;
    }
    const std::size_t n = workers_.size();
    for (std::size_t i = 0; i < n; ++i) {
      node = workers_[i]->deque.steal();
      if (node != nullptr) {
        Profiler::instance().record_cache_sample(false);
        return node;
      }
    }
    return nullptr;
  }

  void execute(TaskNode* node) noexcept {
    auto fn = node->invoke.exchange(nullptr, std::memory_order_acq_rel);
    if (fn == nullptr) {
      return;
    }
    Profiler& prof = Profiler::instance();
    const std::uint64_t start = prof.now_us();
    fn(node->callable);
    const std::uint64_t end = prof.now_us();
    prof.record_span(node->name[0] != '\0' ? node->name : "task", lane_name(node->lane), start, end);

    for (std::uint16_t i = 0; i < node->successor_count; ++i) {
      TaskNode* succ = node->successors[i];
      if (succ != nullptr && succ->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        dispatch(succ);
      }
    }
    recycle(node);
    outstanding_.fetch_sub(1, std::memory_order_release);
  }

  void recycle(TaskNode* node) {
    if (node->destroy != nullptr) {
      node->destroy(node->callable);
    }
    node->invoke.store(nullptr, std::memory_order_relaxed);
    node->destroy = nullptr;
    node->successor_count = 0;
    node->successors.fill(nullptr);
    node->pending.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(free_mu_);
    free_nodes_.push_back(node);
  }

  void worker_loop(std::size_t index) {
    bind_affinity(index);
    unsigned idle = 0;
    while (!stop_.load(std::memory_order_acquire)) {
      TaskNode* node = take_task(index);
      if (node != nullptr) {
        idle = 0;
        execute(node);
        continue;
      }
      ++idle;
      if (idle < 64) {
        cpu_pause();
        continue;
      }
      idle = 0;
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }

  void reset_nodes() {
    std::lock_guard<std::mutex> lock(free_mu_);
    free_nodes_.clear();
    for (std::size_t i = 0; i < task_count_; ++i) {
      TaskNode* node = tasks_[i];
      if (node == nullptr) {
        continue;
      }
      if (node->destroy != nullptr) {
        node->destroy(node->callable);
      }
      node->~TaskNode();
      node_pool_.deallocate(node);
      tasks_[i] = nullptr;
    }
    task_count_ = 0;
  }

  PoolAllocator node_pool_;
  std::vector<TaskNode*> tasks_;
  std::vector<std::unique_ptr<Worker>> workers_;
  MpmcQueue inbox_;
  std::array<std::atomic<std::size_t>, kTaskLaneCount> lane_rr_{};
  std::atomic<std::uint32_t> signal_{0};
  std::atomic<bool> stop_{false};
  std::atomic<std::uint32_t> outstanding_{0};
  std::atomic<bool> submitted_{false};
  std::mutex free_mu_;
  std::vector<TaskNode*> free_nodes_;
  std::size_t max_tasks_;
  std::size_t task_count_{0};
};

TaskGraph::TaskGraph(std::size_t worker_count, std::size_t max_tasks)
    : impl_(new Impl(worker_count, max_tasks, nullptr)) {}

TaskGraph::TaskGraph(std::size_t worker_count, std::size_t max_tasks, TaskLane pin_lane)
    : impl_(new Impl(worker_count, max_tasks, &pin_lane)) {}

TaskGraph::~TaskGraph() {
  delete impl_;
}

TaskHandle TaskGraph::create_task_impl(TaskLane lane, const char* name, void (*invoke)(void*),
                                       void (*destroy)(void*), void (*construct)(void*, void*),
                                       void* callable, std::size_t callable_size,
                                       std::size_t callable_align) {
  TaskHandle handle;
  if (impl_ == nullptr || construct == nullptr || callable_size > kCallableBytes ||
      callable_align > 16) {
    return handle;
  }

  TaskNode* node = nullptr;
  {
    std::lock_guard<std::mutex> lock(impl_->free_mu_);
    if (!impl_->free_nodes_.empty()) {
      node = impl_->free_nodes_.back();
      impl_->free_nodes_.pop_back();
    }
  }

  if (node == nullptr) {
    if (impl_->task_count_ >= impl_->max_tasks_) {
      return handle;
    }
    void* mem = impl_->node_pool_.allocate();
    if (mem == nullptr) {
      return handle;
    }
    node = new (mem) TaskNode();
    impl_->tasks_[impl_->task_count_++] = node;
  }

  node->invoke.store(invoke, std::memory_order_release);
  node->destroy = destroy;
  const bool live = impl_->submitted_.load(std::memory_order_acquire);
  node->pending.store(live ? 0u : 1u, std::memory_order_relaxed);
  node->successor_count = 0;
  node->lane = lane;
  node->name[0] = '\0';
  if (name != nullptr) {
    std::size_t i = 0;
    for (; i + 1 < sizeof(node->name) && name[i] != '\0'; ++i) {
      node->name[i] = name[i];
    }
    node->name[i] = '\0';
  }
  node->successors.fill(nullptr);
  construct(node->callable, callable);
  impl_->outstanding_.fetch_add(1, std::memory_order_relaxed);
  handle.node_ = node;
  if (live) {
    impl_->dispatch(node);
  }
  return handle;
}

bool TaskGraph::add_dependency(TaskHandle prerequisite, TaskHandle dependent) noexcept {
  if (!prerequisite || !dependent || prerequisite.node_ == dependent.node_) {
    return false;
  }
  TaskNode* pred = prerequisite.node_;
  TaskNode* dep = dependent.node_;
  if (pred->successor_count >= kMaxSuccessors) {
    return false;
  }
  pred->successors[pred->successor_count++] = dep;
  dep->pending.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void TaskGraph::submit() {
  if (impl_->submitted_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  for (std::size_t i = 0; i < impl_->task_count_; ++i) {
    TaskNode* node = impl_->tasks_[i];
    if (node != nullptr && node->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      impl_->dispatch(node);
    }
  }
}

void TaskGraph::wait() noexcept {
  while (impl_->outstanding_.load(std::memory_order_acquire) != 0) {
    TaskNode* node = impl_->help();
    if (node != nullptr) {
      impl_->execute(node);
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  }
}

void TaskGraph::reset() {
  wait();
  impl_->reset_nodes();
  impl_->submitted_.store(false, std::memory_order_relaxed);
}

std::size_t TaskGraph::worker_count() const noexcept {
  return impl_->workers_.size();
}

std::size_t TaskGraph::outstanding() const noexcept {
  return impl_->outstanding_.load(std::memory_order_acquire);
}

}  // namespace eoc
