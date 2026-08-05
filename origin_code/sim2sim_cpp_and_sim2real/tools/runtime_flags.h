#ifndef RUNTIME_FLAGS_H
#define RUNTIME_FLAGS_H
#include <atomic>

class RuntimeFlags {
 public:
  // 获取单例实例
  static RuntimeFlags& instance() {
    static RuntimeFlags instance;
    return instance;
  }

  RuntimeFlags(const RuntimeFlags&) = delete;
  RuntimeFlags& operator=(const RuntimeFlags&) = delete;

  // 原子变量访问接口
  bool is_running() const { return running_.load(std::memory_order_acquire); }

  void stop() { running_.store(false, std::memory_order_release); }

 private:
  RuntimeFlags() : running_(true) {}

  std::atomic<bool> running_;
};

#endif