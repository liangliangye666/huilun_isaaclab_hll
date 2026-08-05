#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#undef NDEBUG
#include <glog/logging.h>
#include <iostream>

int main(int argc, char* argv[]) {
  google::InitGoogleLogging("glog_test");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  std::cout << "Hello World!" << std::endl;

  LOG(INFO) << argv[0];
  LOG(INFO) << "This is an info message";
  LOG(WARNING) << "This is a warning message";
  LOG(ERROR) << "This is an error message";

  if (-1 < 0) {
    // 抛出异常并附带错误信息
    throw std::runtime_error("输入值不能为负数");
  }

  LOG(INFO) << "END";
  google::ShutdownGoogleLogging();
  return 0;
}
