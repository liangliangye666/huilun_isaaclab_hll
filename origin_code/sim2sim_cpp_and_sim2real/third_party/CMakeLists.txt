project(third_party)

# 检测当前架构
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64)")
  set(ARCH "ARM64")
  add_subdirectory(arm64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i[3-6]86)")
  set(ARCH "x86")
  add_subdirectory(x86)
else()
  set(ARCH "Unknown")
endif()

# 打印消息
message(STATUS "Detected Architecture: ${ARCH}")
