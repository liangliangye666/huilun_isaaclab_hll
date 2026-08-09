# 当前 ARM64 产物只需要源码重新编译出的 libstand_mode_lib.so、控制库、
# YAML、模型和机器人资产。旧整机应用包 drivers/ 不再安装到 install_arm64，
# 避免生成 wheel/ 目录和根部 double_wheel_app_v3。
if(PHYSICS_ENABLE)
endif()
