project(l5a)
# set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${PROJECT_NAME})
# 完全隔离ROS2环境
if(DEFINED ENV{AMENT_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH_BACKUP ${CMAKE_PREFIX_PATH})
    list(REMOVE_ITEM CMAKE_PREFIX_PATH $ENV{AMENT_PREFIX_PATH})
endif()

# 强制指定libtorch路径
set(Torch_DIR "/usr/local/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

# 恢复ROS2环境
if(DEFINED CMAKE_PREFIX_PATH_BACKUP)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH_BACKUP})
endif()

# 手动创建现代CMake目标别名（关键修复）
if(TARGET torch AND NOT TARGET Torch::Torch)
    add_library(Torch::Torch INTERFACE IMPORTED)
    set_target_properties(
        Torch::Torch
        PROPERTIES INTERFACE_LINK_LIBRARIES "torch"
                   INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
                   INTERFACE_COMPILE_OPTIONS "${TORCH_CXX_FLAGS}"
    )
    message(
        STATUS "Created Torch::Torch interface target wrapping legacy 'torch'"
    )
endif()

# 强制目标验证
if(NOT TARGET Torch::Torch)
    message(FATAL_ERROR "Failed to create Torch::Torch target")
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

add_subdirectory(control)
add_subdirectory(publisher)
add_subdirectory(launch)
add_subdirectory(deploy)
add_subdirectory(test)
add_subdirectory(scripts)

if(SIM_ENABLE)
    find_package(glfw3 REQUIRED)
    add_executable(sim_l5a sim.cc)
    target_link_libraries(
        sim_l5a fsm mujoco::libsimulate mujoco::mujoco glfw
    )
    ament_target_dependencies(sim_l5a rclcpp)
    install(TARGETS sim_l5a DESTINATION bin)
    install(TARGETS sim_l5a DESTINATION lib/${PROJECT_NAME})
    install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
elseif(PHYSICS_ENABLE)
    add_library(stand_mode_lib stand_mode.cc)
    target_link_libraries(stand_mode_lib PUBLIC fsm glog)
    target_link_directories(
        stand_mode_lib PUBLIC ${CMAKE_INSTALL_PREFIX}/deploy
    )

    install(TARGETS stand_mode_lib DESTINATION ${CMAKE_INSTALL_PREFIX})

    # test
    add_executable(stand_mode_test stand_mode_test.cc)
    target_link_libraries(stand_mode_test PUBLIC stand_mode_lib)
    install(TARGETS stand_mode_test DESTINATION bin)
    install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
endif()

ament_package()
