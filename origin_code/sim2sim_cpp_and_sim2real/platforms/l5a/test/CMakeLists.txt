project(test)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Test code
add_executable(yaml_cpp_test yaml_cpp_test.cc)
add_executable(eigen_test eigen_test.cc)
add_executable(glog_my_test glog_test.cc)
add_executable(qpOASES_test qpOASES_test.cc)
add_executable(pinocchio_test pinocchio_test.cc)
add_executable(talker publisher_lambda_function.cpp)
add_executable(listener subscriber_lambda_function.cpp)
add_executable(thread_test thread_test.cc)
add_executable(get_initpos get_initpos.cc)

target_link_libraries(yaml_cpp_test PUBLIC yaml-cpp Eigen3::Eigen)
target_link_libraries(eigen_test PUBLIC Eigen3::Eigen glog gac_math)
target_link_libraries(glog_my_test PUBLIC glog)
target_link_libraries(qpOASES_test PUBLIC glog Eigen3::Eigen qpOASES)
target_link_libraries(pinocchio_test PUBLIC glog Eigen3::Eigen pinocchio)
target_link_libraries(talker glog Eigen3::Eigen)
target_link_libraries(listener glog Eigen3::Eigen)
target_link_libraries(thread_test pthread)
target_link_libraries(get_initpos PUBLIC glog Eigen3::Eigen pinocchio)
ament_target_dependencies(talker rclcpp std_msgs)
ament_target_dependencies(listener rclcpp std_msgs)

target_include_directories(eigen_test PUBLIC ${CMAKE_SOURCE_DIR}/tools)
target_include_directories(get_initpos PUBLIC ${CMAKE_SOURCE_DIR}/math)

install(
  TARGETS yaml_cpp_test
          eigen_test
          glog_my_test
          qpOASES_test
          pinocchio_test
          talker
          listener
          thread_test
          get_initpos
  DESTINATION bin)

rosidl_target_interfaces(talker l5a_msgs "rosidl_typesupport_cpp")

if(SIM_ENABLE)
  add_executable(mujoco_test mujoco_test.cc)
  target_link_libraries(mujoco_test PUBLIC glog mujoco::libsimulate
                                           mujoco::mujoco glfw pthread)
  install(TARGETS mujoco_test DESTINATION bin)

endif(SIM_ENABLE)
