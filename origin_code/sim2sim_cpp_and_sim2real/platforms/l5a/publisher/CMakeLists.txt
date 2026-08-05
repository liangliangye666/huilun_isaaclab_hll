find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)

add_library(simpublisher SHARED ros_publisher.cc)
target_include_directories(simpublisher PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(simpublisher glog yaml-cpp Eigen3::Eigen)
ament_target_dependencies(simpublisher rclcpp std_msgs)

rosidl_target_interfaces(simpublisher l5a_msgs "rosidl_typesupport_cpp")

install(TARGETS simpublisher DESTINATION lib)
