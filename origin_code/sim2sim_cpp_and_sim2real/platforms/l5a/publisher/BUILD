cc_library(
    name = "SimPublisher",
    srcs = ["SimPublisher.cc"],
    hdrs = ["SimPublisher.h"],
    includes = [
        ".",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//sim/model/Y1A:ros2_msgs_cc",
        "//sim/mujoco",
        "//third_party:eigen",
        "//third_party:glog",
        "//third_party:yaml_cpp",
        "@ros2//:geometry_msgs_cc",
        "@ros2//:nav_msgs_cc",
        "@ros2//:rclcpp_cc",
        "@ros2//:rmw_cc",
        "@ros2//:sensor_msgs_cc",
        "@ros2//:std_msgs_cc",
        "@ros2//:tf2_cc",
        "@ros2//:tf2_ros_cc",
    ],
)
