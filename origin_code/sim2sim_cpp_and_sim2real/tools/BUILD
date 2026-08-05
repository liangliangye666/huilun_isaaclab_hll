# -*- python -*-

load(
    "@ros2//:ros_py.bzl",
    "ros_py_binary",
)

exports_files(["cmd_exec.sh"])

ros_py_binary(
    name = "ros2-bag",
    srcs = ["ros2-bag.py"],
    data = [
        "@ros2",
    ],
    visibility = ["//:__subpackages__"],
    deps = [
        # This is the main key for tools like `ros2 topic echo` and
        # `ros2 bag record` to work with custom messages. You must provide the
        # generated definitions as a dependency. Ideally, it should come
        # through a "roll-up" target.
        "//sim:ros2_msgs_py",
        "@bazel_tools//tools/python/runfiles",
    ],
)

# This is a roll-up target to use plotjuggler with the required
# ROS plugins. Usage :
# bazel run //tools:plotjuggler
ros_py_binary(
    name = "plotjuggler",
    srcs = ["plotjuggler.py"],
    data = [
        "@ros2//:plotjuggler_ros_cc",
        "@ros2//:plotjuggler_ros_share",
        "@ros2//:plotjuggler_ros_transitive_py",
    ],
    deps = [
        "//sim/model/Y1A:ros2_msgs_py",
        "@bazel_tools//tools/python/runfiles",
        "@ros2//:nav_msgs_py",
        "@ros2//:plotjuggler_plotjuggler",
    ],
)
