load("@ros2//:ros_py.bzl", "ros_launch")
load("@ros2//:ros_cc.bzl", "ros_cc_binary")

ros_cc_binary(
    name = "main",
    srcs = ["main.cc"],
    linkopts = [
        "-L/usr/lib/x86_64-linux-gnu",
        "-lglfw",
    ],
    rmw_implementation = "rmw_cyclonedds_cpp",
    tags = ["requires-network"],
    deps = [
        "//platforms/Y1A/control:y1a_fsm",
        "//sim/model/Y1A/publisher:SimPublisher",
        "//sim/mujoco",
        "//third_party:glog",
        "//third_party:mujoco_lib",
        "//third_party:rbdl",
        "@ros2//:rosidl_runtime_c_cc",
        "@ros2//:rosidl_typesupport_cpp_cc",
        "@ros2//resources/rmw_isolation:rmw_isolation_cc",
    ],
)

workspace_name = "gac-robotics"

ros_launch(
    name = "Y1A",
    data = [
        ":main",
    ],
    launch_file = "launch/roslaunch_mujoco.py",
    workspace_name = workspace_name,
)
