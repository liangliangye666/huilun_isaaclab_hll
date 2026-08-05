cc_library(
    name = "pd_controller",
    # srcs = ["pd_controller.cc"],
    hdrs = ["pd_controller.h"],
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = ["//third_party:eigen"],
)

cc_library(
    name = "robot_model",
    srcs = [
        "robot_model.cc",
    ],
    hdrs = ["robot_model.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//sim/model/Y1A/publisher:SimPublisher",
        "//third_party:eigen",
        "//third_party:glog",
        "//third_party:mujoco_lib",
        "//third_party:pinocchio",
        "//third_party:rbdl",
        "//third_party:yaml_cpp",
    ],
)

cc_library(
    name = "wbc_weight",
    srcs = [
        "wbc_weight.cc",
    ],
    hdrs = ["wbc_weight.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pd_controller",
        ":robot_model",
        "//sim/model/Y1A/publisher:SimPublisher",
        "//third_party:eigen",
        "//third_party:glog",
        "//third_party:pinocchio",
        "//third_party:qpOASES",
        "//third_party:rbdl",
        "//third_party:yaml_cpp",
    ],
)

cc_library(
    name = "feedback_linear_control",
    srcs = [
        "feedback_linear_control.cc",
    ],
    hdrs = ["feedback_linear_control.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pd_controller",
        ":robot_model",
        "//sim/model/Y1A/publisher:SimPublisher",
        "//third_party:eigen",
        "//third_party:glog",
        "//third_party:pinocchio",
        "//third_party:qpOASES",
        "//third_party:yaml_cpp",
    ],
)

cc_library(
    name = "y1a_fsm",
    srcs = [
        "y1a_fsm.cc",
    ],
    hdrs = ["y1a_fsm.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pd_controller",
        ":robot_model",
        ":wbc_weight",
        "//sim/model/Y1A/publisher:SimPublisher",
        "//third_party:eigen",
        "//third_party:glog",
        "//third_party:mujoco_lib",
        "//third_party:qpOASES",
        "//third_party:yaml_cpp",
    ],
)
