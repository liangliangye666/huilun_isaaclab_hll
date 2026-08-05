cc_library(
    name = "eigen",
    srcs = [],
    hdrs = glob(["Eigen/**"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mujoco_lib",
    srcs = ["mujoco/lib/libmujoco.so.3.2.2"],
    hdrs = glob(["mujoco/**"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "rbdl",
    srcs = [
        "rbdl/lib/librbdl.so.3.3.1",
        "rbdl/lib/librbdl_urdfreader.so.3.3.1",
    ],
    hdrs = glob(["rbdl/include/rbdl/**"]),
    copts = ["-std=c++17"],
    includes = [
        "rbdl/include",
        "rbdl/include/rbdl",
        "rbdl/include/rbdl/addons/urdfreader",
    ],
    visibility = ["//visibility:public"],
    # deps = [
    #     "//third_party:eigen",
    # ],
)

cc_library(
    name = "glog",
    srcs = [
        "glog/lib/libglog.so",
        "glog/lib/libglog.so.0.7.1",
        "glog/lib/libglog.so.2",
    ],
    hdrs = glob(["glog/include/glog/*.h"]),
    includes = ["glog/include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "yaml_cpp",
    srcs = [
        "yaml_cpp/lib/libyaml-cpp.so.0.8",
        "yaml_cpp/lib/libyaml-cpp.so.0.8.0",
    ],
    hdrs = glob(["yaml_cpp/include/**/*"]),
    includes = ["yaml_cpp/include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "qpOASES",
    srcs = [
        "qpOASES/lib/libqpOASES.a",
    ],
    hdrs = glob(["qpOASES/**/*"]),
    includes = [
        "qpOASES/include",
        "qpOASES/include/qpOASES",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pinocchio",
    srcs = [
        "pinocchio/lib/libpinocchio_default.so.3.2.0",
        "pinocchio/lib/libpinocchio_parsers.so.3.2.0",
    ],
    hdrs = glob(["pinocchio/**/*"]),
    includes = [
        "pinocchio/include",
        "pinocchio/include/pinocchio",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pino_new",
    srcs = [
        "pino_new/libpinocchio_lin_x64.a",
    ],
    hdrs = glob(["pino_new/**/*"]),
    includes = [
        "pino_new",
        "pino_new/pinocchio",
    ],
    visibility = ["//visibility:public"],
    deps = ["//third_party:urdfdom"],
)

cc_library(
    name = "urdfdom",
    srcs = [
        "urdfdom/libconsole_bridge_lin_x64.a",
        "urdfdom/libtinyxml_lin_x64.a",
        "urdfdom/liburdfdom_model_lin_x64.a",
    ],
    hdrs = glob(["pino_new/**/*"]),
    includes = [
        ".",
    ],
    visibility = ["//visibility:public"],
)
