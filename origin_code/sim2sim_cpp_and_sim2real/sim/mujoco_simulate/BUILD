cc_library(
    name = "mujoco",
    srcs = [
        "glfw_adapter.cc",
        "glfw_dispatch.cc",
        "lodepng.cc",
        "platform_ui_adapter.cc",
        "simulate.cc",
    ],
    hdrs = glob(["*.h"]),
    copts = ["-std=c++17"],
    includes = [
        ".",
        "../control",
    ],
    visibility = ["//visibility:public"],
    deps = [
        # "//platforms/Y3A/control",
        "//sim:config",
        "//third_party:mujoco_lib",
    ],
)
