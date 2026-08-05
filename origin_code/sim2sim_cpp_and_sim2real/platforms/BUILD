load("@bazel_skylib//rules:common_settings.bzl", "string_flag")

platform(
    name = "platform_y1a",
    constraint_values = [
        ":y1a_platform_constraint",
    ],
)

platform(
    name = "platform_y3a",
    constraint_values = [
        ":y3a_platform_constraint",
    ],
)

string_flag(
    name = "y1a_platform_constraint",
    build_setting_default = "Y1A",
)

string_flag(
    name = "y3a_platform_constraint",
    build_setting_default = "Y3A",
)
