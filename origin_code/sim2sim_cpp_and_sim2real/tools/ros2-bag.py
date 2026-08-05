"""
Bazel-based entry point for `ros2` binary, e.g.
    ros2 interfaces list
    ros2 topic list
    ros2 bag record
"""

import os
import sys

from datetime import datetime
from bazel_tools.tools.python.runfiles import runfiles


def main():
    workspace_path = os.environ.get("WORKSPACE_PATH")
    now = datetime.now()
    date_time_str = now.strftime("-%Y-%m-%d %H:%M:%S")

    manifest = runfiles.Create()
    bin_file = manifest.Rlocation("ros2/ros2")
    argv = [bin_file] + sys.argv[1:]

    argv = [
        bin_file,
        "bag",
        "record",
        "-a",
        "-o",
        workspace_path + "/data/ros2_bag" + date_time_str,
    ]

    os.execv(bin_file, argv)


assert __name__ == "__main__"
main()
