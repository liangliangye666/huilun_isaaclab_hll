#!/bin/bash

# 获取当前 git 分支名
branch_name=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

# 检查是否能获取到分支名
if [ -z "$branch_name" ]; then
    echo "Error: Not in a Git repository, exiting script."
    exit 1  # 退出脚本
fi

# 目标目录为当前 git 分支名
target_dir="/user_space/$branch_name"

scp -r ./install_arm64/* root@192.168.0.1:$target_dir

