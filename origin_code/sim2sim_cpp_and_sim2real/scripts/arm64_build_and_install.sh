#!/bin/bash
rm -r install_arm64

root_path=$(pwd)
cmake --fresh -DCMAKE_TOOLCHAIN_FILE=${root_path}/cmake/aarch64_rostoolchain.cmake -B build_arm64 -S .
time make install -j --no-print-directory -C build_arm64

bash "${root_path}/deb_tool/app_deb_generate.sh" \
  "${root_path}/install_arm64" \
  "${root_path}/deb_tool" \
  "${root_path}/install_arm64"

# install
branch_name=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

if [ -z "$branch_name" ]; then
    echo "Error: Not in a Git repository, exiting script."
    exit 1
fi

# 检查 SSH 连接是否可用
if ! ssh -o ConnectTimeout=3 gac-robotics "exit" &>/dev/null; then
    exit 1
fi

target_dir="/user_space/$branch_name"
echo "✅ SSH connection OK, installing to $target_dir ..."
rsync -avz --chmod=ugo=rx,u+w ./install_arm64/ gac-robotics:"$target_dir/"
echo "✅ Install complete."