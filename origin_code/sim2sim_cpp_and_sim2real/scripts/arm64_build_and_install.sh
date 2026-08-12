#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
build_dir="${project_root}/build_arm64"
install_dir="${project_root}/install_arm64"
toolchain_file="${project_root}/cmake/aarch64_rostoolchain.cmake"
deb_tool_dir="${project_root}/deb_tool"

rm -rf -- "${install_dir}"

cmake --fresh \
  -DCMAKE_TOOLCHAIN_FILE="${toolchain_file}" \
  -B "${build_dir}" \
  -S "${project_root}"
time cmake --build "${build_dir}" --target install --parallel

echo "ARM64 install bundle: ${install_dir}"

bash "${deb_tool_dir}/app_deb_generate.sh" \
  "${install_dir}" \
  "${deb_tool_dir}" \
  "${install_dir}"

echo "ARM64 deb package output dir: ${install_dir}"
