#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
build_dir="${project_root}/build"
install_dir="${project_root}/install"

rm -rf -- "${install_dir}"

cmake --fresh -B "${build_dir}" -S "${project_root}"
time cmake --build "${build_dir}" --target install --parallel
