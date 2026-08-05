#!/bin/bash

set -e

ARCH="${ARCH:-arm64}"
DES_DIR="${1:-}"
SCRIPT_DIR="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTPUT_DIR="${3:-.}"

if [ -z "${DES_DIR}" ]; then
    echo "错误: 缺少安装目录参数"
    echo "用法: $0 <install_dir> [script_dir] [output_dir]"
    exit 1
fi

SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ ! -d "${DES_DIR}" ] || [ -z "$(find "${DES_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "警告: 安装目录不存在或为空 (${DES_DIR})，跳过 deb 包生成"
    echo "请先完成 make install 后再执行此脚本"
    exit 0
fi

SOURCE_DIR="$(mktemp -d)"
trap 'rm -rf "${SOURCE_DIR}"' EXIT

# 放到自己使用的目录下
prototype="l5b"
install_dir="wheel-upstairs"
package_name="${prototype}-${install_dir}"
user_bin_dir="${SOURCE_DIR}/user_space/user/${install_dir}"
mkdir -p "${user_bin_dir}"
cp -a "${DES_DIR}/." "${user_bin_dir}/"

if [ -d "${SCRIPT_DIR}/logrotate.d" ] && [ -n "$(find "${SCRIPT_DIR}/logrotate.d" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    user_lograte_dir="${SOURCE_DIR}/etc/logrotate.d"
    mkdir -p "${user_lograte_dir}"
    mkdir -p "${SOURCE_DIR}/user_space/user/log"
    cp -a "${SCRIPT_DIR}/logrotate.d/." "${user_lograte_dir}/"
fi

if [ -d "${SCRIPT_DIR}/rsyslog.d" ] && [ -n "$(find "${SCRIPT_DIR}/rsyslog.d" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    user_rsyslog_dir="${SOURCE_DIR}/etc/rsyslog.d"
    mkdir -p "${user_rsyslog_dir}"
    cp -a "${SCRIPT_DIR}/rsyslog.d/." "${user_rsyslog_dir}/"
fi

GIT_BRANCH="$(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
GIT_COMMIT_ID="$(git -C "${REPO_DIR}" log -1 --format=%h 2>/dev/null || echo "unknown")"

if [ -z "${MAINTAINER_INFO:-}" ]; then
    name="${USER:-}"
    if [ -z "${name}" ]; then
        name="$(id -u 2>/dev/null || echo "unknown")"
    fi
    MAINTAINER_INFO="Dev Team ${name}"
fi

APP_NAME="${APP_NAME:-${package_name}}"
DESCRIPTION="${DESCRIPTION:-${prototype} algorithm application}"
DEPENDS_PACKAGES="${DEPENDS:-}"

DESCRIPTION_INFO="${DESCRIPTION} (branch: ${GIT_BRANCH}, commit: ${GIT_COMMIT_ID})"
echo "${DESCRIPTION_INFO}"

CONFIRM_OVERWRITE_ARGS=()
if declare -p CONFIRM_OVERWRITE >/dev/null 2>&1; then
    for name in "${CONFIRM_OVERWRITE[@]}"; do
        CONFIRM_OVERWRITE_ARGS+=("--confirm-overwrite" "${name}")
    done
fi

SERVICE_ARGS=("--service-name" "gac_wheel")

COMMIT_ID="$(git -C "${REPO_DIR}" log -1 --format=%h 2>/dev/null || echo "0000000")"
pack_time="$(date +%Y%m%d)"
APP_VERSION="${APP_VERSION:-1.0.${pack_time}+${COMMIT_ID}}"

PACK_ARGS=(
  -n "${APP_NAME}"
  -v "${APP_VERSION}"
  -a "${ARCH}"
  -s "${SOURCE_DIR}"
  -o "${OUTPUT_DIR}"
  -m "${MAINTAINER_INFO}"
  -d "${DESCRIPTION_INFO}"
)

if [ -n "${DEPENDS_PACKAGES}" ]; then
    PACK_ARGS+=(--depends "${DEPENDS_PACKAGES}")
fi

PACK_ARGS+=("${SERVICE_ARGS[@]}" "${CONFIRM_OVERWRITE_ARGS[@]}")

bash "${SCRIPT_DIR}/deb_pack.sh" "${PACK_ARGS[@]}"
