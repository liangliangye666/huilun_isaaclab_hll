#!/bin/bash
# deb-package.sh - 针对 Ubuntu/Linux 的 deb 包制作脚本

set -e

show_usage() {
    cat << 'EOF'
使用方法: $0 [选项]
选项:
  -n, --name <包名>         设置软件包名称 (必需)
  -v, --version <版本>      设置版本号 (必需，格式: 主版本.次版本.修订版)
  -a, --arch <架构>         设置目标架构 (默认: arm64)
                           可选: amd64, arm64, i386, all
  -s, --src-dir <源目录>    包含已准备好的文件的目录 (必需)
  -o, --output-dir <输出目录> 设置deb包输出目录 (默认: 当前目录)
  -m, --maintainer <信息>   设置维护者信息 (默认: 当前用户)
  -d, --description <描述>  设置软件包描述
  --service-name <服务名>   安装/升级时停止并启动已有systemd服务，可多次指定
  --depends <依赖>         设置依赖包 (逗号分隔)
  --confirm-overwrite <路径> 需要在安装时手动确认覆盖的文件/目录，可多次指定
  -h, --help               显示此帮助信息
EOF
}

APP_NAME=""
APP_VERSION="0.0.0"
APP_ARCH="arm64"
SRC_DIR=""
OUTPUT_DIR="."
MAINTAINER="${USER:-unknown} <${USER:-unknown}@localhost>"
DESCRIPTION="Application package"
SECTION="utils"
PRIORITY="optional"
DEPENDS=""
CONFIRM_OVERWRITE=()
SERVICE_NAMES=()

color_echo() {
    echo -e "\033[1;${1}m${2}\033[0m"
}

error_exit() {
    color_echo 31 "错误: $1"
    exit 1
}

quote_array_items() {
    local item
    local quoted_item

    for item in "$@"; do
        printf -v quoted_item '%q' "${item}"
        echo -n "${quoted_item} "
    done
}

create_deb_package() {
    local temp_dir
    temp_dir="$(mktemp -d)"
    trap "rm -rf '${temp_dir}'" EXIT

    local debian_dir="${temp_dir}/DEBIAN"
    local pkg_data_dir="/var/lib/${APP_NAME}"
    local protect_file="${pkg_data_dir}/protected_files"
    local backup_dir="${pkg_data_dir}/backup"

    color_echo 32 "开始创建deb包: ${APP_NAME}_${APP_VERSION}_${APP_ARCH}"
    if [ ${#SERVICE_NAMES[@]} -gt 0 ]; then
        color_echo 33 "安装时管理已有服务: ${SERVICE_NAMES[*]}"
    fi
    if [ ${#CONFIRM_OVERWRITE[@]} -gt 0 ]; then
        color_echo 33 "安装时需确认覆盖: ${CONFIRM_OVERWRITE[*]}"
    fi

    color_echo 36 "复制文件从 ${SRC_DIR} 到打包目录..."
    if [[ -d "${SRC_DIR}" ]]; then
        cp -a "${SRC_DIR}/." "${temp_dir}/"
    else
        error_exit "源目录无效: ${SRC_DIR}"
    fi

    rm -rf "${debian_dir}"
    mkdir -p "${debian_dir}"

    {
        echo "Package: ${APP_NAME}"
        echo "Version: ${APP_VERSION}"
        echo "Section: ${SECTION}"
        echo "Priority: ${PRIORITY}"
        echo "Architecture: ${APP_ARCH}"
        echo "Maintainer: ${MAINTAINER}"
        if [[ -n "${DEPENDS}" ]]; then
            local formatted_depends
            formatted_depends="$(echo "${DEPENDS}" | sed 's/,/, /g')"
            echo "Depends: ${formatted_depends}"
        fi
        echo "Description: ${DESCRIPTION}"
        echo "Source: ${APP_NAME}"
    } > "${debian_dir}/control"

    local confirm_overwrite_str
    confirm_overwrite_str="$(quote_array_items "${CONFIRM_OVERWRITE[@]}")"
    local service_names_str
    service_names_str="$(quote_array_items "${SERVICE_NAMES[@]}")"

    cat > "${debian_dir}/preinst" << EOF
#!/bin/bash
set -e

APP_NAME="${APP_NAME}"
NEW_VERSION="${APP_VERSION}"
CONFIRM_OVERWRITE=(${confirm_overwrite_str})
SERVICE_NAMES=(${service_names_str})
PKG_DATA_DIR="${pkg_data_dir}"
PROTECT_FILE="${protect_file}"
BACKUP_DIR="${backup_dir}"

stop_managed_services() {
    [ \${#SERVICE_NAMES[@]} -gt 0 ] || return 0

    if ! command -v systemctl >/dev/null 2>&1; then
        echo "systemctl 不存在，跳过停止服务: \${SERVICE_NAMES[*]}"
        return 0
    fi

    for svc in "\${SERVICE_NAMES[@]}"; do
        echo "停止服务: \$svc"
        systemctl stop "\$svc" 2>/dev/null || echo "停止服务失败或服务不存在: \$svc，继续安装"
    done
}

mkdir -p "\$PKG_DATA_DIR" "\$BACKUP_DIR"
touch "\$PROTECT_FILE"

echo "============================================="
echo "准备安装 \$APP_NAME 版本 \$NEW_VERSION"
echo "============================================="

stop_managed_services

PROTECTED_PATHS=()
if [ -f "\$PROTECT_FILE" ]; then
    while IFS= read -r line; do
        [ -n "\$line" ] && PROTECTED_PATHS+=("\$line")
    done < "\$PROTECT_FILE"
fi

if dpkg -l "\$APP_NAME" &>/dev/null; then
    CURRENT_VER=\$(dpkg-query -W -f='\${Version}' "\$APP_NAME" 2>/dev/null || echo "0")
    echo "已安装版本: \$CURRENT_VER"
fi

if [ \${#CONFIRM_OVERWRITE[@]} -gt 0 ]; then
    echo -e "\n以下路径安装时需要确认是否覆盖："
    for p in "\${CONFIRM_OVERWRITE[@]}"; do echo "  - \$p"; done
    echo "=================================================="

    for TARGET in "\${CONFIRM_OVERWRITE[@]}"; do
        if [[ " \${PROTECTED_PATHS[@]} " =~ " \$TARGET " ]]; then
            echo "已保护: \$TARGET (跳过覆盖)"
            continue
        fi

        if [ ! -e "\$TARGET" ]; then
            echo "新建: \$TARGET"
            continue
        fi

        echo -e "\n文件/目录已存在: \$TARGET"
        read -p "是否覆盖？[y/N] " -n 1 -r
        echo
        if [[ ! \$REPLY =~ ^[Yy]\$ ]]; then
            echo "保留原有文件: \$TARGET"
            echo "\$TARGET" >> "\$PROTECT_FILE"
            sort -u "\$PROTECT_FILE" -o "\$PROTECT_FILE"

            rel_path="\${TARGET#/}"
            backup_path="\$BACKUP_DIR/\$rel_path"
            mkdir -p "\$(dirname "\$backup_path")"

            if [ -d "\$TARGET" ]; then
                mkdir -p "\$backup_path"
                cp -a "\$TARGET"/. "\$backup_path"/
            else
                cp -a "\$TARGET" "\$backup_path"
            fi
        else
            echo "允许覆盖: \$TARGET"
        fi
    done
    echo
fi

exit 0
EOF

    cat > "${debian_dir}/postinst" << EOF
#!/bin/bash
set -e

APP_NAME="${APP_NAME}"
SERVICE_NAMES=(${service_names_str})
PKG_DATA_DIR="${pkg_data_dir}"
PROTECT_FILE="${protect_file}"
BACKUP_DIR="${backup_dir}"

start_managed_services() {
    [ \${#SERVICE_NAMES[@]} -gt 0 ] || return 0

    if ! command -v systemctl >/dev/null 2>&1; then
        echo "systemctl 不存在，跳过启动服务: \${SERVICE_NAMES[*]}"
        return 0
    fi

    for svc in "\${SERVICE_NAMES[@]}"; do
        echo "启动服务: \$svc"
        systemctl start "\$svc" 2>/dev/null || echo "启动服务失败或服务不存在: \$svc，请手动检查"
    done
}

echo "配置 \$APP_NAME..."

if [ -f "\$PROTECT_FILE" ]; then
    while IFS= read -r path; do
        [ -n "\$path" ] || continue
        rel_path="\${path#/}"
        backup_path="\$BACKUP_DIR/\$rel_path"

        if [ -e "\$backup_path" ]; then
            echo "恢复原有文件: \$path"
            rm -rf "\$path"
            if [ -d "\$backup_path" ]; then
                mkdir -p "\$path"
                cp -a "\$backup_path"/. "\$path"/
            else
                cp -a "\$backup_path" "\$path"
            fi
        fi
    done < "\$PROTECT_FILE"
fi

rm -rf "\$BACKUP_DIR"
rm -f "\$PROTECT_FILE"
rmdir --ignore-fail-on-non-empty "\$PKG_DATA_DIR"

start_managed_services

echo "============================================="
echo "\$APP_NAME 安装完成"
echo "============================================="
exit 0
EOF

    cat > "${debian_dir}/prerm" << EOF
#!/bin/bash
set -e

SERVICE_NAMES=(${service_names_str})

stop_managed_services() {
    [ \${#SERVICE_NAMES[@]} -gt 0 ] || return 0

    if ! command -v systemctl >/dev/null 2>&1; then
        echo "systemctl 不存在，跳过停止服务: \${SERVICE_NAMES[*]}"
        return 0
    fi

    for svc in "\${SERVICE_NAMES[@]}"; do
        echo "停止服务: \$svc"
        systemctl stop "\$svc" 2>/dev/null || echo "停止服务失败或服务不存在: \$svc，继续处理"
    done
}

if [[ "\$1" == "remove" || "\$1" == "upgrade" ]]; then
    stop_managed_services
fi

exit 0
EOF

    cat > "${debian_dir}/postrm" << EOF
#!/bin/bash
set -e

APP_NAME="${APP_NAME}"

if [[ "\$1" == "purge" ]]; then
    rm -rf /var/lib/\$APP_NAME
fi

exit 0
EOF

    chmod 755 "${debian_dir}/preinst" "${debian_dir}/postinst" "${debian_dir}/prerm" "${debian_dir}/postrm"

    mkdir -p "${OUTPUT_DIR}"
    local deb_file="${OUTPUT_DIR}/${APP_NAME}_${APP_VERSION}_${APP_ARCH}.deb"

    color_echo 36 "构建deb包中..."
    if dpkg-deb --build --root-owner-group "${temp_dir}" "${deb_file}"; then
        color_echo 32 "构建成功: ${deb_file}"
    else
        error_exit "deb包构建失败"
    fi

    rm -rf "${temp_dir}"
    trap - EXIT
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [[ $# -eq 0 ]]; then
        show_usage
        exit 0
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -n|--name) APP_NAME="$2"; shift 2 ;;
            -s|--src-dir) SRC_DIR="$2"; shift 2 ;;
            -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            --service-name) SERVICE_NAMES+=("$2"); shift 2 ;;
            --depends) DEPENDS="$2"; shift 2 ;;
            -v|--version) APP_VERSION="$2"; shift 2 ;;
            -a|--arch) APP_ARCH="$2"; shift 2 ;;
            -m|--maintainer) MAINTAINER="$2"; shift 2 ;;
            -d|--description) DESCRIPTION="$2"; shift 2 ;;
            --confirm-overwrite) CONFIRM_OVERWRITE+=("$2"); shift 2 ;;
            -h|--help) show_usage; exit 0 ;;
            *) error_exit "未知参数: $1" ;;
        esac
    done

    [[ -z "${APP_NAME}" || -z "${SRC_DIR}" ]] && error_exit "缺少包名/源目录！"
    [[ ! -d "${SRC_DIR}" ]] && error_exit "源目录不存在: ${SRC_DIR}"

    create_deb_package
fi
