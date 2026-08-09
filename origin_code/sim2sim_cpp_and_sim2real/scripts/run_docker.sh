#!/bin/bash

# 根据当前小工程目录和 Git 分支生成容器名，避免误连接同分支下的旧工程容器。

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
project_name="$(basename "${project_root}")"

docker_image=maguangcai/gac-robotics-dev:latest # 定义要使用的Docker镜像
if branch_name=$(git -C "${project_root}" rev-parse --abbrev-ref HEAD 2>/dev/null); then
    safe_branch="${branch_name//\//-}"
    id="${project_name}-${safe_branch}"
else # 如果失败(不在git仓库),打印警告并把id设为默认值
    echo "Warning: Not in Git repository, use a standalone container name."
    id="${project_name}-ros-foxy"
fi

for gid in $(id -G); do # 循环当前用户所属的所有组ID(id -G),把每个--group-add <gid>拼接到变量group_add_opts上,后续用于docker run,目的是把主机用户的组权限带入容器
  group_add_opts="$group_add_opts --group-add $gid"
done

if [ "$(docker ps -q --filter "name=^$id$")" ]; then # 查询正在运行的容器,名字完全匹配^$id$
    echo "Container $id is already running."    
    echo "Attach on Container $id."
    docker exec -it "$id" bash --rcfile ~/.bashrc # 如果有输入打印上述提示,并且用docker exec -it ...进入容器; --rcfile ~/.bashrc是让bash启动时加载指定的rc文件(容器内的路径)

else
    if [ "$(docker ps -aq --filter "name=^$id$")" ]; then
        echo "Starting existing container $id."
        docker start "$id"
        docker exec -it "$id" bash --rcfile ~/.bashrc
    else
        docker pull $docker_image # 拉镜像(可能比较慢)
        echo "Creating and starting new container $id."
        docker run \
            --network host \
            --privileged \
            --name="$id" \
            --rm \
            --interactive \
            --tty \
            --workdir "${project_root}" \
            --hostname "$(hostname)" \
            --gpus all \
            --env "DISPLAY=$DISPLAY" \
            --env "QT_X11_NO_MITSHM=1" \
            --env "NVIDIA_DRIVER_CAPABILITIES=all" \
            --env="WORKSPACE_PATH=${project_root}" \
            --volume "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            --volume "/run/user:/run/user" \
            --volume "/tmp:/tmp" \
            --volume "/dev:/dev" \
            --volume "$HOME/.ssh:$HOME/.ssh" \
            --volume "/etc/localtime:/etc/localtime:ro" \
            --volume "/etc/passwd:/etc/passwd:ro" \
            --volume "/etc/shadow:/etc/shadow:ro" \
            --volume "/etc/group:/etc/group:ro" \
            --volume "/etc/gshadow:/etc/gshadow:ro" \
            --volume "/etc/apt/apt.conf:/etc/apt/apt.conf:ro" \
            --volume "${project_root}/scripts/bashrc:$HOME/.bashrc" \
            --volume "$HOME/.cache:$HOME/.cache:rw" \
            --volume "$HOME/.ccache:$HOME/.ccache:rw" \
            --volume "$HOME/.gitconfig:$HOME/.gitconfig:rw" \
            --volume "$HOME/.vscode/extensions:$HOME/.vscode-server/extensions:rw" \
            --tmpfs "$HOME:exec,rw,uid=$(id -u)" \
            --tmpfs "$HOME/.vscode-server:exec,rw,uid=$(id -u)" \
            --volume "${project_root}:${project_root}" \
            --user "$(id -u)" \
            $group_add_opts \
            $docker_image \
            bash --rcfile ~/.bashrc
    fi
fi
