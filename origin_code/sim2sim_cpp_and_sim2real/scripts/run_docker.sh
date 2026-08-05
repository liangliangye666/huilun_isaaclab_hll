#!/bin/bash

#根据当前Git分支名作为容器名,检查同名容器是否在运行;若在运行则exec进入它,否则拉取镜像并用一组绑定(volumes/tmpfs/device/env/group)创建并启动一个交互式容器,进入bash

docker_image=maguangcai/gac-robotics-dev:latest # 定义要使用的Docker镜像
if branch_name=$(git rev-parse --abbrev-ref HEAD 2>/dev/null); then # 尝试用git rev-parse --abbrev-ref HEAD获取当前分支名并赋给branch_name
    id="$branch_name" # 如果命令成功(脚本在git仓库里),就把容器名id设为分支名
else # 如果失败(不在git仓库),打印警告并把id设为默认值
    echo "Warning: Not in Git repository, use default ID 'ros-humble'"
    id=ros-foxy  
fi

for gid in $(id -G); do # 循环当前用户所属的所有组ID(id -G),把每个--group-add <gid>拼接到变量group_add_opts上,后续用于docker run,目的是把主机用户的组权限带入容器
  group_add_opts="$group_add_opts --group-add $gid"
done

if [ "$(docker ps -q --filter "name=^$id$")" ]; then # 查询正在运行的容器,名字完全匹配^$id$
    echo "Container $id is already running."    
    echo "Attach on Container $id."
    docker exec -it $id bash --rcfile ~/.bashrc # 如果有输入打印上述提示,并且用docker exec -it ...进入容器; --rcfile ~/.bashrc是让bash启动时加载指定的rc文件(容器内的路径)

else
    if [ "$(docker ps -aq --filter "name=^$id$")" ]; then
        echo "Starting existing container $id."
        docker start $id
        docker exec -it $id bash --rcfile ~/.bashrc
    else
        docker pull $docker_image # 拉镜像(可能比较慢)
        echo "Creating and starting new container $id."
        docker run \
            --network host \
            --privileged \
            --name=$id \
            --rm \
            --interactive \
            --tty \
            --workdir $(pwd) \
            --hostname $(hostname) \
            --gpus all \
            --env "DISPLAY=$DISPLAY" \
            --env "QT_X11_NO_MITSHM=1" \
            --env "NVIDIA_DRIVER_CAPABILITIES=all" \
            --env="WORKSPACE_PATH=$(pwd)" \
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
            --volume "$(pwd)/scripts/bashrc:$HOME/.bashrc" \
            --volume "$HOME/.cache:$HOME/.cache:rw" \
            --volume "$HOME/.ccache:$HOME/.ccache:rw" \
            --volume "$HOME/.gitconfig:$HOME/.gitconfig:rw" \
            --volume "$HOME/.vscode/extensions:$HOME/.vscode-server/extensions:rw" \
            --tmpfs "$HOME:exec,rw,uid=$(id -u)" \
            --tmpfs "$HOME/.vscode-server:exec,rw,uid=$(id -u)" \
            --volume $(pwd):$(pwd) \
            --user $(id -u) \
            $group_add_opts \
            $docker_image \
            bash --rcfile ~/.bashrc
    fi
fi


