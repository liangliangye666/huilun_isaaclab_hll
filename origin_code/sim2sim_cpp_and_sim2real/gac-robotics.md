---
modified: 2026-04-16T11:42:56.978Z
title: gac-robotics
---

# gac-robotics

# 1. 环境配置
## 1.1 拉取 Dokcer 镜像
```Shell
docker pull maguangcai/gac-robotics-dev:latest
```
# 2. 编译,在项目根目录执行以下命令:
## 2.1 运行 Docker
```Shell
./scripts/run_docker.sh
```
## 2.2 编译与安装
```Shell
./scrpits/build_and_install.sh # 编译仿真
./scrpits/arm64_build_and_install.sh # 编译真机
```
# 3. 代码执行
## 3.1 仿真运行代码（以 L2C 为例）
**不记录数据启动**
```Shell
./install/bin/sim_l2c
```
**记录数据启动**
```Shell
ros2 launch L2C mujoco.py
```
## 3.2 实机运行代码(以L2C为例)
### 3.2.1. 连接控制器 wifi,使用 vscode 左下角打开远程窗口,初次连接可能需要给控制器联网,联网教程:
**板端运行:**
```shell
sudo route add -net 0.0.0.0/0 gw 192.168.0.87
sudo chmod +666 /etc/resolv.conf
sudo echo "nameserver 114.114.114.114" > /etc/resolv.conf
其中 192.168.8.203 为本机的 WiFi ip 需要使用 ifconfig 命令检查
```
**本地运行:**
```shell
sudo bash -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'
sudo iptables -F
sudo iptables -P INPUT ACCEPT
sudo iptables -P FORWARD ACCEPT
sudo iptables -t nat -A POSTROUTING -o Mihomo -j MASQUERADE
其中 Meta 是本地的网口的名称.
```
### 3.2.2. 连接到远端之后,将编译生成的安装文件(install_arm)中所有子文件复制到远端控制器指定目录
```shell
查看是否自启开启: ps -ef|grep double_wheel_v3,如果自启开启,则返回三行,若关闭,则返回一行;
关掉自启: sudo systemctl stop gac_wheel, 开启自启: sudo systemctl start gac_wheel
scp -r install_arm64/* root@192.168.0.1:/user_space/mgc
其中,mgc是个人调试文件夹,需要根据实际情况修改
```
### 3.2.3. 配置控制器环境变量
```shell
. script/set_env.sh
```
### 3.2.4. 启动控制器代码
**不记录数据启动**
```shell
./double_wheel_v3
```
**记录数据启动**
```shell
./script/run_l2c.sh
```
# 5. 查看数据
# 5.1. 使用plotjuggler查看数据
**启动 plotjuggler,数据保存在项目根目录的`data`文件夹下**
```Shell
ros2 run plotjuggler plotjuggler
```
将数据中的`yaml`文件拖入 plotjuggler 中即可
# 5.2. 使用xcp软件实时查看数据
**在robot_model对象中,定义了一个数组:observe_value,待观测量赋值给对应维度数组信息即可**
