基于 Isaac Gym 仿真平台训练轮足机器人的控制算法

# 环境配置

## 硬件需求

使用 Isaac Gym 进行训练刚需一张支持 CUDA 功能的 Nvidia 显卡，为了流畅的进行训练以及可视化训练结果，推荐使用显存 16G 以上的 RTX 显卡。

## 系统配置

Isaac Gym 的训练需要在 Linux 系统上进行，推荐使用 Ubuntu20.04，并将 Isaac Gym 配置在 conda 虚拟环境中运行。

## 软件配置

### 安装 Nvidia 驱动con

在 Ubuntu20.04 上安装 Nvidia 驱动，推荐安装 550 版本的驱动，使用 `nvidia-smi` 命令查看驱动是否安装成功。

![nvidia-smi](img/nvidia-smi.png)

### 安装 CUDA

在 Ubuntu20.04 上安装 CUDA11.4，安装完成后可以使用 `nvcc --version` 命令查看 CUDA 版本，如果显示 CUDA 版本信息，则安装成功(此步可省略,Nidia 驱动会自动安装 CUDA,使用`nvidia-smi`查看相应版本)。

![nvcc](img/nvcc.png)

### 安装 miniconda

在 Ubuntu20.04 上安装 miniconda，安装完成后可以使用 `conda --version` 命令查看 miniconda 版本，如果显示 miniconda 版本信息，则安装成功。

![conda](img/conda.png)

## 强化学习训练框架

1. 配置 python 虚拟环境（推荐使用 Python3.8 版本）。

   - 方法 1:在当前项目下创建环境: `conda create --prefix ./env python=3.8`,优先使用此方法;
   - 方法 2:在默认目录下创建虚拟环境： `conda create -n scut python=3.8`

     ![condacreate](img/condacreate.png)

   - 激活项目下环境: `conda activate ./env `,对应方法 1;
   - 激活虚拟环境： `conda activate scut`,对应方法 2;

     ![condaactivate](img/condaactivate.png)

2. 先进 conda 环境,再安装 pytorch 1.13.1 和 cuda-11.7

   - 使用 pip 安装： `pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117  --extra-index-url https://download.pytorch.org/whl/cu117`

     ![pipinstall](img/pipinstall.png)

3. 下载并安装 Isaac Gym

- NVIDIA 官网下载[Isaac Gym](https://developer.nvidia.com/isaac-gym/download)，解压缩后得到 `isaacgym` 文件夹。
- 在虚拟环境中,进入 `issacgym/python` 文件夹，使用 pip 安装： `pip install -e .`

  ![installisaacgym](img/installisaacgym.png)

- 进入 `examples` 文件夹，运行示例代码进行验证： `python franka_cube_ik_osc.py`

  ![runexample](img/runexample.png)

  - 如果出现 `ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory` 错误，则需要将 python 的路径添加到系统环境变量中。

  - 使用命令: `export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/mgc/code/gac-robotics/env/lib`
  - 或者执行: `. wheel_legged_gym/scripts/gym_env_setup.sh`

    ![ImportError](img/ImportError.png)

  - 成功运行示例代码后，可以看到如下界面：

    ![success](img/success.png)

4.  安装训练工程

    文件夹 `wheel_legged_gym` 中包含了: envs-训练任务, rsl_rl-强化学习算法库, scripts-训练脚本, test-测试脚本, utils-工具等;
    envs 中包含了`base`，`l2c_2wheel`，`wheel_legged`, `wheel_legged_vmc`, `wheel_legged_vmc_flat`, `y1a_2wheel_vmc`等几个训练任务,训练时通过 `--task=**`指定项目。

    - 一般在安装训练工程之前还需要安装 `rsl_rl`, `rsl_rl` 是实现强化学习算法（PPO）的库。

      - 使用 Git 克隆仓库: `git clone https://github.com/leggedrobotics/sl_rl.git` 。

      - 切换到 v1.0.2 分支: `cd rsl_rl && git checkout v1.0.2` 。

      - 安装： `pip install -e .` 。

    - 本工程中附带了 rsl_rl，所以省略了上一步，直接在项目根目录下，使用命令 `pip install -e .` 安装训练工程：

      - 打开终端, `cd code/gac-robotics`,使用 `pip install -e .` 安装：

        ![install](img/install.png)

5.  安装pinocchio库

    使用`conda install -c conda-forge pinocchio`命令安装,切勿在项目根目录setup.py中添加"pin"命令,可能导致不同库版本冲突,导致前期安装的pytorch等重装,耗时费力;

# 训练及 Sim2Sim

参考[宇树开源机器人强化学习训练工程](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/README_zh.md)

## 训练

以 `l2c_2wheel` 任务为例。

运行命令进行训练： `python wheel_legged_gym/scripts/train.py --task=l2c_2wheel --headless`,`--headless`禁用图形界面渲染。

训练时会自动保存过程模型到 `logs/l2c_2wheel/<date_time>_<run_name>/model_<iteration>.pt` ，同时会将当前的 `legged_robot_config.py` ， `legged_robot.py` 和 `l2c_2wheel_config.py` ， `l2c_2wheel.py` 四个文件保存到同一目录下。

## Play

在 Gym 中查看训练效果，运行命令： `python wheel_legged_gym/scripts/play_l2c.py --task=l2c_2wheel --load_run=Nov29_09-07-54_` ，查看 `Nov29_09-07-54_` 这个训练的效果。
`--load_run` 后面的参数是训练时的文件夹名称，如果不指定，则默认加载最新的训练。

Play 时会导出 Actor 网络和 Policy 网络，保存为 `logs/l2c_2wheel/exported/policies/actor.pt` 和 `logs/l2c_2wheel/exported/policies/policy_1.pt` 。

## Sim2Sim (Mujoco)

方法 1:在仓库的 main 分支中,L2C 包含了可以使用的 rl 模型,将训练好的策略模型替换掉 platforms/L2C/control/reforce_learning/module/l2c_2wheel.pt,即可开始 sim2sim 仿真编译与运行;
方法 2:在 Mujoco 仿真器中运行 Sim2Sim： `python wheel_legged_gym/scripts/sim2sim_l2c.py --load_model=logs/l2c_2wheel/exported/policies/policy_1.pt`
在使用方法二时若报错,那么运行如下命令:

### 1.1 检查显示管理器是否运行

`systemctl status display-manager` # 应该显示 active (running)

### 1.2 检查 X 服务器状态

`ps aux | grep Xorg` # 应该看到 Xorg 进程

### 1.3 运行基本图形测试

`glxgears` # 应该弹出齿轮动画窗口

此时如果报错:`ImportError: /home/mgc/code/gac-robotics/env/lib/libstdc++.so.6: version 'GLIBCXX_3.4.32' not found (required by /usr/lib/python3/dist-packages/apt_pkg.cpython-312-x86_64-linux-gnu.so)`

那么运行如下命令:

#### 1.3.1 更新环境中的 libstdc++

conda install -c conda-forge libstdcxx-ng

#### 1.3.2 验证更新

strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX

#### 应该显示 GLIBCXX_3.4.32

### 1.4 重新运行基本图形测试

`glxgears`, 如果继续报错找不到命令 glxgears,那么运行 `sudo apt install mesa-utils` 安装一下,问题解决.

# 实机部署 (Sim2Real)

sim2sim 可以达到训练预期效果后即可进行实机部署。

一般来说，实机部署是将 `.pt` 模型文件转为 `.jit` 格式，借助 ONNX 进行模型部署。考虑到采用这种方法部署在 arm 版上需要依赖 ONNX-ARM，而 ONNX-ARM 的编译过程比较复杂，因此华工采用了另一种方法：利用 Matlab 将 `.pt` 模型转化为不依赖任何第三方库的 `C/C++` 代码，然后构建为 `libstand_mode_lib.so` 文件，在 RK3588 上运行。

部署相关的代码在 `deploy` 文件夹下。

## 模型转换

将 Play 时生成的 `actor.pt` 拷贝到 `deploy/policies` 文件夹下，在 `deploy/python` 文件夹中运行 `python transfer_weight.py` ，生成 `actor.mat` 。

![transfermodel](img/transfermodel.png)

## 代码生成

进入 `deploy/matlab` 文件夹，使用 **Matlab R2023b** 依次运行 `ActorNet.m` ， `assign_weight.m` 和 `generate_c.m` 。需要安装 `MATLAB Coder Interface for Deep Learning` 组件

- 运行 `ActorNet.m` ，训练一个与轮足机器人强化学习训练时相同网络大小的 policy。
- 运行 `assign_weight.m` 脚本分配权重参数，将 `actor.mat` 中的参数覆盖 policy 中的参数，存为 `save_actor_policy.mat` 文件。
- 已经编写好了 `predictNet.m` 函数，该函数加载 `save_actor_policy.mat` 文件，用于深度学习网络的预测。
- 运行 `generate_c.m` 调用 Codegen 工具箱将 `predictNet.m` 函数生成为不依赖任何深度学习库的 C/C++ 代码。
- 生成的代码位于 `codegen\lib\predictNet ` 文件夹，将所有的 `.h` 和 `.c` 文件（除 `main.c` 外）分别放到待编译项目的 `include` 和 `src` 文件夹下，配置好 `CMakeLists.txt` 文件，进行交叉编译即可生成 `.so` 文件

  ![codegen](img/codegen.png)

## TensorBoard 数据查看

- 在 code/gac-robotics 目录下执行: tensorboard --logdir=logs --port=6007(指定端口) --reload_interval 5(每 5s 刷新)
- 在浏览器打开: http://localhost:6007
