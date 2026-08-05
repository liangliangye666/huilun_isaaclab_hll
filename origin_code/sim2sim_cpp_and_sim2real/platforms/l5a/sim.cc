// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#undef NDEBUG

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <fstream>

#include <mujoco/mujoco.h>
#include <unistd.h>
#include "array_safety.h"
#include "glfw_adapter.h"

#include "fsm.h"
#include "simulate.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
#  include <windows.h>
#else
#  if defined(__APPLE__)
#    include <mach-o/dyld.h>
#  endif
#  include <sys/errno.h>
#  include <unistd.h>
#endif
}
std::ofstream logfile;
std::string urdf_path = "/sim/model/l5a/urdf/l5aurdf20260521.urdf";
std::string xml_path = "/sim/model/l5a/xml/l5aurdf20260521.xml";

void MyController(const mjModel* m, mjData* d) {
  static l5a::RobotModel robot_model(urdf_path);
  robot_model.UpdateMujocoJointStates(m, d);
  robot_model.UpdateModel();
  static l5a::FSM fsm(robot_model);
  robot_model.vel_x_des_ = 0.5;
  robot_model.vel_y_des_ = -0.0;
  robot_model.omega_des_ = 0.02;
  robot_model.gait_enable_ = false;
  static double count_num = 0;
  static double count_num_plus = 0;
  if(count_num > 1000 && count_num <= 6000){
    robot_model.vel_x_des_ = 0.5;
    // robot_model.omega_des_ = 0.5;
    // robot_model.vel_y_des_ = 0.5;
    robot_model.gait_enable_ = true;
    robot_model.phase_ = std::fmod(count_num_plus * 0.002, 0.8) / 0.8;
    count_num_plus++;
  }else if(count_num > 6000 && count_num <= 9000){
    robot_model.gait_enable_ = true;
    robot_model.phase_ = std::fmod(count_num_plus * 0.002, 0.8) / 0.8;
    count_num_plus++;
    // count_num_plus = 0;
  }else if(count_num > 9000 && count_num <= 14000){
    // robot_model.vel_x_des_ = -0.5;
    // robot_model.omega_des_ = -0.5;
    // robot_model.vel_y_des_ = -0.5;
    robot_model.gait_enable_ = true;
    robot_model.phase_ = std::fmod(count_num_plus * 0.002, 0.8) / 0.8;
    count_num_plus++;
  }
  count_num += 1;
 

  fsm.Run(robot_model);
  Eigen::VectorXd pos_cmd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6); // 初始化力矩命令向量
  Eigen::VectorXd tau_cmd = Eigen::VectorXd::Zero(robot_model.pino_model().nv-6); // 初始化力矩命令向量
  tau_cmd = fsm.tau();
  pos_cmd = fsm.pos();
  // std::cout << "tau_cmd:" << tau_cmd << std::endl;

  // test PD paras for RL training
  // Eigen::VectorXd default_pos(8);
  // default_pos << 0.0872665, 0.261799, -0.510893, 0, 0.0872665, 0.261799, -0.510893, 0;
  // Eigen::VectorXd default_vel = Eigen::VectorXd::Zero(8);
  // Eigen::VectorXd actual_pos = robot_model.q_fixed;
  // Eigen::VectorXd actual_vel = robot_model.qdot.segment(6, robot_model.pino_model().nv-6);
  // Eigen::VectorXd Kp(8);
  // Eigen::VectorXd Ki(8);
  // Eigen::VectorXd Kd(8);
  // Kp = fsm.pos_fb_kp_;
  // Ki << 0, 0, 0, 1, 0, 0, 0, 1;
  // Kd = fsm.pos_fb_kd_;
  // Eigen::VectorXd pos_error = default_pos - actual_pos;
  // Eigen::VectorXd vel_error = default_vel - actual_vel;
  // Eigen::VectorXd tau_pure_ff = Kp.cwiseProduct(pos_error) + Kd.cwiseProduct(vel_error);

  for (size_t i = 0; i < robot_model.pino_model().nv-6; i++) {
    d->ctrl[i] = tau_cmd[i];
    // d->ctrl[i] = tau_pure_ff[i];    // for rl pd paras selection
  }

  int left_id  = mj_name2id(m, mjOBJ_SITE, "wheel_left_site");
  int right_id = mj_name2id(m, mjOBJ_SITE, "wheel_right_site");
  double z_left  = d->site_xpos[3*left_id + 2] - 0.127;
  double z_right = d->site_xpos[3*right_id + 2] - 0.127;
  // std::cout << "z_left:" << z_left << ",z_right:" << z_right << std::endl;


  // logfile << "t, vel_des, vel_act, w_des, w_act, zero, roll, pitch, height_target, height\n";
  logfile << d->time << "," << robot_model.vel_x_des_ << "," << robot_model.qdot[0] << "," << robot_model.vel_y_des_ << "," << robot_model.qdot[1] << "," 
  << robot_model.omega_des_ << "," << robot_model.qdot[5] << "," << 0 << "," << robot_model.q_rpy[3] << "," 
  << robot_model.q_rpy[4] << "," << 0.643 << "," << robot_model.q_rpy[2] << "," << z_left << "," << z_right << "\n";

}

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

// constants
const double syncMisalign = 0.1;          // maximum mis-alignment before re-sync (simulation seconds)
const double simRefreshFraction = 0.007;  // fraction of refresh available for simulation
const int kErrorLength = 1024;            // load error string length

// model and data
mjModel* m = nullptr;
mjData* d = nullptr;

using Seconds = std::chrono::duration<double>;

//---------------------------------------- plugin handling
//-----------------------------------------

// return the path to the directory containing the current executable
// used to determine the location of auto-loaded plugin libraries
std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        // realpath is too small, grow and retry
        buf_size *= 2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
        return "";
      }
    }
    return realpath.get();
  }();
#else
  constexpr char kPathSep = '/';
#  if defined(__APPLE__)
  std::unique_ptr<char[]> buf(nullptr);
  {
    std::uint32_t buf_size = 0;
    _NSGetExecutablePath(nullptr, &buf_size);
    buf.reset(new char[buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &buf_size)) {
      std::cerr << "unexpected error from _NSGetExecutablePath\n";
    }
  }
  const char* path = buf.get();
#  else
  const char* path = "/proc/self/exe";
#  endif
  std::string realpath = [&]() -> std::string {
    std::unique_ptr<char[]> realpath(nullptr);
    std::uint32_t buf_size = 128;
    bool success = false;
    while (!success) {
      realpath.reset(new (std::nothrow) char[buf_size]);
      if (!realpath) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      std::size_t written = readlink(path, realpath.get(), buf_size);
      if (written < buf_size) {
        realpath.get()[written] = '\0';
        success = true;
      } else if (written == -1) {
        if (errno == EINVAL) {
          // path is already not a symlink, just use it
          return path;
        }

        std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
        return "";
      } else {
        // realpath is too small, grow and retry
        buf_size *= 2;
      }
    }
    return realpath.get();
  }();
#endif

  if (realpath.empty()) {
    return "";
  }

  for (std::size_t i = realpath.size() - 1; i > 0; --i) {
    if (realpath.c_str()[i] == kPathSep) {
      return realpath.substr(0, i);
    }
  }

  // don't scan through the entire file system's root
  return "";
}

// scan for libraries in the plugin directory to load additional plugins
void scanPluginLibraries() {
  // check and print plugins that are linked directly into the executable
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

  // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
  const std::string sep = "\\";
#else
  const std::string sep = "/";
#endif

  // try to open the ${EXECDIR}/MUJOCO_PLUGIN_DIR directory
  // ${EXECDIR} is the directory containing the simulate binary itself
  // MUJOCO_PLUGIN_DIR is the MUJOCO_PLUGIN_DIR preprocessor macro
  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    return;
  }

  const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char* filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}

//------------------------------------------- simulation
//-------------------------------------------

const char* Diverged(int disableflags, const mjData* d) {
  if (disableflags & mjDSBL_AUTORESET) {
    for (mjtWarning w : {mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS}) {
      if (d->warning[w].number > 0) {
        return mju_warningText(w, d->warning[w].lastinfo);
      }
    }
  }
  return nullptr;
}

mjModel* LoadModel(const char* file, mj::Simulate& sim) {
  // this copy is needed so that the mju::strlen call below compiles
  char filename[mj::Simulate::kMaxFilenameLength];
  mju::strcpy_arr(filename, file);

  // make sure filename is not empty
  if (!filename[0]) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel* mnew = 0;
  auto load_start = mj::Simulate::Clock::now();
  if (mju::strlen_arr(filename) > 4 &&
      !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb", mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4)) {
    mnew = mj_loadModel(filename, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);

    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }
  auto load_interval = mj::Simulate::Clock::now() - load_start;
  double load_seconds = Seconds(load_interval).count();

  // if no error and load took more than 1/4 seconds, report load time
  if (!loadError[0] && load_seconds > 0.25) {
    mju::sprintf_arr(loadError, "Model loaded in %.2g seconds", load_seconds);
  }

  mju::strcpy_arr(sim.load_error, loadError);

  if (!mnew) {
    std::printf("%s\n", loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
    sim.run = 0;
  }

  return mnew;
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& sim) {
  // cpu-sim syncronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;
  mju_copy(d->qpos, m->key_qpos, m->nq * 1);

  // run until asked to exit
  while (!sim.exitrequest.load() && RuntimeFlags::instance().is_running()) {
    if (sim.droploadrequest.load()) {
      sim.LoadMessage(sim.dropfilename);
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.dropfilename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;

        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    if (sim.uiloadrequest.load()) {
      sim.uiloadrequest.fetch_sub(1);
      sim.LoadMessage(sim.filename);
      mjModel* mnew = LoadModel(sim.filename, sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.Load(mnew, dnew, sim.filename);

        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        mj_deleteData(d);
        mj_deleteModel(m);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

      } else {
        sim.LoadMessageClear();
      }
    }

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery
    //  life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

      // run only if model is present
      if (m) {
        // running
        if (sim.run) {
          bool stepped = false;

          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger
          // than syncmisalign
          bool misaligned = std::abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 || misaligned || sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // run single step, let next iteration deal with timing
            mj_step(m, d);
            MyController(m, d);
            const char* message = Diverged(m->opt.disableflags, d);
            if (message) {
              sim.run = 0;
              mju::strcpy_arr(sim.load_error, message);
            } else {
              stepped = true;
            }
          }

          // in-sync: step until ahead of cpu
          else {
            bool measured = false;
            mjtNum prevSim = d->time;

            double refreshTime = simRefreshFraction / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown = std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              // inject noise
              sim.InjectNoise();

              // call mj_step
              mj_step(m, d);
              MyController(m, d);
              const char* message = Diverged(m->opt.disableflags, d);
              if (message) {
                sim.run = 0;
                mju::strcpy_arr(sim.load_error, message);
              } else {
                stepped = true;
              }

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }

          // save current state to history buffer
          if (stepped) {
            sim.AddToHistory();
          }
        }

        // paused
        else {
          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          sim.speed_changed = true;
        }
      }
    }  // release std::lock_guard<std::mutex>
  }
}
}  // namespace

//-------------------------------------- physics_thread
//--------------------------------------------

void PhysicsThread(mj::Simulate* sim, const char* filename) {
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr) {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m) {
      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      d = mj_makeData(m);
    }
    if (d) {
      sim->Load(m, d, filename);

      // lock the sim mutex
      const std::unique_lock<std::recursive_mutex> lock(sim->mtx);

      mj_forward(m, d);

    } else {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  mj_deleteData(d);
  mj_deleteModel(m);
}

//------------------------------------------ main
//--------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running
// under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char* msg) { rosetta_error_msg = msg; }
#endif

void MujocoSimulator(int argc, char** argv) {
  // 初始化 ROS2
  rclcpp::init(argc, argv);
  auto message_handle = ROS::Publisher::getInstance();

  // 创建多线程执行器（默认线程数 = CPU核心数）
  auto executor = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  executor->add_node(message_handle);  // 将节点添加到执行器

  // 定义执行器的异步运行函数
  auto spin_func = [](std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_ptr) {
    executor_ptr->spin();  // 启动多线程处理回调
    LOG(INFO) << "ROS2 MultiThreadedExecutor exited";
  };

  // 在独立线程中运行执行器
  auto spin_thread = std::thread(spin_func, executor);

  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  static auto sim = std::make_unique<mj::Simulate>(std::make_unique<mj::GlfwAdapter>(), &cam, &opt, &pert, /* is_passive = */ false);

  std::string workspacePath = std::getenv("PROJECT_ROOT_DIR");
  std::string path = workspacePath + xml_path;
  const char* filename = path.c_str();
  LOG(INFO) << "FILE_NAME: " << filename;
  if (argc > 1) {
    filename = argv[1];
  }

  // start physics thread
  sim->run = 0;
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename);

  // start simulation UI loop (blocking call)
  sim->RenderLoop();
  if (!RuntimeFlags::instance().is_running()) {
    rclcpp::shutdown();
    LOG(INFO) << "Shutting down ROS success";
  }

  executor->cancel();
  spin_thread.join();
  physicsthreadhandle.join();
  rclcpp::shutdown();
  message_handle = nullptr;
}

void exit_handler(int sig) {
  LOG(INFO) << "Exit handler success";
  RuntimeFlags::instance().stop();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging("l4a");
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;

  signal(SIGINT, exit_handler);
  signal(SIGTERM, exit_handler);
  logfile.open("log.csv");
  logfile << "t, vel_x_des, vel_x_act, vel_y_des, vel_y_act, w_des, w_act, zero, roll, pitch, height_target, height, z_left, z_right\n";
  MujocoSimulator(argc, argv);

  return 0;
}
