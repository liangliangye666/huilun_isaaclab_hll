#ifndef JOYSTICK_H
#define JOYSTICK_H

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#if PHYSICS_ENABLE
#  include "standMode_types.h"
#endif

#include <SDL2/SDL.h>
#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "math/gac_math.h"
#include "ros_publisher.h"
#include "runtime_flags.h"

#include <glog/logging.h>

class JoyStick {
 public:
  JoyStick();
  ~JoyStick();
#ifdef SIM_ENABLE
  void Update();
#endif
#ifdef PHYSICS_ENABLE
  void Update(standmode_input_t* standmode_input);
#endif
  double forward_or_back() { return forward_or_back_; };
  double left_or_right() { return left_or_right_; };
  bool forward_pad() { return forward_pad_; };
  bool back_pad() { return back_pad_; };
  bool left_pad() { return left_pad_; };
  bool right_pad() { return right_pad_; };
  double up_or_down() { return up_or_down_; };
  double fsm_id() { return fsm_id_; };
  void set_fsm_id(int fsm_id) { fsm_id_ = fsm_id; };

  static constexpr float AXIS_DEAD_ZONE = 0.06f;

 private:
  void gamepadThreadFunc();
  static float applyDeadZone(float raw_value, float dead_zone);
  struct GamepadState {
    std::vector<bool> buttons;
    std::vector<float> axes;
    std::mutex mutex;
  };

  GamepadState gamepad_state_;
  SDL_GameController* controller_ = nullptr;
  std::thread gamepad_thread_;
  bool is_joystick_connected_;

  //
  double forward_or_back_, forward_or_back_last_;
  double left_or_right_, left_or_right_last_;
  bool forward_pad_, back_pad_, left_pad_, right_pad_;
  double up_or_down_;
  int fsm_id_;

  std::shared_ptr<ROS::Publisher> sim_publisher_;
};

#endif