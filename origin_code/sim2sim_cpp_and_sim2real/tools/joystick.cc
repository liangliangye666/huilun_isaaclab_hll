// JoyStick.cc
#include "joystick.h"
#include <csignal>
#include <iostream>

JoyStick::JoyStick() {
#if SIM_ENABLE
  is_joystick_connected_ = true;
  if (SDL_Init(SDL_INIT_GAMECONTROLLER) < 0) {
    std::cerr << "SDL初始化失败: " << SDL_GetError() << std::endl;
    return;
  }

  if (SDL_NumJoysticks() < 1) {
    std::cerr << "未检测到仿真手柄！" << std::endl;
    is_joystick_connected_ = false;
    SDL_Quit();
    // return;
  }

  controller_ = SDL_GameControllerOpen(0);
  if (!controller_) {
    std::cerr << "无法打开仿真手柄: " << SDL_GetError() << std::endl;
    is_joystick_connected_ = false;
    SDL_Quit();
    // return;
  }

  {
    std::lock_guard<std::mutex> lock(gamepad_state_.mutex);
    gamepad_state_.buttons.resize(SDL_CONTROLLER_BUTTON_MAX, false);
    gamepad_state_.axes.resize(SDL_CONTROLLER_AXIS_MAX, 0.0f);
  }
  if (is_joystick_connected_) {
    gamepad_thread_ = std::thread(&JoyStick::gamepadThreadFunc, this);
  }

#endif

  forward_or_back_ = 0;
  forward_or_back_last_ = 0;
  left_or_right_ = 0;
  left_or_right_last_ = 0;
  up_or_down_ = 0;
  fsm_id_ = 0;

  sim_publisher_ = ROS::Publisher::getInstance();
}

JoyStick::~JoyStick() {
  RuntimeFlags::instance().stop();
  if (gamepad_thread_.joinable()) {
    gamepad_thread_.join();
  }
  if (controller_) {
    SDL_GameControllerClose(controller_);
  }
  SDL_Quit();
}

float JoyStick::applyDeadZone(float raw_value, float dead_zone) {
  if (std::abs(raw_value) < dead_zone) {
    return 0.0f;
  }
  return (raw_value - dead_zone * std::copysign(1.0f, raw_value)) /
         (1.0f - dead_zone);
}

void JoyStick::gamepadThreadFunc() {
  SDL_Event event;
  while (RuntimeFlags::instance().is_running()) {
    while (SDL_PollEvent(&event)) {
      std::lock_guard<std::mutex> lock(gamepad_state_.mutex);

      if (event.type == SDL_CONTROLLERBUTTONDOWN ||
          event.type == SDL_CONTROLLERBUTTONUP) {
        if (event.cbutton.button < gamepad_state_.buttons.size()) {
          gamepad_state_.buttons[event.cbutton.button] =
              (event.type == SDL_CONTROLLERBUTTONDOWN);
        }
      } else if (event.type == SDL_CONTROLLERAXISMOTION) {
        if (event.caxis.axis < gamepad_state_.axes.size()) {
          float raw_value = event.caxis.value / 32767.0f;
          gamepad_state_.axes[event.caxis.axis] =
              applyDeadZone(raw_value, AXIS_DEAD_ZONE);
        }
      }
    }
    SDL_Delay(10);
  }
}

#ifdef SIM_ENABLE
void JoyStick::Update() {
  std::lock_guard<std::mutex> lock(gamepad_state_.mutex);
  forward_or_back_ = -gamepad_state_.axes[1];
  if (gamepad_state_.buttons[9]) {
    up_or_down_ = -gamepad_state_.axes[3];
  } else {
    left_or_right_ = gamepad_state_.axes[2];
  }
  forward_or_back_ =
      GacMath::Ramp(forward_or_back_, forward_or_back_last_, 1, 0.002);
  left_or_right_ = GacMath::Ramp(left_or_right_, left_or_right_last_, 1, 0.002);

  forward_pad_ = gamepad_state_.buttons[11];
  back_pad_ = gamepad_state_.buttons[12];
  left_pad_ = gamepad_state_.buttons[13];
  right_pad_ = gamepad_state_.buttons[14];

  if (gamepad_state_.buttons[5]) {
    fsm_id_ = 0;
  } else if (gamepad_state_.buttons[9] && gamepad_state_.buttons[11]) {
    fsm_id_ = 1;
  } else if (gamepad_state_.buttons[9] && gamepad_state_.buttons[12]) {
    fsm_id_ = 2;
  }

  forward_or_back_last_ = forward_or_back_;
  left_or_right_last_ = left_or_right_;

  sim_publisher_->Publish("joystick/forward_or_back", forward_or_back_);
  sim_publisher_->Publish("joystick/left_or_right", left_or_right_);
  sim_publisher_->Publish("joystick/forward_pad",
                          static_cast<double>(forward_pad_));
  sim_publisher_->Publish("joystick/back_pad", static_cast<double>(back_pad_));
  sim_publisher_->Publish("joystick/left_pad", static_cast<double>(left_pad_));
  sim_publisher_->Publish("joystick/right_pad",
                          static_cast<double>(right_pad_));
  sim_publisher_->Publish("joystick/fsm_id_", fsm_id_);
}
#endif
#if PHYSICS_ENABLE
void JoyStick::Update(standmode_input_t* standmode_input) {
  std::lock_guard<std::mutex> lock(gamepad_state_.mutex);
  forward_or_back_ = (standmode_input->Handle_signals.a1 - 0.5) * -2;
  left_or_right_ = (standmode_input->Handle_signals.a3 - 0.5) * 2;
  forward_or_back_ =
      GacMath::Ramp(forward_or_back_, forward_or_back_last_, 0.5, 0.005);
  left_or_right_ =
      GacMath::Ramp(left_or_right_, left_or_right_last_, 0.5, 0.005);

  forward_pad_ = standmode_input->Handle_signals.a7 < 0.1;
  back_pad_ = standmode_input->Handle_signals.a7 > 0.9;
  left_pad_ = standmode_input->Handle_signals.a6 < 0.1;
  right_pad_ = standmode_input->Handle_signals.a6 > 0.9;

  forward_or_back_last_ = forward_or_back_;
  left_or_right_last_ = left_or_right_;

  sim_publisher_->Publish("joystick/forward_or_back", forward_or_back_);
  sim_publisher_->Publish("joystick/left_or_right", left_or_right_);
  sim_publisher_->Publish("joystick/forward_pad",
                          static_cast<double>(forward_pad_));
  sim_publisher_->Publish("joystick/back_pad", static_cast<double>(back_pad_));
  sim_publisher_->Publish("joystick/left_pad", static_cast<double>(left_pad_));
  sim_publisher_->Publish("joystick/right_pad",
                          static_cast<double>(right_pad_));
  sim_publisher_->Publish("joystick/fsm_id_", fsm_id_);
}

#endif
