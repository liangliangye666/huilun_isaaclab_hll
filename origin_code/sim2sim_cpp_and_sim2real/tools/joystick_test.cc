#include <SDL2/SDL.h>
#include <atomic>
#include <csignal>
#include <iostream>
#include <mutex>
#include <vector>

constexpr float AXIS_DEAD_ZONE = 0.06f;

struct GamepadState {
  std::vector<bool> buttons;
  std::vector<float> axes;
  std::mutex mutex;
};

GamepadState gamepad_state;
std::atomic<bool> running{true};

float applyDeadZone(float raw_value, float dead_zone) {
  if (std::abs(raw_value) < dead_zone) return 0.0f;
  return (raw_value - dead_zone * std::copysign(1.0f, raw_value)) / (1.0f - dead_zone);
}

void signalHandler(int signal) { running = false; }

int main() {
  std::signal(SIGINT, signalHandler);

  if (SDL_Init(SDL_INIT_GAMECONTROLLER | SDL_INIT_VIDEO) < 0) {
    std::cerr << "SDL 初始化失败: " << SDL_GetError() << std::endl;
    return 1;
  }

  if (SDL_NumJoysticks() < 1) {
    std::cerr << "未检测到手柄！" << std::endl;
    SDL_Quit();
    return 1;
  }

  SDL_GameController* controller = SDL_GameControllerOpen(0);
  if (!controller) {
    std::cerr << "无法打开手柄: " << SDL_GetError() << std::endl;
    SDL_Quit();
    return 1;
  }

  {
    std::lock_guard<std::mutex> lock(gamepad_state.mutex);
    gamepad_state.buttons.resize(SDL_CONTROLLER_BUTTON_MAX, false);
    gamepad_state.axes.resize(SDL_CONTROLLER_AXIS_MAX, 0.0f);
  }

  while (running) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      std::lock_guard<std::mutex> lock(gamepad_state.mutex);

      if (event.type == SDL_QUIT) {
        running = false;
      }

      if (event.type == SDL_CONTROLLERBUTTONDOWN || event.type == SDL_CONTROLLERBUTTONUP) {
        if (event.cbutton.button < gamepad_state.buttons.size()) {
          gamepad_state.buttons[event.cbutton.button] = (event.type == SDL_CONTROLLERBUTTONDOWN);
        }
      } else if (event.type == SDL_CONTROLLERAXISMOTION) {
        if (event.caxis.axis < gamepad_state.axes.size()) {
          float raw_value = event.caxis.value / 32767.0f;
          gamepad_state.axes[event.caxis.axis] = applyDeadZone(raw_value, AXIS_DEAD_ZONE);
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(gamepad_state.mutex);

      // 打印按钮状态
      for (size_t i = 0; i < gamepad_state.buttons.size(); ++i) {
        if (gamepad_state.buttons[i]) {
          std::cout << "按钮 " << i << " 被按下" << std::endl;
        }
      }

      // 打印轴状态
      for (size_t i = 0; i < gamepad_state.axes.size(); ++i) {
        if (std::abs(gamepad_state.axes[i]) > AXIS_DEAD_ZONE) {
          std::cout << "轴 " << i << ": " << gamepad_state.axes[i] << std::endl;
        }
      }
    }

    SDL_Delay(10);
  }

  SDL_GameControllerClose(controller);
  SDL_Quit();
  return 0;
}