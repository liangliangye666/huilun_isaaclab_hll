# add_subdirectory(publisher)

if(SIM_ENABLE)
  set(SOURCES "glfw_adapter.cc" "glfw_dispatch.cc" "lodepng.cc"
              "platform_ui_adapter.cc" "simulate.cc")

  add_library(libsimulate SHARED ${SOURCES})
  add_library(mujoco::libsimulate ALIAS libsimulate)
  target_include_directories(
    libsimulate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                       ${CMAKE_CURRENT_SOURCE_DIR}/../../tools)
  target_link_libraries(libsimulate PUBLIC mujoco::mujoco glfw)

  install(TARGETS libsimulate DESTINATION lib)
endif()
