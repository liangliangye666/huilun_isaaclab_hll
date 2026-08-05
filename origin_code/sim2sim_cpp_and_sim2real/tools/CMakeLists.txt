add_library(joystick joystick.cc)

target_link_libraries(joystick PUBLIC SDL2 pthread glog simpublisher
                                      gac_math)

target_include_directories(
    joystick PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                    )

if(SIM_ENABLE)
    target_compile_definitions(joystick PUBLIC SIM_ENABLE=1)

    add_executable(joystick_test joystick_test.cc)
    target_link_libraries(joystick_test PUBLIC joystick)
    install(TARGETS joystick_test DESTINATION bin)
elseif(PHYSICS_ENABLE)
    target_compile_definitions(joystick PUBLIC PHYSICS_ENABLE=1)
    target_include_directories(joystick PUBLIC ${PROJECT_SOURCE_DIR}/platforms/y4a/deploy)
endif()

target_include_directories(joystick PUBLIC ${CMAKE_SOURCE_DIR})

install(TARGETS joystick DESTINATION lib)
