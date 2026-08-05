add_library(fsm SHARED fsm.cc)
add_library(robot_model SHARED robot_model.cc)
add_library(rl rl.cc)
add_library(pd_controller SHARED pd_controller.cc)
add_library(joint_vel_est SHARED joint_vel_est.cc)
add_library(imu SHARED imu.cc)
add_library(kalman_filter SHARED kalman_filter.cc)

target_link_libraries(
    fsm PUBLIC robot_model rl)
target_include_directories(fsm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(SIM_ENABLE)
    message(
        STATUS
        "SIM_ENABLE is ON: Linking Mujoco and enabling simulation features")
    target_link_libraries(
        robot_model
        PUBLIC glog
        Eigen3::Eigen
        pinocchio
        mujoco::mujoco
        simpublisher
        yaml-cpp
        joystick
        joint_vel_est
        imu
        kalman_filter)
    target_compile_definitions(robot_model PUBLIC SIM_ENABLE=1)
elseif(PHYSICS_ENABLE)
    message(STATUS "SIM_ENABLE is OFF: Excluding Mujoco")
    target_link_libraries(
        robot_model
        PUBLIC glog
        Eigen3::Eigen
        pinocchio
        simpublisher
        yaml-cpp
        joystick
        joint_vel_est
        imu
        kalman_filter)
    target_compile_definitions(robot_model PUBLIC PHYSICS_ENABLE=1)
endif()
target_link_libraries(
    rl PUBLIC glog robot_model Eigen3::Eigen yaml-cpp Torch::Torch
    pd_controller)

target_include_directories(
    robot_model
    PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../deploy
    ${CMAKE_SOURCE_DIR}/platforms/l5a/control)
target_include_directories(rl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS})

target_link_libraries(pd_controller PUBLIC Eigen3::Eigen)
target_include_directories(pd_controller PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(joint_vel_est PUBLIC Eigen3::Eigen)
target_include_directories(joint_vel_est PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(imu PUBLIC kalman_filter simpublisher)
target_include_directories(imu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(kalman_filter PUBLIC Eigen3::Eigen glog)
target_include_directories(imu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR})

install(TARGETS robot_model DESTINATION lib)
install(TARGETS fsm DESTINATION lib)
install(TARGETS rl pd_controller joint_vel_est imu kalman_filter DESTINATION lib)

set(BASE_DIR ${CMAKE_SOURCE_DIR})
file(RELATIVE_PATH RELATIVE_PATH_TO_BASE ${BASE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if(SIM_ENABLE)
    install(FILES rl_parameters_sim.yaml DESTINATION ${RELATIVE_PATH_TO_BASE})
elseif(PHYSICS_ENABLE)
    install(FILES rl_parameters_physics.yaml
        DESTINATION ${RELATIVE_PATH_TO_BASE}
    )
endif()
install(DIRECTORY module/
    DESTINATION ${RELATIVE_PATH_TO_BASE}/module
)
