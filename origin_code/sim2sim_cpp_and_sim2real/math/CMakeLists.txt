set(SOURCE math_utlis.cc trajectory_interpolation.cc)
add_library(gac_math SHARED ${SOURCE})
target_link_libraries(gac_math PUBLIC Eigen3::Eigen glog)
target_include_directories(gac_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS gac_math DESTINATION lib)

add_executable(math_test math_test.cc)
target_link_libraries(math_test PUBLIC gac_math glog)

install(TARGETS math_test DESTINATION bin)
