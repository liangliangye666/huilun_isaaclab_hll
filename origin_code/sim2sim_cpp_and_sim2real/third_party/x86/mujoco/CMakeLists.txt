add_library(mujoco SHARED IMPORTED GLOBAL)
set_target_properties(
  mujoco
  PROPERTIES IMPORTED_LOCATION
             ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmujoco.so.3.2.2
             INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(mujoco::mujoco ALIAS mujoco)

# target_include_directories(mujoco W ${CMAKE_CURRENT_SOURCE_DIR}/include)
# target_include_directories(mujoco INTERFACE
# ${CMAKE_CURRENT_SOURCE_DIR}/simulate)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libmujoco.so.3.2.2
        DESTINATION lib)
