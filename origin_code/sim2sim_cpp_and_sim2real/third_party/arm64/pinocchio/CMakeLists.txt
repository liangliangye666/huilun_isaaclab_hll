add_library(pinocchio SHARED IMPORTED GLOBAL)
set_target_properties(
  pinocchio
  PROPERTIES IMPORTED_LOCATION
             ${CMAKE_CURRENT_SOURCE_DIR}/lib/libpinocchio_default.so.3.3.0
             IMPORTED_LINK_INTERFACE_LIBRARIES
             ${CMAKE_CURRENT_SOURCE_DIR}/lib/libpinocchio_parsers.so.3.3.0
             INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include
             INTERFACE_LINK_OPTIONS "LINKER:-rpath-link=${ROS_FOXY_LIB_DIR}")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libpinocchio_default.so.3.3.0
              ${CMAKE_CURRENT_SOURCE_DIR}/lib/libpinocchio_parsers.so.3.3.0
        DESTINATION lib)
