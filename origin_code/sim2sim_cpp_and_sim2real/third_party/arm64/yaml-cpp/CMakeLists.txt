add_library(yaml-cpp SHARED IMPORTED GLOBAL)
set_target_properties(
  yaml-cpp
  PROPERTIES IMPORTED_LOCATION
             ${CMAKE_CURRENT_SOURCE_DIR}/lib/libyaml-cpp.so.0.8.0
             INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libyaml-cpp.so.0.8.0
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libyaml-cpp.so
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libyaml-cpp.so.0.8 DESTINATION lib)
