add_library(glog SHARED IMPORTED GLOBAL)
set_target_properties(
  glog
  PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/libglog.so.0.7.1
             INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(glog INTERFACE GLOG_USE_GLOG_EXPORT=1)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libglog.so.0.7.1
              ${CMAKE_CURRENT_SOURCE_DIR}/lib/libglog.so.2 DESTINATION lib)
