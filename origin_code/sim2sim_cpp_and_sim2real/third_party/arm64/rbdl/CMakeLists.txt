add_library(rbdl SHARED IMPORTED GLOBAL)
set_target_properties(
  rbdl
  PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/librbdl.so.3.3.1"
    INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/include/rbdl/addons/urdfreader"
)

add_library(rbdl_urdfreader SHARED IMPORTED GLOBAL)
set_target_properties(
  rbdl_urdfreader
  PROPERTIES
    IMPORTED_LOCATION
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/librbdl_urdfreader.so.3.3.1"
    INTERFACE_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/include/rbdl/addons/urdfreader"
)

# target_link_libraries(rbdl INTERFACE rbdl_urdfreader)
target_link_libraries(rbdl_urdfreader INTERFACE rbdl)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/librbdl.so.3.3.1
              ${CMAKE_CURRENT_SOURCE_DIR}/lib/librbdl_urdfreader.so.3.3.1
        DESTINATION lib)
