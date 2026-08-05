#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pinocchio::pinocchio_default" for configuration "Release"
set_property(TARGET pinocchio::pinocchio_default APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pinocchio::pinocchio_default PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpinocchio_default.so.3.3.0"
  IMPORTED_SONAME_RELEASE "libpinocchio_default.so.3.3.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS pinocchio::pinocchio_default )
list(APPEND _IMPORT_CHECK_FILES_FOR_pinocchio::pinocchio_default "${_IMPORT_PREFIX}/lib/libpinocchio_default.so.3.3.0" )

# Import target "pinocchio::pinocchio_parsers" for configuration "Release"
set_property(TARGET pinocchio::pinocchio_parsers APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pinocchio::pinocchio_parsers PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libpinocchio_parsers.so.3.3.0"
  IMPORTED_SONAME_RELEASE "libpinocchio_parsers.so.3.3.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS pinocchio::pinocchio_parsers )
list(APPEND _IMPORT_CHECK_FILES_FOR_pinocchio::pinocchio_parsers "${_IMPORT_PREFIX}/lib/libpinocchio_parsers.so.3.3.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
