#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "vnoid_lib" for configuration ""
set_property(TARGET vnoid_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(vnoid_lib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "mujoco::mujoco;Eigen3::Eigen"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libvnoid_lib.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS vnoid_lib )
list(APPEND _IMPORT_CHECK_FILES_FOR_vnoid_lib "${_IMPORT_PREFIX}/lib/libvnoid_lib.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
