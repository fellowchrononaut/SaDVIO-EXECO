# Optional Lite Any Stereo V2 stereo matcher on OpenVINO (doc/dense_las2_stereo.md).
# Needs the OpenVINO C++ runtime (the `openvino` pip wheel ships it: <site-packages>/openvino/cmake); no LAS sources.
#   isaeslam_enable_las2(<target>)  - no-op unless -DISAESLAM_WITH_LAS2=ON
option(ISAESLAM_WITH_LAS2 "Build the Lite Any Stereo V2 (OpenVINO, CPU) dense stereo matcher" OFF)
set(ISAESLAM_OPENVINO_DIR "" CACHE PATH "Directory holding OpenVINOConfig.cmake for ISAESLAM_WITH_LAS2")

function(isaeslam_enable_las2 target)
  if(NOT ISAESLAM_WITH_LAS2)
    return()
  endif()
  find_package(OpenVINO REQUIRED COMPONENTS Runtime HINTS ${ISAESLAM_OPENVINO_DIR})
  get_target_property(ov_lib openvino::runtime IMPORTED_LOCATION_RELEASE)
  get_filename_component(ov_lib_dir "${ov_lib}" DIRECTORY)
  message(STATUS "Lite Any Stereo V2 matcher on: OpenVINO ${OpenVINO_VERSION} (${ov_lib_dir})")
  target_compile_definitions(${target} PRIVATE ISAESLAM_WITH_LAS2)
  target_link_libraries(${target} openvino::runtime)
  # The wheel's libraries (OpenVINO plugins, its own oneTBB) are outside the loader path; keep them found after install
  set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "${ov_lib_dir}")
  set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "${ov_lib_dir}")
endfunction()
