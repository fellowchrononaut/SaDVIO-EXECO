# Optional learned place recognition for the loop closure (loop_detector "learned"; doc/loop_closure).
# A global image descriptor (e.g. MegaLoc exported to ONNX by doc/loop_closure/tools/export_megaloc_onnx.py) runs on
#   - the CPU through OpenVINO:  -DISAESLAM_WITH_VPR_OPENVINO=ON -DISAESLAM_OPENVINO_DIR=<dir with OpenVINOConfig.cmake>
#   - the GPU through TensorRT:  -DISAESLAM_WITH_VPR_TENSORRT=ON (ISAESLAM_TENSORRT_ROOT / ISAESLAM_CUDA_ROOT as for FFS)
#   isaeslam_enable_vpr(<target>)  - no-op unless one of the options is ON
option(ISAESLAM_WITH_VPR_OPENVINO "Learned loop detection on the CPU (OpenVINO)" OFF)
option(ISAESLAM_WITH_VPR_TENSORRT "Learned loop detection on the GPU (TensorRT)" OFF)
set(ISAESLAM_OPENVINO_DIR "" CACHE PATH "Directory holding OpenVINOConfig.cmake")
set(ISAESLAM_CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA runtime root")
set(ISAESLAM_TENSORRT_ROOT "/usr" CACHE PATH "TensorRT root")

function(isaeslam_enable_vpr target)
  if(ISAESLAM_WITH_VPR_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime HINTS ${ISAESLAM_OPENVINO_DIR})
    get_target_property(ov_lib openvino::runtime IMPORTED_LOCATION_RELEASE)
    get_filename_component(ov_lib_dir "${ov_lib}" DIRECTORY)
    message(STATUS "Learned loop detection on the CPU: OpenVINO ${OpenVINO_VERSION} (${ov_lib_dir})")
    target_compile_definitions(${target} PRIVATE ISAESLAM_VPR_OPENVINO)
    target_link_libraries(${target} openvino::runtime)
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "${ov_lib_dir}")
    set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "${ov_lib_dir}")
  endif()
  if(ISAESLAM_WITH_VPR_TENSORRT)
    find_path(ISAESLAM_VPR_NVINFER_INCLUDE NvInfer.h
      HINTS ${ISAESLAM_TENSORRT_ROOT} PATH_SUFFIXES include include/x86_64-linux-gnu)
    find_library(ISAESLAM_VPR_NVINFER_LIB nvinfer
      HINTS ${ISAESLAM_TENSORRT_ROOT} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
    find_path(ISAESLAM_VPR_CUDART_INCLUDE cuda_runtime_api.h
      HINTS ${ISAESLAM_CUDA_ROOT} PATH_SUFFIXES include targets/x86_64-linux/include)
    find_library(ISAESLAM_VPR_CUDART_LIB cudart
      HINTS ${ISAESLAM_CUDA_ROOT} PATH_SUFFIXES lib64 targets/x86_64-linux/lib)
    foreach(v ISAESLAM_VPR_NVINFER_INCLUDE ISAESLAM_VPR_NVINFER_LIB ISAESLAM_VPR_CUDART_INCLUDE ISAESLAM_VPR_CUDART_LIB)
      if(NOT ${v})
        message(FATAL_ERROR "ISAESLAM_WITH_VPR_TENSORRT: ${v} not found (set ISAESLAM_TENSORRT_ROOT / ISAESLAM_CUDA_ROOT)")
      endif()
    endforeach()
    message(STATUS "Learned loop detection on the GPU: ${ISAESLAM_VPR_NVINFER_LIB}")
    target_compile_definitions(${target} PRIVATE ISAESLAM_VPR_TENSORRT)
    target_include_directories(${target} SYSTEM PRIVATE ${ISAESLAM_VPR_NVINFER_INCLUDE} ${ISAESLAM_VPR_CUDART_INCLUDE})
    target_link_libraries(${target} ${ISAESLAM_VPR_NVINFER_LIB} ${ISAESLAM_VPR_CUDART_LIB})
  endif()
endfunction()
