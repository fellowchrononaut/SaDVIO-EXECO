# Optional Fast-FoundationStereo stereo matcher (doc/dense_ffs_stereo.md).
# Needs the TensorRT runtime + headers and the CUDA runtime; no nvcc and no Fast-FoundationStereo sources.
#   isaeslam_enable_ffs(<target>)  - no-op unless -DISAESLAM_WITH_FFS=ON
option(ISAESLAM_WITH_FFS "Build the Fast-FoundationStereo (TensorRT) dense stereo matcher" OFF)
set(ISAESLAM_CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA runtime root for ISAESLAM_WITH_FFS")
set(ISAESLAM_TENSORRT_ROOT "/usr" CACHE PATH "TensorRT root for ISAESLAM_WITH_FFS")

function(isaeslam_enable_ffs target)
  if(NOT ISAESLAM_WITH_FFS)
    return()
  endif()
  find_path(ISAESLAM_NVINFER_INCLUDE NvInfer.h
    HINTS ${ISAESLAM_TENSORRT_ROOT} PATH_SUFFIXES include include/x86_64-linux-gnu)
  find_library(ISAESLAM_NVINFER_LIB nvinfer
    HINTS ${ISAESLAM_TENSORRT_ROOT} PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
  find_path(ISAESLAM_CUDART_INCLUDE cuda_runtime_api.h
    HINTS ${ISAESLAM_CUDA_ROOT} PATH_SUFFIXES include targets/x86_64-linux/include)
  find_library(ISAESLAM_CUDART_LIB cudart
    HINTS ${ISAESLAM_CUDA_ROOT} PATH_SUFFIXES lib64 targets/x86_64-linux/lib)
  foreach(v ISAESLAM_NVINFER_INCLUDE ISAESLAM_NVINFER_LIB ISAESLAM_CUDART_INCLUDE ISAESLAM_CUDART_LIB)
    if(NOT ${v})
      message(FATAL_ERROR "ISAESLAM_WITH_FFS: ${v} not found (set ISAESLAM_TENSORRT_ROOT / ISAESLAM_CUDA_ROOT)")
    endif()
  endforeach()
  message(STATUS "Fast-FoundationStereo matcher on: ${ISAESLAM_NVINFER_LIB}, ${ISAESLAM_CUDART_LIB}")
  target_compile_definitions(${target} PRIVATE ISAESLAM_WITH_FFS)
  target_include_directories(${target} SYSTEM PRIVATE ${ISAESLAM_NVINFER_INCLUDE} ${ISAESLAM_CUDART_INCLUDE})
  target_link_libraries(${target} ${ISAESLAM_NVINFER_LIB} ${ISAESLAM_CUDART_LIB})
endfunction()
