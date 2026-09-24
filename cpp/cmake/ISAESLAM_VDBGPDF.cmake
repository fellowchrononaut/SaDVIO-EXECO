# Optional VDB-GPDF dense mapping (vendored in cpp/thirdparty/vdb_gpdf, doc/dense_vdb_gpdf.md).
# Needs OpenVDB >= 9 (oneTBB builds; upstream uses github.com/nachovizzo/openvdb nacho/vdbfusion), PCL and glog.
#   isaeslam_enable_vdbgpdf(<target> <path to cpp/>)  - no-op unless -DISAESLAM_WITH_VDBGPDF=ON
option(ISAESLAM_WITH_VDBGPDF "Build the VDB-GPDF dense mapping backend (dense_mesh_method: vdbgpdf)" OFF)

function(isaeslam_enable_vdbgpdf target cpp_dir)
  if(NOT ISAESLAM_WITH_VDBGPDF)
    return()
  endif()
  list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/OpenVDB")
  find_package(OpenVDB REQUIRED)
  find_package(PCL REQUIRED COMPONENTS common search kdtree features)
  find_package(OpenMP REQUIRED)
  find_library(ISAESLAM_GLOG_LIB glog REQUIRED)
  message(STATUS "VDB-GPDF backend on: OpenVDB ${OpenVDB_VERSION}, PCL ${PCL_VERSION}")
  set(vdb ${cpp_dir}/thirdparty/vdb_gpdf)
  target_sources(${target} PRIVATE ${vdb}/src/VDBVolume.cpp ${vdb}/src/MarchingCubes.cpp)
  # upstream builds these with -O3 -fopenmp and a large Eigen stack limit (vdb_gpdf/CMakeLists.txt)
  set_source_files_properties(${vdb}/src/VDBVolume.cpp ${vdb}/src/MarchingCubes.cpp PROPERTIES
    COMPILE_OPTIONS "-O3;-Wno-unused-variable;-Wno-sign-compare;-Wno-reorder;-Wno-unused-but-set-variable")
  target_compile_definitions(${target} PRIVATE ISAESLAM_WITH_VDBGPDF EIGEN_STACK_ALLOCATION_LIMIT=10000000)
  target_include_directories(${target} PRIVATE ${vdb}/include)
  target_include_directories(${target} SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
  target_link_libraries(${target} OpenVDB::openvdb ${PCL_LIBRARIES} ${ISAESLAM_GLOG_LIB} OpenMP::OpenMP_CXX)
endfunction()
