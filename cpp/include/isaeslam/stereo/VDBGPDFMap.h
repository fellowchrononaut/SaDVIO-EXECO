#ifndef VDB_GPDF_MAP_H
#define VDB_GPDF_MAP_H

#include "isaeslam/stereo/DenseMesh.h"
#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace isae {

// Settings of the vendored VDB-GPDF volume (cpp/thirdparty/vdb_gpdf, doc/dense_vdb_gpdf.md). Names and
// defaults follow the upstream ROS parameters (VDB_GPDF ros2 branch, RGB-D "cow" preset); presets are
// read from <config>/vdbgpdf/<name>.yaml with the upstream key names by loadVDBGPDFPreset().
struct VDBGPDFMapConfig {
    // fusion variant: "gpdf" = VDB-GPDF (local GPDFs fused into the global VDB, the paper's method),
    // "tsdf" = the projective TSDF integration upstream keeps for its VDBFusion comparison
    std::string fusion = "gpdf";

    float  sdf_trunc             = 0.1f;   // "tsdf" only
    bool   space_carving         = true;   // "tsdf" only
    int    distance_method       = 0;      // 0: reverting GPDF, 1: Log-GPDF
    float  voxel_size_local      = 0.02f;
    int    voxel_overlapping     = -1;
    int    voxel_downsample      = 5;
    float  voxel_size_global     = 0.05f;
    int    variance_method       = 0;      // 0: constant weight, 1: 1-variance, 2/3: 1/variance fusion
    float  variance_cap          = 0.1f;
    float  variance_on_surface   = 0.001f;
    int    surface_normal_method = 0;      // 0: PCL normals, 1: raycasting
    int    surface_normal_num    = 20;
    float  surface_value         = -1.0f;
    float  query_iterval         = 0.05f;
    int    query_trunc_in        = 3;
    int    query_trunc_out       = 2;
    float  freespace_iterval     = 0.05f;
    int    freespace_trunc_out   = 2;
    float  query_downsample      = -1.0f;
    double map_lambda_scale      = 900.0;
    double map_noise             = 0.01;
    double smooth_param          = 100.0;

    // upstream mapper-level settings
    float  min_scan_range   = 0.1f;   // metres from the sensor origin
    float  max_scan_range   = 20.0f;
    bool   fill_holes       = true;
    float  recon_min_weight = 0.1f;
    bool   debug_print      = false;
};

// Reads a preset file (upstream key names, flat or under vdb_gpdf_mapping_node/ros__parameters);
// keys it lacks keep the values already in cfg. Throws std::runtime_error if the file cannot be read.
void loadVDBGPDFPreset(const std::string& path, VDBGPDFMapConfig& cfg);

// Global VDB-GPDF (or VDBFusion-style TSDF) map fed with world-frame point clouds.
// Only available in builds with ISAESLAM_WITH_VDBGPDF (OpenVDB, PCL, glog).
class VDBGPDFMap {
  public:
    explicit VDBGPDFMap(const VDBGPDFMapConfig& cfg);
    ~VDBGPDFMap();

    // points_world: one scan / keyframe in the world frame; origin: sensor position in the world
    void integrate(const std::vector<Eigen::Vector3d>& points_world, const Eigen::Vector3d& origin);

    // Marching cubes on the global distance grid (upstream ExtractTriangleMesh); cost grows with the map
    DenseMesh mesh() const;

    const VDBGPDFMapConfig& config() const { return _cfg; }

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    VDBGPDFMapConfig _cfg;
};

} // namespace isae
#endif
