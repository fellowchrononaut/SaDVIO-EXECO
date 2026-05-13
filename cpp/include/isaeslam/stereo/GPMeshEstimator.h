#ifndef GP_MESH_ESTIMATOR_H
#define GP_MESH_ESTIMATOR_H

#include "isaeslam/stereo/DenseMesh.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace isae {

struct GPMeshConfig {
    double cell_size        = 1.0;   // SLAMesh voxel/cell size in metres
    int    num_test         = 6;     // prediction grid per cell side, paper uses 6x6
    int    min_pts_per_cell = 8;
    double kernel_length    = 1.2;   // exponential kernel scale used by the reference code
    double variance_sensor  = 0.1;   // input noise standard deviation
    double max_variance     = 0.5;   // accept faces with average GP variance below this
    double max_depth        = 20.0;  // metres
    bool   full_cover       = false; // false: predict at cell-centred samples, as in SLAMesh defaults
    double eigen_1          = 48.0;  // surface/non-surface PCA threshold
    double eigen_2          = 0.95;  // distinctive angle threshold
    double eigen_3          = 0.2;   // obscure angle threshold
    bool   stitch_seams               = true; // connect compatible neighbouring GP cell patches
    double seam_max_variance          = 0.5;  // accept seam faces below this average GP variance
    double seam_max_edge_length       = 0.5;  // metres; rejects long bridge triangles
    double seam_max_prediction_gap    = 0.25; // metres along the GP prediction axis
    double seam_min_normal_cos        = 0.5;  // abs(dot) between seam and patch normals
};

class GPMeshEstimator {
  public:
    explicit GPMeshEstimator(const GPMeshConfig& cfg = GPMeshConfig()) : _cfg(cfg) {}

    // disp_float: float disparity map in pixels (already divided by 16)
    // f_rect: rectified focal length, baseline: stereo baseline [m], cx/cy: rectified principal point
    // T_w_rectcam: world-frame pose of the rectified left camera
    DenseMesh estimate(const cv::Mat& disp_float,
                       double f_rect, double baseline,
                       double cx_rect, double cy_rect,
                       const Eigen::Affine3d& T_w_rectcam);

  private:
    GPMeshConfig _cfg;
};

} // namespace isae
#endif
