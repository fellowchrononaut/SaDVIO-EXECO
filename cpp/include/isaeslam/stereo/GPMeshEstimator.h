#ifndef GP_MESH_ESTIMATOR_H
#define GP_MESH_ESTIMATOR_H

#include "isaeslam/stereo/DenseMesh.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace isae {

struct GPMeshConfig {
    double cell_size         = 0.20;  // world-frame BEV cell size in metres
    int    n_out             = 3;     // prediction grid per cell side
    int    min_pts_per_cell  = 4;
    int    max_pts_per_cell  = 30;
    double kappa             = 1.0;   // Laplacian kernel lengthscale
    double sigma2_noise      = 0.04;  // SGBM obs noise variance [m^2]
    double max_variance      = 0.25;  // drop vertices with sigma^2 > this
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
