#ifndef PRIMAL_DUAL_MESH_ESTIMATOR_H
#define PRIMAL_DUAL_MESH_ESTIMATOR_H

#include "isaeslam/stereo/DenseMesh.h"
#include "isaeslam/data/frame.h"
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <vector>
#include <memory>

namespace isae {

struct PrimalDualConfig {
    int    steiner_spacing = 20;   // px, spacing between Steiner grid points
    double lambda          = 0.5;  // data/tracking weight from the paper
    int    num_iterations  = 120;
    double tau             = 0.01; // primal step
    double sigma           = 0.1;  // dual step
    double theta           = 1.0;  // Chambolle-Pock extrapolation
    double min_depth       = 0.5;  // metres
    double max_depth       = 20.0; // metres
};

class PrimalDualMeshEstimator {
  public:
    explicit PrimalDualMeshEstimator(const PrimalDualConfig& cfg = PrimalDualConfig()) : _cfg(cfg) {}

    // frame is needed to project VIO landmarks into rectified image
    // R1_inv: R1.transpose() = rotation from rectified to original left cam (from stereoRectify R1)
    DenseMesh estimate(const cv::Mat& disp_float,
                       double f_rect, double baseline,
                       double cx_rect, double cy_rect,
                       const Eigen::Affine3d& T_w_rectcam,
                       const cv::Mat& R1_inv,
                       const std::shared_ptr<Frame>& frame);

  private:
    PrimalDualConfig _cfg;
};

} // namespace isae
#endif
