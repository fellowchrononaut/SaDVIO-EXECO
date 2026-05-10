#ifndef SGBM_ZNCC_MESH_ESTIMATOR_H
#define SGBM_ZNCC_MESH_ESTIMATOR_H

#include "isaeslam/stereo/DenseMesh.h"

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace isae {

struct SGBMZNCCConfig {
    int    vertex_stride          = 16;   // image-space sampling step in pixels
    int    zncc_patch_size        = 7;    // odd patch size around triangle centroid
    double zncc_threshold         = 0.8;
    double max_edge_length        = 2.0;  // metres
    double max_depth_jump         = 1.0;  // metres between triangle vertices
    double max_depth              = 20.0; // metres
};

class SGBMZNCCMeshEstimator {
  public:
    explicit SGBMZNCCMeshEstimator(const SGBMZNCCConfig& cfg = SGBMZNCCConfig()) : _cfg(cfg) {}

    DenseMesh estimate(const cv::Mat& disp,
                       const cv::Mat& left_img,
                       const cv::Mat& right_img,
                       double f_rect,
                       double baseline,
                       double cx_rect,
                       double cy_rect,
                       const Eigen::Affine3d& T_w_rectcam) const;

  private:
    struct VertexSample {
        cv::Point2f uv;
        double disparity = 0.0;
        double depth = 0.0;
        Eigen::Vector3d p_world = Eigen::Vector3d::Zero();
    };

    bool passGeometry(const VertexSample& a, const VertexSample& b, const VertexSample& c) const;
    bool passZNCC(const cv::Mat& left_img,
                  const cv::Mat& right_img,
                  const VertexSample& a,
                  const VertexSample& b,
                  const VertexSample& c) const;

    SGBMZNCCConfig _cfg;
};

} // namespace isae

#endif // SGBM_ZNCC_MESH_ESTIMATOR_H
