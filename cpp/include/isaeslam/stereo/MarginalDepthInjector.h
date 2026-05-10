#ifndef MARGINAL_DEPTH_INJECTOR_H
#define MARGINAL_DEPTH_INJECTOR_H

#include "isaeslam/stereo/DenseMesh.h"
#include "isaeslam/stereo/GPMeshEstimator.h"
#include "isaeslam/stereo/PrimalDualMeshEstimator.h"
#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/frame.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

namespace isae {

struct MarginalDepthConfig {
    int    num_disparities  = 64;
    int    block_size       = 5;
    double scale_factor     = 1.0;
    int    stride           = 2;
    int    sgbm_num_threads = 2;
};

// Holds the latest processed result, readable by the ROS visualizer.
struct DenseResult {
    cv::Mat   depth_img; // float32 depth in metres, same size as input image
    DenseMesh mesh;      // GP mesh in world frame (vertices, faces, normals)
    bool      valid = false;
};

class MarginalDepthInjector {
  public:
    MarginalDepthInjector(const Eigen::Matrix3d& K_L, const Eigen::Vector4d& d_L,
                          const Eigen::Matrix3d& K_R, const Eigen::Vector4d& d_R,
                          const Eigen::Affine3d& T_right_in_left,
                          const cv::Size& img_size,
                          const MarginalDepthConfig& cfg = MarginalDepthConfig());

    ~MarginalDepthInjector();

    // Queue a frame for async dense mesh computation.
    // Must be called BEFORE discardLastFrame().
    void queueFrame(const std::shared_ptr<Frame>& frame, const std::shared_ptr<Mesh3D>& mesh);

    // Consume the latest result. Returns false if no new result is available.
    // Clears the stored result so the next call returns false until new data arrives.
    bool pollResult(DenseResult& out);

  private:
    struct QueueItem {
        cv::Mat img_L, img_R;
        Eigen::Affine3d T_w_rectcam;
        std::shared_ptr<Frame> frame;
        std::shared_ptr<Mesh3D> mesh;
        bool valid = false;
    };

    void workerLoop();
    void processItem(const QueueItem& item);

    // Stereo rectification maps (computed once)
    cv::Mat _map_L_x, _map_L_y, _map_R_x, _map_R_y;
    cv::Mat _R1_mat; // rotation from original to rectified left (3x3 float)

    // Rectified camera intrinsics
    double _f_rect, _cx_rect, _cy_rect, _baseline;

    cv::Ptr<cv::StereoSGBM> _sgbm;
    MarginalDepthConfig _cfg;

    GPMeshConfig _gp_cfg;
    PrimalDualConfig _pd_cfg;

    // Queue (single-slot, drop policy)
    std::mutex _q_mtx;
    std::condition_variable _q_cv;
    QueueItem _pending;
    std::atomic<bool> _running{true};
    std::thread _worker;

    // Latest result, written by worker, read by visualizer
    std::mutex  _result_mtx;
    DenseResult _latest_result;
};

} // namespace isae
#endif
