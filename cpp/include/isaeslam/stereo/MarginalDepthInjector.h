#ifndef MARGINAL_DEPTH_INJECTOR_H
#define MARGINAL_DEPTH_INJECTOR_H

#include "isaeslam/stereo/DenseMesh.h"
#include "isaeslam/stereo/GPGlobalMap.h"
#include "isaeslam/stereo/GPMeshEstimator.h"
#include "isaeslam/stereo/PrimalDualMeshEstimator.h"
#include "isaeslam/stereo/SGBMZNCCMeshEstimator.h"
#include "isaeslam/stereo/StereoMatcher.h"
#include "isaeslam/stereo/VDBGPDFMap.h"
#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/frame.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <thread>
#include <atomic>
#include <string>

namespace isae {

struct MarginalDepthConfig {
    std::string stereo_matcher  = "sgbm"; // "sgbm" or "ffs" (Fast-FoundationStereo, needs ISAESLAM_WITH_FFS)
    std::string ffs_engine_path;          // TensorRT engine for "ffs"
    double      ffs_lr_check_px = 0.0;    // "ffs": left-right consistency threshold in px (0 = off)
    int         num_disparities = 64;
    int         block_size      = 5;
    double      scale_factor    = 1.0;
    int         stride          = 2;
    double      max_depth       = 20.0;  // metres — points beyond this are dropped
    std::string mesh_method     = "none"; // "none"=depth only, "gp"=GP, "pd"=PD, "zncc"=SGBM+Delaunay+ZNCC, "vdbgpdf"=VDB-GPDF map
    double      zncc_threshold  = 0.8;
    double      max_length_threshold = 2.0;
    int         uniqueness_ratio    = 10;
    int         speckle_window_size = 100;
    int         speckle_range       = 32;
    int         disp12_max_diff     = 1;
    int         pre_filter_cap      = 0;
    bool        publish_sgbm_images = false;

    // SLAMesh-style GP meshing from dense SGBM points
    double gp_cell_size        = 1.0;
    int    gp_num_test         = 6;
    int    gp_min_pts_per_cell = 8;
    double gp_kernel_length    = 1.2;
    double gp_variance_sensor  = 0.1;
    double gp_max_variance     = 0.5;
    bool   gp_full_cover       = false;
    double gp_eigen_1          = 48.0;
    double gp_eigen_2          = 0.95;
    double gp_eigen_3          = 0.2;
    bool   gp_stitch_seams            = true;
    double gp_seam_max_variance       = 0.5;
    double gp_seam_max_edge_length    = 0.5;
    double gp_seam_max_prediction_gap = 0.25;
    double gp_seam_min_normal_cos     = 0.5;

    // Global GP map: SLAMesh map update (A) and optional frame-to-model registration (B)
    bool        keep_all_keyframes           = false;
    bool        gp_global_map                = false;
    double      gp_variance_map_update       = 0.5;
    int         gp_max_raw_points_per_cell   = 2000;
    bool        gp_register                  = false;
    int         gp_register_times            = 5;
    double      gp_variance_register         = 0.1;
    int         gp_cross_cell_overlap_length = 1;
    double      gp_register_converge_thr     = 1e-5;
    double      gp_register_huber            = 0.1;
    int         gp_register_min_matches      = 30;
    double      gp_register_max_translation  = 0.5;
    double      gp_register_max_rotation_deg = 10.0;
    bool        gp_register_carry_correction = true;
    bool        gp_register_depth_weighting  = false;
    double      gp_register_depth_ref        = 2.0;
    std::string gp_global_mesh_path          = "log_slam/dense_gp_global_mesh.ply";
    int         gp_save_every                = 5;

    // VDB-GPDF global map (mesh_method "vdbgpdf", doc/dense_vdb_gpdf.md; needs ISAESLAM_WITH_VDBGPDF)
    std::string vdbgpdf_preset_path;                                      // preset yaml, upstream key names
    int         vdbgpdf_stride     = 2;                                   // pixel stride of the points fed to the map
    int         vdbgpdf_mesh_every = 5;                                   // keyframes between mesh extraction + PLY save
    std::string vdbgpdf_mesh_path  = "log_slam/dense_vdbgpdf_mesh.ply";

    // Primal-dual mesh optimization over SGBM inverse depth
    int    pd_steiner_spacing = 20;
    double pd_lambda          = 0.5;
    int    pd_num_iterations  = 120;
    double pd_tau             = 0.01;
    double pd_sigma           = 0.1;
    double pd_theta           = 1.0;
    double pd_min_depth       = 0.5;
};

// Holds the latest processed result, readable by the ROS visualizer.
struct DenseResult {
    cv::Mat                      depth_img;   // float32 depth in metres, same size as input image
    cv::Mat                      sgbm_left_img;  // rectified/scaled grayscale image used by SGBM
    cv::Mat                      sgbm_right_img; // rectified/scaled grayscale image used by SGBM
    std::vector<Eigen::Vector3d> point_cloud; // world-frame 3D points back-projected from depth_img
    DenseMesh                    mesh;        // GP/PD mesh (vertices, faces, normals) — empty if method="none"
    bool                         valid = false;
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

    std::unique_ptr<StereoMatcher> _matcher;
    MarginalDepthConfig _cfg;

    GPMeshConfig _gp_cfg;
    PrimalDualConfig _pd_cfg;
    SGBMZNCCConfig _zncc_cfg;

    // Queue: single-slot drop policy by default; FIFO when cfg.keep_all_keyframes
    std::mutex _q_mtx;
    std::condition_variable _q_cv;
    std::deque<QueueItem> _queue;
    size_t _dropped = 0;

    // Global GP map (only used by the worker thread)
    void integrateGlobalMap(const QueueItem& item, const cv::Mat& disp_float, double f_rect,
                            double cx_rect, double cy_rect, std::vector<Eigen::Vector3d>& point_cloud,
                            DenseMesh& dense_mesh);
    std::unique_ptr<GPGlobalMap> _gp_map;

    // VDB-GPDF map (only used by the worker thread)
    void integrateVDBGPDF(const QueueItem& item, const cv::Mat& disp_float, double f_rect, double cx_rect,
                          double cy_rect, DenseMesh& dense_mesh);
    std::unique_ptr<VDBGPDFMap> _vdb_map;
    DenseMesh _vdb_mesh; // last extracted mesh, republished between extractions
    std::ofstream _vdb_log; // dense_vdbgpdf_keyframes.csv: pose each keyframe was fused with
    int _integrated = 0;
    std::ofstream _reg_log;
    std::atomic<bool> _running{true};
    std::thread _worker;

    // Latest result, written by worker, read by visualizer
    std::mutex  _result_mtx;
    DenseResult _latest_result;
};

} // namespace isae
#endif
