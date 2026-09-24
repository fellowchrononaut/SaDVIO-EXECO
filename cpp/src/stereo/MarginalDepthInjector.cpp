#include "isaeslam/stereo/MarginalDepthInjector.h"
#include "isaeslam/data/sensors/ASensor.h"
#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <unistd.h>  // nice()

namespace isae {

MarginalDepthInjector::MarginalDepthInjector(const Eigen::Matrix3d& K_L,
                                              const Eigen::Vector4d& d_L,
                                              const Eigen::Matrix3d& K_R,
                                              const Eigen::Vector4d& d_R,
                                              const Eigen::Affine3d& T_right_in_left,
                                              const cv::Size& img_size,
                                              const MarginalDepthConfig& cfg)
    : _cfg(cfg)
{
    // Convert Eigen matrices to OpenCV
    cv::Mat K_L_cv, K_R_cv, d_L_cv, d_R_cv;
    cv::eigen2cv(K_L, K_L_cv);
    cv::eigen2cv(K_R, K_R_cv);

    // distortion coefficients as 1x4
    cv::Mat d_L_tmp(1, 4, CV_64F);
    cv::Mat d_R_tmp(1, 4, CV_64F);
    for (int i = 0; i < 4; ++i) {
        d_L_tmp.at<double>(0, i) = d_L(i);
        d_R_tmp.at<double>(0, i) = d_R(i);
    }
    d_L_cv = d_L_tmp;
    d_R_cv = d_R_tmp;

    // T_right_in_left: T_{left <- right}
    // OpenCV stereoRectify convention: R, T are the pose of the left camera in the right camera frame
    // i.e., T_{right <- left}, which is T_right_in_left.inverse()
    Eigen::Affine3d T_L_in_R = T_right_in_left.inverse();
    Eigen::Matrix3d R_eig = T_L_in_R.linear();
    Eigen::Vector3d t_eig = T_L_in_R.translation();

    cv::Mat R_cv, T_cv;
    cv::eigen2cv(R_eig, R_cv);
    cv::eigen2cv(t_eig, T_cv);

    // Stereo rectification
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K_L_cv, d_L_cv, K_R_cv, d_R_cv,
                      img_size, R_cv, T_cv,
                      R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 0.0);

    // Store R1 (original -> rectified left)
    R1.convertTo(_R1_mat, CV_32F);

    // Extract rectified intrinsics from P1
    _f_rect  = P1.at<double>(0, 0);
    _cx_rect = P1.at<double>(0, 2);
    _cy_rect = P1.at<double>(1, 2);

    // Extract baseline: P2(0,3) = -f_rect * baseline
    _baseline = -P2.at<double>(0, 3) / P2.at<double>(0, 0);
    _zncc_cfg.vertex_stride    = std::max(8, cfg.stride * 4);
    _zncc_cfg.zncc_threshold   = cfg.zncc_threshold;
    _zncc_cfg.max_edge_length  = cfg.max_length_threshold;
    _zncc_cfg.max_depth        = cfg.max_depth;

    _gp_cfg.cell_size        = cfg.gp_cell_size;
    _gp_cfg.num_test         = cfg.gp_num_test;
    _gp_cfg.min_pts_per_cell = cfg.gp_min_pts_per_cell;
    _gp_cfg.kernel_length    = cfg.gp_kernel_length;
    _gp_cfg.variance_sensor  = cfg.gp_variance_sensor;
    _gp_cfg.max_variance     = cfg.gp_max_variance;
    _gp_cfg.max_depth        = cfg.max_depth;
    _gp_cfg.full_cover       = cfg.gp_full_cover;
    _gp_cfg.eigen_1          = cfg.gp_eigen_1;
    _gp_cfg.eigen_2          = cfg.gp_eigen_2;
    _gp_cfg.eigen_3          = cfg.gp_eigen_3;
    _gp_cfg.stitch_seams            = cfg.gp_stitch_seams;
    _gp_cfg.seam_max_variance       = cfg.gp_seam_max_variance;
    _gp_cfg.seam_max_edge_length    = cfg.gp_seam_max_edge_length;
    _gp_cfg.seam_max_prediction_gap = cfg.gp_seam_max_prediction_gap;
    _gp_cfg.seam_min_normal_cos     = cfg.gp_seam_min_normal_cos;

    if (cfg.gp_register && !cfg.gp_global_map)
        std::cerr << "[DenseMesh] dense_gp_register needs dense_gp_global_map; registration disabled" << std::endl;
    if (cfg.mesh_method == "gp" && cfg.gp_global_map) {
        GPGlobalMapConfig gcfg;
        gcfg.gp                          = _gp_cfg;
        gcfg.variance_map_update         = cfg.gp_variance_map_update;
        gcfg.max_raw_points_per_cell     = cfg.gp_max_raw_points_per_cell;
        gcfg.register_enabled            = cfg.gp_register;
        gcfg.register_times              = cfg.gp_register_times;
        gcfg.variance_register           = cfg.gp_variance_register;
        gcfg.cross_cell_overlap_length   = cfg.gp_cross_cell_overlap_length;
        gcfg.converge_thr                = cfg.gp_register_converge_thr;
        gcfg.huber_delta                 = cfg.gp_register_huber;
        gcfg.min_matches                 = cfg.gp_register_min_matches;
        gcfg.max_correction_translation  = cfg.gp_register_max_translation;
        gcfg.max_correction_rotation_deg = cfg.gp_register_max_rotation_deg;
        gcfg.carry_correction            = cfg.gp_register_carry_correction;
        gcfg.depth_weighting             = cfg.gp_register_depth_weighting;
        gcfg.depth_weighting_ref         = cfg.gp_register_depth_ref;
        _gp_map = std::make_unique<GPGlobalMap>(gcfg);

        const std::filesystem::path mesh_path(cfg.gp_global_mesh_path);
        const std::filesystem::path dir = mesh_path.has_parent_path() ? mesh_path.parent_path()
                                                                      : std::filesystem::path(".");
        std::filesystem::create_directories(dir);
        _reg_log.open(dir / "dense_gp_registration.csv", std::ios::trunc);
        _reg_log << "timestamp_ns,keyframe,status,attempted,accepted,iterations,matches,"
                    "scan_cells,cells_fused,cells_added,map_cells,"
                    "vo_x,vo_y,vo_z,vo_qx,vo_qy,vo_qz,vo_qw,"
                    "used_x,used_y,used_z,used_qx,used_qy,used_qz,used_qw,"
                    "corr_t_m,corr_rot_deg\n";
        std::cout << "[DenseMesh] global GP map on (registration " << (cfg.gp_register ? "on" : "off")
                  << ", keep_all_keyframes=" << cfg.keep_all_keyframes << ") -> " << cfg.gp_global_mesh_path
                  << std::endl;
    }

    _pd_cfg.steiner_spacing = cfg.pd_steiner_spacing;
    _pd_cfg.lambda          = cfg.pd_lambda;
    _pd_cfg.num_iterations  = cfg.pd_num_iterations;
    _pd_cfg.tau             = cfg.pd_tau;
    _pd_cfg.sigma           = cfg.pd_sigma;
    _pd_cfg.theta           = cfg.pd_theta;
    _pd_cfg.min_depth       = cfg.pd_min_depth;
    _pd_cfg.max_depth       = cfg.max_depth;

    // Compute undistort+rectify maps for both cameras
    cv::initUndistortRectifyMap(K_L_cv, d_L_cv, R1, P1,
                                img_size, CV_32FC1,
                                _map_L_x, _map_L_y);
    cv::initUndistortRectifyMap(K_R_cv, d_R_cv, R2, P2,
                                img_size, CV_32FC1,
                                _map_R_x, _map_R_y);

    // Set up SGBM
    int block_size = cfg.block_size;
    int num_disp   = cfg.num_disparities;
    int P1_sgbm    = 8  * 1 * block_size * block_size;
    int P2_sgbm    = 32 * 1 * block_size * block_size;
    _sgbm = cv::StereoSGBM::create(0, num_disp, block_size,
                                    P1_sgbm, P2_sgbm,
                                    cfg.disp12_max_diff,
                                    cfg.pre_filter_cap,
                                    cfg.uniqueness_ratio,
                                    cfg.speckle_window_size,
                                    cfg.speckle_range,
                                    cv::StereoSGBM::MODE_SGBM_3WAY);

    // Start worker thread
    _worker = std::thread(&MarginalDepthInjector::workerLoop, this);
}

MarginalDepthInjector::~MarginalDepthInjector() {
    _running = false;
    _q_cv.notify_all();
    if (_worker.joinable())
        _worker.join();
    if (_gp_map && _cfg.gp_save_every > 0)
        _gp_map->savePly(_cfg.gp_global_mesh_path);
}

void MarginalDepthInjector::queueFrame(const std::shared_ptr<Frame>& frame,
                                        const std::shared_ptr<Mesh3D>& mesh) {
    if (!frame || frame->getSensors().empty())
        return;

    // Get images from sensors
    cv::Mat img_L, img_R;
    if (frame->getSensors().size() >= 1)
        img_L = frame->getSensors().at(0)->getRawData().clone();
    if (frame->getSensors().size() >= 2)
        img_R = frame->getSensors().at(1)->getRawData().clone();

    if (img_L.empty() || img_R.empty())
        return;

    // Compute T_w_rectcam:
    // R1 maps from original left cam to rectified left cam: p_rect = R1 * p_orig
    // T_{w<-rectcam} = T_{w<-cam} * T_{cam<-rectcam} = T_{w<-cam} * R1^T
    Eigen::Matrix3d R1_eig;
    {
        cv::Mat R1_double;
        _R1_mat.convertTo(R1_double, CV_64F);
        cv::cv2eigen(R1_double, R1_eig);
    }

    // getSensor2WorldTransform gives T_{world <- sensor} = T_{w <- cam}
    Eigen::Affine3d T_w_rectcam = frame->getSensors().at(0)->getSensor2WorldTransform();
    T_w_rectcam.linear() = T_w_rectcam.linear() * R1_eig.transpose();
    // Translation stays the same (camera origin doesn't move under pure rotation)

    QueueItem item;
    item.img_L       = std::move(img_L);
    item.img_R       = std::move(img_R);
    item.T_w_rectcam = T_w_rectcam;
    item.frame       = frame;
    item.mesh        = mesh;
    item.valid       = true;

    {
        std::lock_guard<std::mutex> lock(_q_mtx);
        if (!_cfg.keep_all_keyframes && !_queue.empty()) {
            _dropped += _queue.size(); // drop previous if not consumed
            _queue.clear();
        }
        _queue.push_back(std::move(item));
    }
    _q_cv.notify_one();
}

void MarginalDepthInjector::workerLoop() {
    // Lower this thread's OS priority so VIO threads always win CPU arbitration
    // when competing for the same core. When cores are free, SGBM runs at full
    // speed unthrottled. nice() on Linux is per-thread and only affects scheduling
    // decisions under contention — not an artificial cap like cv::setNumThreads().
    [[maybe_unused]] int _nice_ret = nice(10);

    while (_running) {
        QueueItem item;
        {
            std::unique_lock<std::mutex> lock(_q_mtx);
            _q_cv.wait(lock, [this]{ return !_queue.empty() || !_running; });
            if (!_running) break;
            item = std::move(_queue.front());
            _queue.pop_front();
        }
        processItem(item);
    }
}

void MarginalDepthInjector::processItem(const QueueItem& item) {
    if (!item.valid || item.img_L.empty() || item.img_R.empty())
        return;

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();

    // Remap (rectify) images
    cv::Mat rect_L, rect_R;
    cv::remap(item.img_L, rect_L, _map_L_x, _map_L_y, cv::INTER_LINEAR);
    cv::remap(item.img_R, rect_R, _map_R_x, _map_R_y, cv::INTER_LINEAR);

    // Scale if needed
    if (std::abs(_cfg.scale_factor - 1.0) > 1e-6) {
        cv::resize(rect_L, rect_L, cv::Size(), _cfg.scale_factor, _cfg.scale_factor, cv::INTER_LINEAR);
        cv::resize(rect_R, rect_R, cv::Size(), _cfg.scale_factor, _cfg.scale_factor, cv::INTER_LINEAR);
    }

    // Convert to grayscale if needed
    cv::Mat gray_L, gray_R;
    if (rect_L.channels() == 3)
        cv::cvtColor(rect_L, gray_L, cv::COLOR_BGR2GRAY);
    else
        gray_L = rect_L;

    if (rect_R.channels() == 3)
        cv::cvtColor(rect_R, gray_R, cv::COLOR_BGR2GRAY);
    else
        gray_R = rect_R;

    // Run SGBM
    cv::Mat disp_raw;
    _sgbm->compute(gray_L, gray_R, disp_raw);

    auto t1 = clk::now();

    // Convert to float (SGBM output is 16-bit fixed-point, divide by 16)
    cv::Mat disp_float;
    disp_raw.convertTo(disp_float, CV_32F, 1.0 / 16.0);

    // Scale intrinsics if images were rescaled
    double f_rect  = _f_rect  * _cfg.scale_factor;
    double cx_rect = _cx_rect * _cfg.scale_factor;
    double cy_rect = _cy_rect * _cfg.scale_factor;

    // Convert disparity to metric depth (float32, metres).
    // Use scaled focal length (f_rect) so depth is consistent with scaled-image pixel coords.
    cv::Mat depth_img = cv::Mat::zeros(disp_float.size(), CV_32F);
    for (int v = 0; v < disp_float.rows; ++v) {
        const float* d_row = disp_float.ptr<float>(v);
        float*       z_row = depth_img.ptr<float>(v);
        for (int u = 0; u < disp_float.cols; ++u) {
            if (d_row[u] > 0.f)
                z_row[u] = static_cast<float>(f_rect * _baseline / d_row[u]);
        }
    }

    auto t2 = clk::now();

    // Backproject depth image to world-frame point cloud (always, regardless of mesh method).
    // Uses stride from config to subsample — reduces density without losing structure.
    std::vector<Eigen::Vector3d> point_cloud;
    {
        const int str        = std::max(1, _cfg.stride);
        const float max_z    = static_cast<float>(_cfg.max_depth);
        point_cloud.reserve((depth_img.rows / str) * (depth_img.cols / str));
        for (int v = 0; v < depth_img.rows; v += str) {
            const float* z_row = depth_img.ptr<float>(v);
            for (int u = 0; u < depth_img.cols; u += str) {
                float z = z_row[u];
                if (z <= 0.f || !std::isfinite(z) || z > max_z)
                    continue;
                double Z = z;
                double X = (u - cx_rect) * Z / f_rect;
                double Y = (v - cy_rect) * Z / f_rect;
                point_cloud.push_back(item.T_w_rectcam * Eigen::Vector3d(X, Y, Z));
            }
        }
    }

    auto t3 = clk::now();

    // Run mesh estimator based on method — dense pipeline is fully independent
    // of the sparse Mesh3D; injectDensePoints() is never called.
    DenseMesh dense_mesh;
    if (_cfg.mesh_method == "gp" && _gp_map) {
        integrateGlobalMap(item, disp_float, f_rect, cx_rect, cy_rect, point_cloud, dense_mesh);
    } else if (_cfg.mesh_method == "gp") {
        GPMeshEstimator gp_estimator(_gp_cfg);
        dense_mesh = gp_estimator.estimate(disp_float, f_rect, _baseline,
                                           cx_rect, cy_rect, item.T_w_rectcam);
    } else if (_cfg.mesh_method == "pd") {
        PrimalDualMeshEstimator pd_estimator(_pd_cfg);
        dense_mesh = pd_estimator.estimate(disp_float, f_rect, _baseline,
                                           cx_rect, cy_rect, item.T_w_rectcam,
                                           _R1_mat, item.frame);
    } else if (_cfg.mesh_method == "zncc" || _cfg.mesh_method == "sgbm_zncc") {
        SGBMZNCCMeshEstimator zncc_estimator(_zncc_cfg);
        dense_mesh = zncc_estimator.estimate(disp_float, gray_L, gray_R,
                                             f_rect, _baseline, cx_rect, cy_rect,
                                             item.T_w_rectcam);
    }

    auto t4 = clk::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    std::cout << "[DenseMesh] SGBM=" << ms(t0,t1) << "ms"
              << "  depth=" << ms(t1,t2) << "ms"
              << "  cloud=" << ms(t2,t3) << "ms"
              << "  " << _cfg.mesh_method << "=" << ms(t3,t4) << "ms"
              << "  total=" << ms(t0,t4) << "ms"
              << "  pts=" << point_cloud.size()
              << "  mesh_v=" << dense_mesh.vertices.size()
              << "  mesh_f=" << dense_mesh.faces.size()
              << "  dropped=" << _dropped << std::endl;

    // Store result for the ROS visualizer to poll
    {
        std::lock_guard<std::mutex> lock(_result_mtx);
        _latest_result.depth_img = depth_img.clone();
        if (_cfg.publish_sgbm_images) {
            _latest_result.sgbm_left_img  = gray_L.clone();
            _latest_result.sgbm_right_img = gray_R.clone();
        } else {
            _latest_result.sgbm_left_img.release();
            _latest_result.sgbm_right_img.release();
        }
        _latest_result.point_cloud = std::move(point_cloud);
        _latest_result.mesh        = dense_mesh;
        _latest_result.valid       = true;
    }
}

void MarginalDepthInjector::integrateGlobalMap(const QueueItem& item, const cv::Mat& disp_float, double f_rect,
                                               double cx_rect, double cy_rect,
                                               std::vector<Eigen::Vector3d>& point_cloud, DenseMesh& dense_mesh) {
    // Same point selection as GPMeshEstimator::estimate, kept in the rectified camera frame so the
    // map can re-place them when registration moves the keyframe.
    std::vector<Eigen::Vector3d> points_cam;
    points_cam.reserve(disp_float.rows * disp_float.cols / 2);
    for (int v = 0; v < disp_float.rows; ++v) {
        const float* row_ptr = disp_float.ptr<float>(v);
        for (int u = 0; u < disp_float.cols; ++u) {
            const float disp = row_ptr[u];
            if (disp <= 0.0f || !std::isfinite(disp))
                continue;
            const double Z = f_rect * _baseline / static_cast<double>(disp);
            if (Z <= 0.0 || Z > _gp_cfg.max_depth || !std::isfinite(Z))
                continue;
            points_cam.emplace_back((u - cx_rect) * Z / f_rect, (v - cy_rect) * Z / f_rect, Z);
        }
    }

    const GPIntegrationReport rep = _gp_map->integrate(points_cam, item.T_w_rectcam);
    ++_integrated;

    // Keep the published per-keyframe cloud consistent with the pose the map used.
    const Eigen::Affine3d C = rep.T_w_cam_used * item.T_w_rectcam.inverse();
    for (auto& p : point_cloud)
        p = C * p;
    dense_mesh = _gp_map->mesh();

    if (_reg_log.is_open()) {
        auto pose = [&](const Eigen::Affine3d& T) {
            const Eigen::Quaterniond q(T.linear());
            _reg_log << "," << T.translation().x() << "," << T.translation().y() << "," << T.translation().z()
                     << "," << q.x() << "," << q.y() << "," << q.z() << "," << q.w();
        };
        const Eigen::Affine3d corr = rep.T_w_cam_used * rep.T_w_cam_vo.inverse();
        _reg_log << std::setprecision(10) << (item.frame ? item.frame->getTimestamp() : 0ULL) << ","
                 << _integrated << "," << rep.status << "," << rep.register_attempted << ","
                 << rep.register_accepted << "," << rep.iterations << "," << rep.matches << ","
                 << rep.scan_cells << "," << rep.cells_fused << "," << rep.cells_added << "," << rep.map_cells;
        pose(rep.T_w_cam_vo);
        pose(rep.T_w_cam_used);
        _reg_log << "," << corr.translation().norm() << ","
                 << Eigen::AngleAxisd(corr.linear()).angle() * 180.0 / M_PI << "\n";
        _reg_log.flush();
    }
    std::cout << "[DenseMesh] gp_map kf=" << _integrated << " reg=" << rep.status << " it=" << rep.iterations
              << " matches=" << rep.matches << " cells: scan=" << rep.scan_cells << " fused=" << rep.cells_fused
              << " added=" << rep.cells_added << " map=" << rep.map_cells << std::endl;

    if (_cfg.gp_save_every > 0 && _integrated % _cfg.gp_save_every == 0 &&
        !_gp_map->savePly(_cfg.gp_global_mesh_path))
        std::cerr << "[DenseMesh] could not write " << _cfg.gp_global_mesh_path << std::endl;
}

bool MarginalDepthInjector::pollResult(DenseResult& out) {
    std::lock_guard<std::mutex> lock(_result_mtx);
    if (!_latest_result.valid)
        return false;
    out = std::move(_latest_result);
    _latest_result.valid = false;
    return true;
}

} // namespace isae
