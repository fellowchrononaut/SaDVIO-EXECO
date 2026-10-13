#include "isaeslam/stereo/MarginalDepthInjector.h"
#include "isaeslam/data/sensors/ASensor.h"
#include <opencv2/core/eigen.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <unordered_set>
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
        _gp_global_cfg = gcfg;
        if (!cfg.anchored)
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

    if (cfg.mesh_method == "vdbgpdf") {
#ifdef ISAESLAM_WITH_VDBGPDF
        VDBGPDFMapConfig vcfg;
        loadVDBGPDFPreset(cfg.vdbgpdf_preset_path, vcfg);
        _vdb_cfg = vcfg;
        if (!cfg.anchored)
            _vdb_map = std::make_unique<VDBGPDFMap>(vcfg);
        const std::filesystem::path mesh_path(cfg.vdbgpdf_mesh_path);
        const std::filesystem::path dir = mesh_path.has_parent_path() ? mesh_path.parent_path()
                                                                      : std::filesystem::path(".");
        std::filesystem::create_directories(dir);

        _vdb_log.open(dir / "dense_vdbgpdf_keyframes.csv", std::ios::trunc);
        _vdb_log << "timestamp_ns,keyframe,points,integrate_ms,x,y,z,qx,qy,qz,qw\n";
        std::cout << "[DenseMesh] VDB-GPDF map on (preset " << cfg.vdbgpdf_preset_path << ", stride "
                  << cfg.vdbgpdf_stride << ", keep_all_keyframes=" << cfg.keep_all_keyframes << ") -> "
                  << cfg.vdbgpdf_mesh_path << std::endl;
#else
        throw std::runtime_error("[DenseMesh] dense_mesh_method=vdbgpdf needs a build with -DISAESLAM_WITH_VDBGPDF=ON");
#endif
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

    // Set up the disparity backend (SGBM, Fast-FoundationStereo or Lite Any Stereo V2)
    StereoMatcherConfig mcfg;
    mcfg.matcher             = cfg.stereo_matcher;
    mcfg.num_disparities     = cfg.num_disparities;
    mcfg.block_size          = cfg.block_size;
    mcfg.uniqueness_ratio    = cfg.uniqueness_ratio;
    mcfg.speckle_window_size = cfg.speckle_window_size;
    mcfg.speckle_range       = cfg.speckle_range;
    mcfg.disp12_max_diff     = cfg.disp12_max_diff;
    mcfg.pre_filter_cap      = cfg.pre_filter_cap;
    mcfg.ffs_engine_path     = cfg.ffs_engine_path;
    mcfg.ffs_lr_check_px     = cfg.ffs_lr_check_px;
    mcfg.las2_onnx_path      = cfg.las2_onnx_path;
    mcfg.las2_device         = cfg.las2_device;
    mcfg.las2_precision      = cfg.las2_precision;
    mcfg.las2_threads        = cfg.las2_threads;
    mcfg.las2_lr_check_px    = cfg.las2_lr_check_px;
    _matcher = createStereoMatcher(mcfg);

    if (cfg.anchored) {
        AnchoredMapsConfig acfg;
        acfg.mesh_path  = cfg.anchored_mesh_path;
        acfg.save_every = cfg.anchored_save_every;
        acfg.submap_kfs = std::max(1, cfg.submap_kfs);
        acfg.fused_view      = cfg.fused_view;
        acfg.fused_mesh_path = cfg.fused_mesh_path;
        // The fused view: vdbgpdf merges its own submap volumes (same settings); the other methods integrate their
        // geometry into TSDF chunks first (5 cm voxels, 10 cm truncation, no space carving: a chunk holds geometry
        // seen from several cameras; up to 8 m from the camera, as the stereo_tsdf preset: the GP estimators keep
        // stereo points up to 20 m, which made 60-78 M voxels of noise on the EuRoC rooms)
        acfg.view_from_geometry = cfg.mesh_method != "vdbgpdf";
        if (cfg.mesh_method == "vdbgpdf") {
            acfg.view_cfg = _vdb_cfg;
        } else {
            acfg.view_cfg.fusion           = "tsdf";
            acfg.view_cfg.sdf_trunc        = 0.1f;
            acfg.view_cfg.space_carving    = false;
            acfg.view_cfg.voxel_size_global = 0.05f;
            acfg.view_cfg.min_scan_range   = 0.05f;
            acfg.view_cfg.max_scan_range   = 8.f;
            acfg.view_cfg.recon_min_weight = 0.1f;
        }
        _anchors        = std::make_unique<AnchoredMaps>(acfg);
        std::cout << "[DenseMesh] anchored maps (" << cfg.mesh_method << ", "
                  << (fusedMethod() ? "submaps of " + std::to_string(acfg.submap_kfs) + " keyframes"
                                    : std::string("one anchor per keyframe"))
                  << ", fused view " << cfg.fused_view << ") -> " << cfg.anchored_mesh_path << std::endl;
        if (!cfg.keep_all_keyframes)
            std::cerr << "[DenseMesh] anchored maps without dense_keep_all_keyframes: keyframes may be dropped"
                      << std::endl;
    }

    // Start worker thread
    _worker = std::thread(&MarginalDepthInjector::workerLoop, this);
}

MarginalDepthInjector::~MarginalDepthInjector() {
    _running = false;
    _q_cv.notify_all();
    if (_worker.joinable())
        _worker.join();
    writeFinal();
}

void MarginalDepthInjector::finish() {
    _draining = true;
    _q_cv.notify_all();
    if (_worker.joinable())
        _worker.join();
    writeFinal();
}

void MarginalDepthInjector::writeFinal() {
    std::call_once(_final_once, [this] {
        if (_gp_map && _cfg.gp_save_every > 0)
            _gp_map->savePly(_cfg.gp_global_mesh_path);
        if (_vdb_map && _cfg.vdbgpdf_mesh_every > 0)
            writeDenseMeshPly(_vdb_map->mesh(), _cfg.vdbgpdf_mesh_path, "SaDVIO dense VDB-GPDF map");
        if (_anchors && _anchors->size() > 0) {
            _anchors->closeOpenSubmap([](AnchoredMaps::Anchor& a) {
                return a.vdb ? a.vdb->mesh() : a.gp ? a.gp->mesh() : a.mesh;
            });
            _anchors->finish();
        }
    });
}

void MarginalDepthInjector::setKeyframeCorrection(KeyframeCorrection fn) {
    if (_anchors)
        _anchors->setKeyframeCorrection(std::move(fn));
}

void MarginalDepthInjector::queueFrame(const std::shared_ptr<Frame>& frame,
                                        const std::shared_ptr<Mesh3D>& mesh, int segment) {
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
    item.T_w_f       = frame->getFrame2WorldTransform();
    item.frame       = frame;
    item.mesh        = mesh;
    item.segment     = segment;
    item.epoch       = _world_epoch;
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
            _q_cv.wait(lock, [this]{ return !_queue.empty() || !_running || _draining; });
            if (!_running || (_draining && _queue.empty())) break;
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

    // Disparity in pixels (float, <= 0 = invalid)
    cv::Mat disp_float = _matcher->compute(gray_L, gray_R);

    auto t1 = clk::now();

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
    dumpKeyframe(item, disp_float, f_rect, cx_rect, cy_rect);

    // Run mesh estimator based on method — dense pipeline is fully independent
    // of the sparse Mesh3D; injectDensePoints() is never called.
    DenseMesh dense_mesh;
    if (_anchors && fusedMethod()) {
        integrateAnchoredFused(item, disp_float, f_rect, cx_rect, cy_rect, dense_mesh);
    } else if (_cfg.mesh_method == "gp" && _gp_map) {
        integrateGlobalMap(item, disp_float, f_rect, cx_rect, cy_rect, point_cloud, dense_mesh);
    } else if (_cfg.mesh_method == "vdbgpdf" && _vdb_map) {
        integrateVDBGPDF(item, disp_float, f_rect, cx_rect, cy_rect, dense_mesh);
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

    if (_anchors && !fusedMethod()) {
        // Per-keyframe methods: the keyframe is its own anchor; its geometry (the point cloud for "none", 5 cm
        // voxels) is moved into its camera frame, exactly (the estimators build it at the keyframe's pose)
        DenseMesh g;
        if (_cfg.mesh_method == "none") {
            std::unordered_set<long long> seen;
            for (const auto& p : point_cloud) {
                const Eigen::Vector3i k = (p / 0.05).array().floor().cast<int>();
                const long long key = (static_cast<long long>(k.x()) * 73856093LL) ^
                                      (static_cast<long long>(k.y()) * 19349663LL) ^
                                      (static_cast<long long>(k.z()) * 83492791LL);
                if (seen.insert(key).second)
                    g.vertices.push_back(p);
            }
        } else {
            g = dense_mesh;
        }
        _anchors->addKeyframe(item.frame ? item.frame->getTimestamp() : 0ULL, item.segment, item.epoch,
                              item.T_w_rectcam, transformMesh(g, item.T_w_rectcam.inverse()));
        _anchors->keyframeDone(nullptr);
        dense_mesh = _anchors->globalMesh();
    }

    auto t4 = clk::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    std::cout << "[DenseMesh] " << _matcher->name() << "=" << ms(t0,t1) << "ms"
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

void MarginalDepthInjector::integrateVDBGPDF(const QueueItem& item, const cv::Mat& disp_float, double f_rect,
                                             double cx_rect, double cy_rect, DenseMesh& dense_mesh) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const int str = std::max(1, _cfg.vdbgpdf_stride);
    std::vector<Eigen::Vector3d> points_world;
    points_world.reserve((disp_float.rows / str) * (disp_float.cols / str));
    for (int v = 0; v < disp_float.rows; v += str) {
        const float* row_ptr = disp_float.ptr<float>(v);
        for (int u = 0; u < disp_float.cols; u += str) {
            const float disp = row_ptr[u];
            if (disp <= 0.0f || !std::isfinite(disp))
                continue;
            const double Z = f_rect * _baseline / static_cast<double>(disp);
            if (Z <= 0.0 || Z > _cfg.max_depth || !std::isfinite(Z))
                continue;
            points_world.push_back(item.T_w_rectcam *
                                   Eigen::Vector3d((u - cx_rect) * Z / f_rect, (v - cy_rect) * Z / f_rect, Z));
        }
    }
    _vdb_map->integrate(points_world, item.T_w_rectcam.translation());
    ++_integrated;
    const auto t1 = clk::now();
    if (_vdb_log.is_open()) {
        const Eigen::Quaterniond q(item.T_w_rectcam.linear());
        const Eigen::Vector3d& t = item.T_w_rectcam.translation();
        _vdb_log << std::setprecision(10) << (item.frame ? item.frame->getTimestamp() : 0ULL) << "," << _integrated
                 << "," << points_world.size() << ","
                 << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "," << t.x() << ","
                 << t.y() << "," << t.z() << "," << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
        _vdb_log.flush();
    }

    // Marching cubes over the whole map is the expensive part; run it every vdbgpdf_mesh_every keyframes
    if (_vdb_mesh.faces.empty() || (_cfg.vdbgpdf_mesh_every > 0 && _integrated % _cfg.vdbgpdf_mesh_every == 0)) {
        _vdb_mesh = _vdb_map->mesh();
        if (_cfg.vdbgpdf_mesh_every > 0 &&
            !writeDenseMeshPly(_vdb_mesh, _cfg.vdbgpdf_mesh_path, "SaDVIO dense VDB-GPDF map"))
            std::cerr << "[DenseMesh] could not write " << _cfg.vdbgpdf_mesh_path << std::endl;
    }
    dense_mesh = _vdb_mesh;
    const auto ms = [](auto a, auto b) { return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count(); };
    std::cout << "[DenseMesh] vdbgpdf kf=" << _integrated << " pts=" << points_world.size()
              << " integrate=" << ms(t0, t1) << "ms mesh=" << ms(t1, clk::now()) << "ms faces=" << _vdb_mesh.faces.size()
              << std::endl;
}

void MarginalDepthInjector::integrateAnchoredFused(const QueueItem& item, const cv::Mat& disp_float, double f_rect,
                                                   double cx_rect, double cy_rect, DenseMesh& dense_mesh) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    const bool vdb = _cfg.mesh_method == "vdbgpdf";
    // Same point selection as the global maps: vdbgpdf every vdbgpdf_stride-th pixel up to max_depth, gp every
    // pixel up to the GP's max depth; in the rectified camera frame
    const int str         = vdb ? std::max(1, _cfg.vdbgpdf_stride) : 1;
    const double max_z    = vdb ? _cfg.max_depth : _gp_cfg.max_depth;
    std::vector<Eigen::Vector3d> points_cam;
    points_cam.reserve((disp_float.rows / str) * (disp_float.cols / str));
    for (int v = 0; v < disp_float.rows; v += str) {
        const float* row_ptr = disp_float.ptr<float>(v);
        for (int u = 0; u < disp_float.cols; u += str) {
            const float disp = row_ptr[u];
            if (disp <= 0.0f || !std::isfinite(disp))
                continue;
            const double Z = f_rect * _baseline / static_cast<double>(disp);
            if (Z <= 0.0 || Z > max_z || !std::isfinite(Z))
                continue;
            points_cam.emplace_back((u - cx_rect) * Z / f_rect, (v - cy_rect) * Z / f_rect, Z);
        }
    }

    const AnchoredMaps::Mesher mesher = [](AnchoredMaps::Anchor& a) {
        return a.vdb ? a.vdb->mesh() : a.gp ? a.gp->mesh() : a.mesh;
    };
    Eigen::Affine3d T_a_c;
    AnchoredMaps::Anchor& a = _anchors->submapFor(item.frame ? item.frame->getTimestamp() : 0ULL, item.segment,
                                                  item.epoch, item.T_w_rectcam, T_a_c, mesher);
    if (vdb) {
        if (!a.vdb)
            a.vdb = std::make_unique<VDBGPDFMap>(_vdb_cfg);
        std::vector<Eigen::Vector3d> points_anchor;
        points_anchor.reserve(points_cam.size());
        for (const auto& p : points_cam)
            points_anchor.push_back(T_a_c * p);
        a.vdb->integrate(points_anchor, T_a_c.translation());
    } else {
        if (!a.gp)
            a.gp = std::make_unique<GPGlobalMap>(_gp_global_cfg);
        a.gp->integrate(points_cam, T_a_c); // with dense_gp_register: against this submap only
    }
    ++_integrated;
    const auto t1 = clk::now();
    if (_vdb_log.is_open()) {
        const Eigen::Quaterniond q(item.T_w_rectcam.linear());
        const Eigen::Vector3d& t = item.T_w_rectcam.translation();
        _vdb_log << std::setprecision(10) << (item.frame ? item.frame->getTimestamp() : 0ULL) << "," << _integrated
                 << "," << points_cam.size() << ","
                 << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << "," << t.x() << ","
                 << t.y() << "," << t.z() << "," << q.x() << "," << q.y() << "," << q.z() << "," << q.w() << "\n";
        _vdb_log.flush();
    }
    // Meshing of the open submap only (closed ones keep their mesh), then the global mesh, every save_every keyframes
    _anchors->keyframeDone([&](AnchoredMaps::Anchor& o) { o.mesh = mesher(o); });
    dense_mesh = _anchors->globalMesh();
    const auto ms = [](auto x, auto y) { return std::chrono::duration_cast<std::chrono::milliseconds>(y - x).count(); };
    std::cout << "[DenseMesh] " << _cfg.mesh_method << " submap " << a.id << " kf=" << _integrated
              << " pts=" << points_cam.size() << " integrate=" << ms(t0, t1) << "ms mesh=" << ms(t1, clk::now())
              << "ms faces=" << dense_mesh.faces.size() << std::endl;
}

void MarginalDepthInjector::dumpKeyframe(const QueueItem& item, const cv::Mat& disp_float, double f_rect,
                                         double cx_rect, double cy_rect) {
    static const char* dir = std::getenv("EXECO_DENSE_DUMP");
    if (!dir || !item.frame)
        return;
    // Points as the VDB map takes them (every vdbgpdf_stride-th pixel, depth up to max_depth), rectified camera
    // frame, float32. File: uint64 ts, int32 segment, int32 epoch, double[16] T_w_rectcam, double[16] T_w_f
    // (column-major, odometry at queue time), uint32 n, float32[3 n]
    const int str = std::max(1, _cfg.vdbgpdf_stride);
    std::vector<float> pts;
    for (int v = 0; v < disp_float.rows; v += str) {
        const float* row_ptr = disp_float.ptr<float>(v);
        for (int u = 0; u < disp_float.cols; u += str) {
            const float disp = row_ptr[u];
            if (disp <= 0.0f || !std::isfinite(disp))
                continue;
            const double Z = f_rect * _baseline / static_cast<double>(disp);
            if (Z <= 0.0 || Z > _cfg.max_depth || !std::isfinite(Z))
                continue;
            pts.push_back(static_cast<float>((u - cx_rect) * Z / f_rect));
            pts.push_back(static_cast<float>((v - cy_rect) * Z / f_rect));
            pts.push_back(static_cast<float>(Z));
        }
    }
    std::filesystem::create_directories(dir);
    const unsigned long long ts = item.frame->getTimestamp();
    std::ofstream f(std::string(dir) + "/kf_" + std::to_string(ts) + ".bin", std::ios::binary | std::ios::trunc);
    const int32_t seg = item.segment, ep = item.epoch;
    const uint32_t n  = static_cast<uint32_t>(pts.size() / 3);
    f.write(reinterpret_cast<const char*>(&ts), sizeof(ts));
    f.write(reinterpret_cast<const char*>(&seg), sizeof(seg));
    f.write(reinterpret_cast<const char*>(&ep), sizeof(ep));
    f.write(reinterpret_cast<const char*>(item.T_w_rectcam.matrix().data()), 16 * sizeof(double));
    f.write(reinterpret_cast<const char*>(item.T_w_f.matrix().data()), 16 * sizeof(double));
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(reinterpret_cast<const char*>(pts.data()), pts.size() * sizeof(float));
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
