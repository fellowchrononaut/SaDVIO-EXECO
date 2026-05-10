#include "isaeslam/stereo/MarginalDepthInjector.h"
#include "isaeslam/data/sensors/ASensor.h"
#include <opencv2/core/eigen.hpp>
#include <iostream>

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
                                    1, 0, 10, 100, 32,
                                    cv::StereoSGBM::MODE_SGBM_3WAY);

    // Start worker thread
    _worker = std::thread(&MarginalDepthInjector::workerLoop, this);
}

MarginalDepthInjector::~MarginalDepthInjector() {
    _running = false;
    _q_cv.notify_all();
    if (_worker.joinable())
        _worker.join();
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
        _pending = std::move(item); // drop previous if not consumed
    }
    _q_cv.notify_one();
}

void MarginalDepthInjector::workerLoop() {
    while (_running) {
        QueueItem item;
        {
            std::unique_lock<std::mutex> lock(_q_mtx);
            _q_cv.wait(lock, [this]{ return _pending.valid || !_running; });
            if (!_running) break;
            std::swap(item, _pending);
            _pending.valid = false;
        }
        processItem(item);
    }
}

void MarginalDepthInjector::processItem(const QueueItem& item) {
    if (!item.valid || item.img_L.empty() || item.img_R.empty())
        return;

    // Limit OpenCV thread usage during SGBM
    cv::setNumThreads(_cfg.sgbm_num_threads);

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

    // Convert to float (SGBM output is 16-bit fixed-point, divide by 16)
    cv::Mat disp_float;
    disp_raw.convertTo(disp_float, CV_32F, 1.0 / 16.0);

    // Scale intrinsics if images were rescaled
    double f_rect  = _f_rect  * _cfg.scale_factor;
    double cx_rect = _cx_rect * _cfg.scale_factor;
    double cy_rect = _cy_rect * _cfg.scale_factor;

    // Convert disparity to metric depth (float32, metres)
    cv::Mat depth_img = cv::Mat::zeros(disp_float.size(), CV_32F);
    for (int v = 0; v < disp_float.rows; ++v) {
        const float* d_row = disp_float.ptr<float>(v);
        float*       z_row = depth_img.ptr<float>(v);
        for (int u = 0; u < disp_float.cols; ++u) {
            if (d_row[u] > 0.f)
                z_row[u] = static_cast<float>(_f_rect * _baseline / d_row[u]);
        }
    }

    // Run GP mesh estimator
    GPMeshEstimator gp_estimator(_gp_cfg);
    DenseMesh dense_mesh = gp_estimator.estimate(disp_float,
                                                  f_rect, _baseline,
                                                  cx_rect, cy_rect,
                                                  item.T_w_rectcam);

    // Inject dense points into the sparse Mesh3D point cloud
    if (item.mesh && !dense_mesh.vertices.empty()) {
        item.mesh->injectDensePoints(dense_mesh.vertices);
    }

    // Store result for the ROS visualizer to poll
    {
        std::lock_guard<std::mutex> lock(_result_mtx);
        _latest_result.depth_img = depth_img.clone();
        _latest_result.mesh      = dense_mesh;
        _latest_result.valid     = true;
    }
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
