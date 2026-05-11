
#include "isaeslam/slamCore.h"
#include <opencv2/core.hpp>

namespace isae {

bool SLAMBiMono::init() {

    // get first frame and set keyframe
    _frame = _slam_param->getDataProvider()->next();
    while (_frame->getSensors().empty()) {
        _frame = _slam_param->getDataProvider()->next();
    }

    // Prior on the first frame, it is set as the origin
    _frame->setWorld2FrameTransform(Eigen::Affine3d::Identity());
    _frame->setPrior(Eigen::Affine3d::Identity(), 100 * Vector6d::Ones());

    // detect all features on all sensors
    detectFeatures(_frame->getSensors().at(0));

    // Track features in frame
    trackFeatures(_frame->getSensors().at(0),
                  _frame->getSensors().at(1),
                  _matches_in_frame,
                  _matches_in_frame_lmk,
                  _frame->getSensors().at(0)->getFeatures());

    // Filter matches in frame
    _matches_in_frame = epipolarFiltering(_frame->getSensors().at(0), _frame->getSensors().at(1), _matches_in_frame);

    // init the velocity
    _6d_velocity = 0.00001 * Vector6d::Ones();

    // init first landmarks
    initLandmarks(_frame);
    _slam_param->getOptimizerFront()->landmarkOptimization(_frame);

    // Create the 3D mesh
    if (_slam_param->_config.mesh3D) {
        _mesher->addNewKF(_frame);
    }

    // Ignore features that were not triangulated
    cleanFeatures(_frame);
    detectFeatures(_frame->getSensors().at(0));

    profiling();

    // Construct dense stereo injector if enabled
    if (_slam_param->_config.dense_depth) {
        auto cam_cfgs = _slam_param->getDataProvider()->getCamConfigs();
        if (cam_cfgs.size() >= 2) {
            auto& cL = *cam_cfgs.at(0);
            auto& cR = *cam_cfgs.at(1);
            Eigen::Affine3d T_right_in_left = cL.T_s_f * cR.T_s_f.inverse();

            auto img = _frame->getSensors().at(0)->getRawData();
            cv::Size imsz(img.cols, img.rows);

            MarginalDepthConfig dcfg;
            dcfg.num_disparities  = _slam_param->_config.stereo_depth_num_disp;
            dcfg.block_size       = _slam_param->_config.stereo_depth_block_size;
            dcfg.scale_factor     = _slam_param->_config.stereo_depth_scale;
            dcfg.stride           = _slam_param->_config.stereo_depth_stride;
            dcfg.max_depth        = _slam_param->_config.stereo_depth_max_depth;
            dcfg.mesh_method      = _slam_param->_config.dense_mesh_method;
            dcfg.zncc_threshold   = _slam_param->_config.ZNCC_tsh;
            dcfg.max_length_threshold = _slam_param->_config.max_length_tsh;
            dcfg.uniqueness_ratio    = _slam_param->_config.stereo_depth_uniqueness_ratio;
            dcfg.speckle_window_size = _slam_param->_config.stereo_depth_speckle_window_size;
            dcfg.speckle_range       = _slam_param->_config.stereo_depth_speckle_range;
            dcfg.disp12_max_diff     = _slam_param->_config.stereo_depth_disp12_max_diff;
            dcfg.pre_filter_cap      = _slam_param->_config.stereo_depth_pre_filter_cap;
            dcfg.gp_cell_size        = _slam_param->_config.dense_gp_cell_size;
            dcfg.gp_num_test         = _slam_param->_config.dense_gp_num_test;
            dcfg.gp_min_pts_per_cell = _slam_param->_config.dense_gp_min_pts_per_cell;
            dcfg.gp_kernel_length    = _slam_param->_config.dense_gp_kernel_length;
            dcfg.gp_variance_sensor  = _slam_param->_config.dense_gp_variance_sensor;
            dcfg.gp_max_variance     = _slam_param->_config.dense_gp_max_variance;
            dcfg.gp_full_cover       = _slam_param->_config.dense_gp_full_cover;
            dcfg.gp_eigen_1          = _slam_param->_config.dense_gp_eigen_1;
            dcfg.gp_eigen_2          = _slam_param->_config.dense_gp_eigen_2;
            dcfg.gp_eigen_3          = _slam_param->_config.dense_gp_eigen_3;
            dcfg.pd_steiner_spacing  = _slam_param->_config.dense_pd_steiner_spacing;
            dcfg.pd_lambda           = _slam_param->_config.dense_pd_lambda;
            dcfg.pd_num_iterations   = _slam_param->_config.dense_pd_num_iterations;
            dcfg.pd_tau              = _slam_param->_config.dense_pd_tau;
            dcfg.pd_sigma            = _slam_param->_config.dense_pd_sigma;
            dcfg.pd_theta            = _slam_param->_config.dense_pd_theta;
            dcfg.pd_min_depth        = _slam_param->_config.dense_pd_min_depth;

            _depth_injector = std::make_shared<MarginalDepthInjector>(
                cL.K, cL.d, cR.K, cR.d, T_right_in_left, imsz, dcfg);
        }
    }

    // Send frame to optimizer
    _frame_to_optim = _frame;
    _is_init        = true;
    _successive_fails = 0;
    _nkeyframes++;

    return true;
}

bool SLAMBiMono::frontEndStep() {

    // Get next frame
    _frame = _slam_param->getDataProvider()->next();

    // Ignore frames without images
    if (_frame->getSensors().empty())
        return true;
    _nframes++;

    // Predict pose with constant velocity model
    double dt = (_frame->getTimestamp() - getLastKF()->getTimestamp()) * 1e-9;
    Eigen::Affine3d T_f_w =
        geometry::se3_Vec6dtoRT(_6d_velocity * dt).inverse() * getLastKF()->getWorld2FrameTransform();
    _frame->setWorld2FrameTransform(T_f_w);

    // Detect all features (only if we use the matcher)
    isae::timer::tic();
    if (_slam_param->_config.tracker == "matcher") {
        detectFeatures(_frame->getSensors().at(0));
    }
    _avg_detect_t = (_avg_detect_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    // Match or track the features in time
    uint nmatches_in_time;
    isae::timer::tic();
    if (_slam_param->_config.tracker == "klt") {
        nmatches_in_time = trackFeatures(getLastKF()->getSensors().at(0),
                                         _frame->getSensors().at(0),
                                         _matches_in_time,
                                         _matches_in_time_lmk,
                                         getLastKF()->getSensors().at(0)->getFeatures());
    } else {
        nmatches_in_time = matchFeatures(getLastKF()->getSensors().at(0),
                                         _frame->getSensors().at(0),
                                         _matches_in_time,
                                         _matches_in_time_lmk,
                                         getLastKF()->getSensors().at(0)->getFeatures());
    }

    _avg_matches_time = (_avg_matches_time * (_nframes - 1) + nmatches_in_time) / _nframes;
    _avg_match_time_t = (_avg_match_time_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    // Get P3d from n-1 matched features and estimate 3D pose from 2D (n)/3D (n-1) matchings
    // to predict pose. Also remove outliers from tracks_in_time vector
    isae::timer::tic();
    bool good_it   = predict(_frame);
    _avg_predict_t = (_avg_predict_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    if (good_it) {
        _successive_fails = 0;

        // Epipolar Filtering for matches in time
        isae::timer::tic();
        int removed_matching_nb = _matches_in_time["pointxd"].size();
        _matches_in_time =
            epipolarFiltering(getLastKF()->getSensors().at(0), _frame->getSensors().at(0), _matches_in_time);
        removed_matching_nb -= _matches_in_time["pointxd"].size();
        _removed_feat = (_removed_feat * (_nframes - 1) + removed_matching_nb) / _nframes;
        _avg_filter_t = (_avg_filter_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

        // Remove Outliers in case of klt
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            outlierRemoval();
            _avg_clean_t = (_avg_clean_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;
        }

        // Update tracked landmarks
        updateLandmarks(_matches_in_time_lmk);

        // Single Frame ESKF Update
        isae::timer::tic();

        Eigen::MatrixXd cov;
        Eigen::Affine3d T_last_curr, T_w_f;
        T_last_curr = getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform();
        ESKFEstimator eskf;
        eskf.estimateTransformBetween(getLastKF(), _frame, _matches_in_time_lmk["pointxd"], T_last_curr, cov);
        T_w_f = getLastKF()->getFrame2WorldTransform() * T_last_curr;
        _frame->setdTCov(cov);
        _frame->setWorld2FrameTransform(T_w_f.inverse());

        _avg_frame_opt_t = (_avg_frame_opt_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;
        _lmk_inmap       = (_lmk_inmap * (_nframes - 1) + _frame->getLandmarks()["pointxd"].size()) / _nframes;

        // Compute velocity and motion model
        _6d_velocity =
            (geometry::se3_RTtoVec6d(getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform())) / dt;
    } else {

        // If the prediction is wrong, we reinitialize the odometry from the last KF:
        // - A KF is voted
        // - All matches in time are removed
        // Can be improved: redetect new points, retrack old features....

        _successive_fails++;
        outlierRemoval();
        _frame->setKeyFrame();
    }


    if (shouldInsertKeyframe(_frame)) {

        // Frame is added
        _nkeyframes++;

        // Repopulate in the case of klt tracking
        typed_vec_features new_features;
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            new_features  = detectFeatures(_frame->getSensors().at(0));
            _avg_detect_t = (_avg_detect_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        }

        // Recover Map Landmark
        isae::timer::tic();
        uint resu = recoverFeatureFromMapLandmarks(_frame->getSensors().at(0));
        
        _avg_lmk_resur_t = (_avg_lmk_resur_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        _avg_resur_lmk   = (_avg_lmk_resur_t * (_nkeyframes - 1) + resu) / _nkeyframes;

        // Track features in frame
        isae::timer::tic();
        uint nmatches_in_frame = trackFeatures(_frame->getSensors().at(0),
                                               _frame->getSensors().at(1),
                                               _matches_in_frame,
                                               _matches_in_frame_lmk,
                                               _frame->getSensors().at(0)->getFeatures());

        // Epipolar Filtering for matches in frame
        _matches_in_frame =
            epipolarFiltering(_frame->getSensors().at(0), _frame->getSensors().at(1), _matches_in_frame);
        _matches_in_frame_lmk =
            epipolarFiltering(_frame->getSensors().at(0), _frame->getSensors().at(1), _matches_in_frame_lmk);

        // Update tracked landmarks
        updateLandmarks(_matches_in_frame_lmk);

        _avg_matches_frame = (_avg_matches_frame * (_nkeyframes - 1) + nmatches_in_frame) / _nkeyframes;
        _avg_match_frame_t = (_avg_match_frame_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

        // Landmark Initialization:
        // - Triangulate new points : LR + (n-1) / n
        // - Optimize points only because optimal mid-point is not optimal for LM
        // - Reject outliers with reprojection error
        isae::timer::tic();
        initLandmarks(_frame);
        _slam_param->getOptimizerFront()->landmarkOptimization(_frame);
        _avg_lmk_init_t = (_avg_lmk_init_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

        // Wait the end of optim
        while (_frame_to_optim != nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        _frame_to_optim = _frame;

    } else {
        // If no KF is voted, the frame is discarded and the landmarks are cleaned
        _frame->cleanLandmarks();
    }

    // Init the SLAM again in case of successive failures or if the frame is too far from the last KF
    if ((getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform()).translation().norm() > 10 ||
        (_successive_fails > 5)) {

        _is_init = false;
        _local_map->reset();
        _slam_param->getOptimizerBack()->resetMarginalization();

        return true;
    }

    // Send the frame to the viewer
    _frame_to_display = _frame;

    return true;
}

bool SLAMBiMono::backEndStep() {

    // Optimize when a frame is declared as optimizable
    if (_frame_to_optim) {

        // Add frame to local map
        _local_map->addFrame(_frame_to_optim);
        _frame_to_optim->setKeyFrame();

        // 3D Mesh update
        if (_slam_param->_config.mesh3D) {
            _mesher->addNewKF(_frame_to_optim);
            _mesh_to_display = _mesher->_mesh_3d;
        }

        // Marginalization (+ sparsification) of the last frame
        isae::timer::tic();
        while (_local_map->getMarginalizationFlag()) {
            auto frame_to_marg = _local_map->getFrames().at(0);
            auto frame_to_keep = _local_map->getFrames().at(1);

            // Queue frame for async dense mesh BEFORE discardLastFrame() frees images
            if (_depth_injector)
                _depth_injector->queueFrame(frame_to_marg, nullptr);

            if (_slam_param->_config.marginalization == 1)
                _slam_param->getOptimizerBack()->marginalize(frame_to_marg,
                                                             frame_to_keep,
                                                             _slam_param->_config.sparsification == 1);

            // Uncomment below to enable global map
            // _global_map->addFrame(_local_map->getFrames().at(0));

            _map_mutex.lock();
            _local_map->discardLastFrame();
            _map_mutex.unlock();
        }
        _avg_marg_t = (_avg_marg_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

        // Optimize Local Map
        isae::timer::tic();
        _slam_param->getOptimizerBack()->localMapBA(_local_map, _local_map->getFixedFrameNumber());
        _avg_wdw_opt_t = (_avg_wdw_opt_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        profiling();

        // Reset frame to optim
        _frame_to_optim = nullptr;

        // Send the local map to the viewer
        _local_map_to_display = _local_map;
    }

    return true;
}

} // namespace isae
