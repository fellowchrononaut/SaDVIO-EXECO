#include "isaeslam/slamCore.h"

namespace isae {

bool SLAMNonOverlappingFovVIO::init() {

    // monovio's initialization on camera 0: the map is metric and gravity-aligned when it succeeds
    if (!SLAMMonoVIO::init())
        return false;

    // Camera 1 starts tracking from the last KF
    _matches_in_time_cam1.clear();
    _matches_in_time_cam1_lmk.clear();
    detectFeatures(getLastKF()->getSensors().at(1));
    return true;
}

bool SLAMNonOverlappingFovVIO::frontEndStep() {

    // Get next frame
    _frame = nextFrame();

    // Process IMU data
    if (_frame->getIMU()) {
        _frame->getIMU()->setLastIMU(_last_IMU);
        _frame->getIMU()->setLastKF(getLastKF());

        // Ignore measurement if it failed
        if (_frame->getIMU()->processIMU()) {
            _last_IMU = _frame->getIMU();
        } else {
            _n_rejected_imu++;
            _frame->clearIMU(); // never keep an unprocessed IMU measurement
        }
    }

    // Skip if there is no images in the frame
    if (_frame->getSensors().empty())
        return true;

    _nframes++;

    // Estimate the transformation between frames using IMU for the rotation and cst velocity for translation
    double dt             = (_frame->getTimestamp() - getLastKF()->getTimestamp()) * 1e-9;
    Eigen::Affine3d dT    = getLastKF()->getWorld2FrameTransform() * _last_IMU->_T_w_f_imu;
    Eigen::Affine3d T_f_w = dT.inverse() * getLastKF()->getWorld2FrameTransform();
    _frame->setWorld2FrameTransform(T_f_w);

    // Detect all features (only if we use the matcher)
    isae::timer::tic();
    if (_slam_param->_config.tracker == "matcher") {
        detectFeatures(_frame->getSensors().at(0));
        detectFeatures(_frame->getSensors().at(1));
    }
    _avg_detect_t = (_avg_detect_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    // Match or track the features in time, on each camera
    uint nmatches_in_time;
    isae::timer::tic();
    if (_slam_param->_config.tracker == "klt") {
        nmatches_in_time = trackFeatures(getLastKF()->getSensors().at(0),
                                         _frame->getSensors().at(0),
                                         _matches_in_time,
                                         _matches_in_time_lmk,
                                         getLastKF()->getSensors().at(0)->getFeatures());
        nmatches_in_time += trackFeatures(getLastKF()->getSensors().at(1),
                                          _frame->getSensors().at(1),
                                          _matches_in_time_cam1,
                                          _matches_in_time_cam1_lmk,
                                          getLastKF()->getSensors().at(1)->getFeatures());
    } else {
        nmatches_in_time = matchFeatures(getLastKF()->getSensors().at(0),
                                         _frame->getSensors().at(0),
                                         _matches_in_time,
                                         _matches_in_time_lmk,
                                         getLastKF()->getSensors().at(0)->getFeatures());
        nmatches_in_time += matchFeatures(getLastKF()->getSensors().at(1),
                                          _frame->getSensors().at(1),
                                          _matches_in_time_cam1,
                                          _matches_in_time_cam1_lmk,
                                          getLastKF()->getSensors().at(1)->getFeatures());
    }

    _avg_matches_time = (_avg_matches_time * (_nframes - 1) + nmatches_in_time) / _nframes;
    _avg_match_time_t = (_avg_match_time_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    // Essential RANSAC on the matches in time and PnP RANSAC on the landmark matches remove the outliers, per camera
    // (these estimators take the extrinsics of one camera)
    isae::timer::tic();
    Eigen::MatrixXd cov;
    Eigen::Affine3d dT_t;
    EpipolarPoseEstimator est;
    PnPPoseEstimator est_pnp;
    est.estimateTransformBetween(getLastKF(), _frame, _matches_in_time["pointxd"], dT_t, cov);
    est_pnp.estimateTransformBetween(getLastKF(), _frame, _matches_in_time_lmk["pointxd"], dT_t, cov);
    est.estimateTransformBetween(getLastKF(), _frame, _matches_in_time_cam1["pointxd"], dT_t, cov);
    est_pnp.estimateTransformBetween(getLastKF(), _frame, _matches_in_time_cam1_lmk["pointxd"], dT_t, cov);
    _avg_predict_t = (_avg_predict_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

    // If enough tracks on the two cameras together, perform the ESKF update
    const size_t n_tracks = _matches_in_time_lmk["pointxd"].size() + _matches_in_time["pointxd"].size() +
                            _matches_in_time_cam1_lmk["pointxd"].size() + _matches_in_time_cam1["pointxd"].size();
    if (n_tracks > 5) {
        _successive_fails = 0;
        _last_visual_ts   = _frame->getTimestamp();

        // Epipolar Filtering for matches in time
        isae::timer::tic();
        int removed_matching_nb = _matches_in_time["pointxd"].size() + _matches_in_time_cam1["pointxd"].size();
        _matches_in_time =
            epipolarFiltering(getLastKF()->getSensors().at(0), _frame->getSensors().at(0), _matches_in_time);
        _matches_in_time_cam1 =
            epipolarFiltering(getLastKF()->getSensors().at(1), _frame->getSensors().at(1), _matches_in_time_cam1);
        removed_matching_nb -= _matches_in_time["pointxd"].size() + _matches_in_time_cam1["pointxd"].size();
        _removed_feat = (_removed_feat * (_nframes - 1) + removed_matching_nb) / _nframes;
        _avg_filter_t = (_avg_filter_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;

        // Remove Outliers in case of klt
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            outlierRemovalBothCameras();
            _avg_clean_t = (_avg_clean_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;
        }

        // Update tracked landmarks
        updateLandmarks(_matches_in_time_lmk);
        updateLandmarks(_matches_in_time_cam1_lmk);

        // Single Frame ESKF Update over the landmarks of both cameras
        isae::timer::tic();
        vec_match lmk_matches = _matches_in_time_lmk["pointxd"];
        lmk_matches.insert(lmk_matches.end(),
                           _matches_in_time_cam1_lmk["pointxd"].begin(),
                           _matches_in_time_cam1_lmk["pointxd"].end());

        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
        Eigen::Affine3d T_last_curr, T_w_f;
        bool eskf_ok = false; // the ESKF update was applied (logged with its covariance)
        T_last_curr  = getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform();
        ESKFEstimator eskf;
        const Eigen::Affine3d T_pred = T_last_curr;
        eskf_ok = eskf.estimateTransformBetweenCameras(getLastKF(), _frame, lmk_matches, T_last_curr, cov);
        if (!plausibleUpdate(T_pred, T_last_curr)) {
            std::cerr << "ESKF update rejected (implausible jump), prediction kept" << std::endl;
            T_last_curr = T_pred;
            eskf_ok     = false;
        }
        T_w_f = getLastKF()->getFrame2WorldTransform() * T_last_curr;
        _frame->setdTCov(cov);
        logFrontCovariance(getLastKF(), T_last_curr, cov, eskf_ok);
        _frame->setWorld2FrameTransform(T_w_f.inverse());

        _avg_frame_opt_t = (_avg_frame_opt_t * (_nframes - 1) + isae::timer::silentToc()) / _nframes;
        _lmk_inmap       = (_lmk_inmap * (_nframes - 1) + _frame->getLandmarks()["pointxd"].size()) / _nframes;

        // Compute velocity and motion model
        _6d_velocity =
            (geometry::se3_RTtoVec6d(getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform())) / dt;

    } else {

        // Not enough tracks on either camera: the pose comes from the IMU, a KF is voted and both cameras detect again
        std::cout << "Not enough tracked Features" << std::endl;
        _successive_fails++;
        _frame->setWorld2FrameTransform(_last_IMU->_T_w_f_imu.inverse());
        outlierRemovalBothCameras();
        detectFeatures(_frame->getSensors().at(0));
        detectFeatures(_frame->getSensors().at(1));
        _frame->setKeyFrame();
    }

    if (dt > 0.5)
        _frame->setKeyFrame();

    if (shouldInsertKeyframeBothCameras(_frame)) {
        _nkeyframes++;

        // An IMU is set to this KF if there is not one already
        if (!_frame->getIMU()) {

            // Check if the _last_imu is in the future
            while (_last_IMU->_timestamp_imu > _frame->getTimestamp()) {
                _last_IMU = _last_IMU->getLastIMU();
            }

            std::shared_ptr<IMU> imu_kf = std::make_shared<IMU>(
                _slam_param->getDataProvider()->getIMUConfig(), _last_IMU->getAcc(), _last_IMU->getGyr());
            imu_kf->setLastIMU(_last_IMU);
            imu_kf->setLastKF(getLastKF());
            _frame->setIMU(imu_kf, _last_IMU->getFrame2SensorTransform());

            // Process IMU but without updating the frame pose
            Eigen::Affine3d T_f_w = _frame->getWorld2FrameTransform();
            _frame->getIMU()->processIMU();
            _frame->setWorld2FrameTransform(T_f_w);
        }

        // Landmark Initialization on both cameras:
        // - Triangulate new points over time (n-1) / n, through each camera's extrinsics
        // - Optimize points only because optimal mid-point is not optimal for LM
        // - Reject outliers with reprojection error
        isae::timer::tic();
        if (_parallax > 0.5) // Ignore landmark initialization if not enough parallax (mono)
            initLandmarksBothCameras(_frame);
        _slam_param->getOptimizerFront()->landmarkOptimization(_frame);
        _avg_lmk_init_t = (_avg_lmk_init_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;

        // Repopulate in the case of klt tracking
        if (_slam_param->_config.tracker == "klt") {
            isae::timer::tic();
            detectFeatures(_frame->getSensors().at(0));
            detectFeatures(_frame->getSensors().at(1));
            _avg_detect_t = (_avg_detect_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        }

        // Recover Map Landmarks, on both cameras (a lens can recover what the other one mapped)
        isae::timer::tic();
        uint resu = recoverFeatureFromMapLandmarks(_frame->getSensors().at(0));
        resu += recoverFeatureFromMapLandmarks(_frame->getSensors().at(1));

        _avg_lmk_resur_t = (_avg_lmk_resur_t * (_nkeyframes - 1) + isae::timer::silentToc()) / _nkeyframes;
        _avg_resur_lmk   = (_avg_lmk_resur_t * (_nkeyframes - 1) + resu) / _nkeyframes;

        // Wait the end of optim
        waitBackEnd();
        _frame_to_optim = _frame;
        logKfFeatures(_frame);
        _parallax_to_optim = _parallax;
        _last_IMU          = _frame_to_optim->getIMU();

    } else {
        // If no KF is voted, the frame is discarded and the landmarks are cleaned
        _frame->cleanLandmarks();
    }

    // Init the SLAM again in case of successive failures or if the frame is too far from the last KF
    const bool too_far =
        (getLastKF()->getWorld2FrameTransform() * _frame->getFrame2WorldTransform()).translation().norm() > 10;
    if (too_far || (_successive_fails > maxLostFrames())) {
        std::cout << "Reinitializing SLAM after " << _successive_fails << " successive fails or too far from last KF"
                  << std::endl;
        waitBackEnd(); // the back end must not add a pending KF to the reset map

        // After a visual dropout the next initialization can start from the state the IMU carried (see SLAMBiMonoVIO)
        if (too_far)
            _carried.valid = false;
        else
            captureCarriedState(_last_IMU);
        logWindowBeforeReset();
        _is_init = false;
        _local_map->reset();
        _slam_param->getOptimizerBack()->resetMarginalization(); // its prior refers to the frames just discarded

        return true;
    }

    // Send the frame to the viewer
    _frame_to_display = _frame;
    logPerFrame(_frame);

    return true;
}

void SLAMNonOverlappingFovVIO::outlierRemovalBothCameras() {

    // Keep the features of _frame that are tracked in time without being outliers (SLAMCore::outlierRemoval, per
    // camera)
    const std::vector<std::pair<typed_vec_match *, typed_vec_match *>> tracks = {
        {&_matches_in_time, &_matches_in_time_lmk}, {&_matches_in_time_cam1, &_matches_in_time_cam1_lmk}};
    for (size_t c = 0; c < tracks.size(); c++) {
        std::shared_ptr<ImageSensor> cam = _frame->getSensors().at(c);
        isae::typed_vec_features clean_features;
        for (auto tf : cam->getFeatures()) {
            for (const typed_vec_match *m : {tracks[c].first, tracks[c].second}) {
                auto it = m->find(tf.first);
                if (it == m->end())
                    continue;
                for (auto &match : it->second)
                    if (!match.second->isOutlier())
                        clean_features[tf.first].push_back(match.second);
            }
            cam->purgeFeatures(tf.first);
            cam->addFeatures(tf.first, clean_features[tf.first]);
        }
    }
}

void SLAMNonOverlappingFovVIO::initLandmarksBothCameras(std::shared_ptr<Frame> &f) {

    for (const typed_vec_match *lmk_tracks : {&_matches_in_time_lmk, &_matches_in_time_cam1_lmk}) {

        // Landmarks tracked but not initialized yet: triangulate them from all their observations
        for (const auto &ttracks_in_time : *lmk_tracks) {
            for (const auto &ttime : ttracks_in_time.second) {
                std::shared_ptr<ALandmark> lmk = ttime.first->getLandmark().lock();
                if (!lmk || lmk->isInitialized())
                    continue;

                // Only features still attached to a sensor can be triangulated
                std::vector<std::shared_ptr<AFeature>> features;
                for (auto feat : lmk->getFeatures()) {
                    std::shared_ptr<AFeature> fs = feat.lock();
                    if (fs && fs->getSensor())
                        features.push_back(fs);
                }
                _slam_param->getLandmarksInitializer()[ttracks_in_time.first]->initFromFeatures(features);
            }
        }
    }

    // New landmarks from the tracks in time of each camera (no matches between the cameras)
    for (const typed_vec_match *tracks : {&_matches_in_time, &_matches_in_time_cam1}) {
        for (const auto &ttracks_in_time : *tracks) {
            for (const auto &ttime : ttracks_in_time.second) {
                std::vector<std::shared_ptr<AFeature>> feats = {ttime.first, ttime.second};
                _slam_param->getLandmarksInitializer()[ttracks_in_time.first]->initFromFeatures(feats);
            }
        }
    }
}

bool SLAMNonOverlappingFovVIO::shouldInsertKeyframeBothCameras(std::shared_ptr<Frame> &f) {

    // Average parallax of the tracks of both cameras, each with its own rotation compensated
    double avg_parallax  = 0;
    double n_matches     = 0;
    double n_matches_lmk = 0;
    const std::vector<std::pair<typed_vec_match *, typed_vec_match *>> tracks = {
        {&_matches_in_time, &_matches_in_time_lmk}, {&_matches_in_time_cam1, &_matches_in_time_cam1_lmk}};
    for (size_t c = 0; c < tracks.size(); c++) {
        const Eigen::Affine3d T_lc_c = getLastKF()->getSensors().at(c)->getWorld2SensorTransform() *
                                       f->getSensors().at(c)->getSensor2WorldTransform();
        for (const typed_vec_match *m : {tracks[c].first, tracks[c].second}) {
            for (const auto &tmatch : *m) {
                n_matches += tmatch.second.size();
                if (m == tracks[c].second)
                    n_matches_lmk += tmatch.second.size();
                for (const auto &match : tmatch.second) {
                    const double c_angle = match.first->getBearingVectors().at(0).transpose() * T_lc_c.rotation() *
                                           match.second->getBearingVectors().at(0);
                    avg_parallax += std::acos(std::min(1.0, std::max(-1.0, c_angle)));
                }
            }
        }
    }
    avg_parallax = n_matches > 0 ? avg_parallax / n_matches * 180 / M_PI : 0;
    _parallax    = avg_parallax;

    // Same conditions as SLAMCore::shouldInsertKeyframe (mono rule for the landmark count)
    if (f->isKeyFrame()) {
        logKfVote(f, n_matches, n_matches_lmk, "forced");
        return true;
    }
    if (avg_parallax < _min_movement_parallax) {
        logKfVote(f, n_matches, n_matches_lmk, "no_motion");
        return false;
    }
    if (avg_parallax > _max_movement_parallax) {
        f->setKeyFrame();
        logKfVote(f, n_matches, n_matches_lmk, "parallax");
        return true;
    }
    if (n_matches < _min_lmk_number) {
        f->setKeyFrame();
        logKfVote(f, n_matches, n_matches_lmk, "few_lmk");
        return true;
    }
    logKfVote(f, n_matches, n_matches_lmk, "none");
    return false;
}

} // namespace isae
