#include "isaeslam/slamCore.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <unordered_set>

namespace isae {

SLAMCore::SLAMCore(std::shared_ptr<isae::SLAMParameters> slam_param) : _slam_param(slam_param) {
    std::cout << _slam_param->_config.min_kf_number << "," << _slam_param->_config.max_kf_number << ","
              << _slam_param->_config.min_lmk_number << "," << _slam_param->_config.fixed_frame_number << ","
              << _slam_param->_config.min_movement_parallax << "," << _slam_param->_config.max_movement_parallax
              << std::endl;

    _max_movement_parallax = _slam_param->_config.max_movement_parallax;
    _min_movement_parallax = _slam_param->_config.min_movement_parallax;
    _min_lmk_number        = _slam_param->_config.min_lmk_number;

    _local_map  = std::make_shared<LocalMap>(_slam_param->_config.min_kf_number,
                                            _slam_param->_config.max_kf_number,
                                            _slam_param->_config.fixed_frame_number);
    _global_map = std::make_shared<GlobalMap>();
    // Each KF leaving the window goes to the loop closure with its final estimate, before its data is cleaned
    if (_slam_param->_config.loop_closure == 1 && _slam_param->_config.slam_mode != "mono")
        _local_map->setDiscardCallback([this](const std::shared_ptr<Frame> &f) { handToLoopClosure(f); });
    if (slam_param->_config.mesh3D)
        _mesher = std::make_shared<Mesher>(
            _slam_param->_config.slam_mode, _slam_param->_config.ZNCC_tsh, _slam_param->_config.max_length_tsh);

    _avg_detect_t      = 0;
    _avg_lmk_init_t    = 0;
    _avg_lmk_resur_t   = 0;
    _avg_matches_frame = 0;
    _avg_match_frame_t = 0;
    _avg_matches_time  = 0;
    _avg_match_time_t  = 0;
    _avg_predict_t     = 0;
    _avg_filter_t      = 0;
    _avg_processing_t  = 0;
    _avg_frame_opt_t   = 0;
    _avg_wdw_opt_t     = 0;
    _avg_clean_t       = 0;
    _avg_marg_t        = 0;
    _removed_feat      = 0;
    _lmk_inmap         = 0;
    _nframes           = 0;
    _nkeyframes        = 0;
};

void SLAMCore::outlierRemoval() {

    // Remove outliers from the current kf after tracking / matching
    isae::typed_vec_features clean_features;

    for (auto tf : _frame->getSensors().at(0)->getFeatures()) {

        for (auto m : _matches_in_time[tf.first]) {
            if (!m.second->isOutlier()) {
                clean_features[tf.first].push_back(m.second);
            }
        }

        for (auto m : _matches_in_time_lmk[tf.first]) {
            if (!m.second->isOutlier()) {
                clean_features[tf.first].push_back(m.second);
            }
        }

        _frame->getSensors().at(0)->purgeFeatures(tf.first);
        _frame->getSensors().at(0)->addFeatures(tf.first, clean_features[tf.first]);
    }
}

void SLAMCore::cleanFeatures(std::shared_ptr<Frame> &f) {

    // remove feature with outlier ldmk and feature
    isae::typed_vec_features clean_features;
    for (size_t i = 0; i < 1; i++) {
        for (auto tfeat : f->getSensors().at(i)->getFeatures()) {
            for (auto feat : tfeat.second) {
                if (feat->getLandmark().lock()) {
                    if (!feat->getLandmark().lock()->isOutlier() && !feat->isOutlier())
                        clean_features[tfeat.first].push_back(feat);
                }
            }
            f->getSensors().at(i)->purgeFeatures(tfeat.first);
            f->getSensors().at(i)->addFeatures(tfeat.first, clean_features[tfeat.first]);
        }
    }
}

void SLAMCore::updateLandmarks(typed_vec_match matches_lmk) {

    // Update all existing landmarks tracked in time
    for (auto &tmatches_lmk : matches_lmk) {
        // Init all tracked feature with existing landmarks
        for (auto &match_lmk : tmatches_lmk.second) {
            _slam_param->getLandmarksInitializer()[tmatches_lmk.first]->initFromMatch(match_lmk);
        }
    }
}

void SLAMCore::initLandmarks(std::shared_ptr<Frame> &f) {

    // Init unitialized landmarks
    for (auto &ttracks_in_time : _matches_in_time_lmk) {

        // Init all tracked feature in frame
        int nb_created = 0;
        for (auto &ttime : ttracks_in_time.second) {

            // Check if the landmark is not initialized (a stale match may have lost its landmark)
            std::shared_ptr<ALandmark> lmk = ttime.first->getLandmark().lock();
            if (!lmk || lmk->isInitialized())
                continue;

            // Build the feature vector (only features still attached to a sensor can be triangulated)
            std::vector<std::shared_ptr<AFeature>> features;
            for (auto feat : lmk->getFeatures()) {
                std::shared_ptr<AFeature> f = feat.lock();
                if (f && f->getSensor())
                    features.push_back(f);
            }
            _slam_param->getLandmarksInitializer()[ttracks_in_time.first]->initFromFeatures(features);
            nb_created++;
        }
    }

    // Init landmarks with tracks in time (+ seek for matches in frame if in stereo)
    for (auto &ttracks_in_time : _matches_in_time) {

        // Init all tracked feature with track in frame
        vec_match to_init;
        int nb_created = 0;
        for (auto &ttime : ttracks_in_time.second) {
            std::vector<std::shared_ptr<AFeature>> feats;
            if (!ttime.first->getSensor() || !ttime.second->getSensor())
                continue;

            to_init.push_back(ttime);
            feats.push_back(ttime.first);
            feats.push_back(ttime.second);
            nb_created++;

            // Add a new feature for triangulation if it is also matched in frame (bimono case)
            if (_slam_param->getDataProvider()->getNCam() == 2) {
                for (auto &tframe : _matches_in_frame[ttracks_in_time.first]) {
                    if (tframe.first->getLandmark().lock())
                        continue;

                    if (ttime.second == tframe.first) {
                        // Check if the feat has enough parallax
                        if ((tframe.first->getPoints().at(0) - tframe.second->getPoints().at(0)).norm() < 4) {
                            break;
                        } else {
                            to_init.push_back(tframe);
                            feats.push_back(tframe.second);
                            break;
                        }
                    }
                }
            }

            _slam_param->getLandmarksInitializer()[ttracks_in_time.first]->initFromFeatures(feats);
        }
    }

    // Initializing landmarks with L / R matches only in the worst case
    // Need to initialize the remaining N landmarks with in frame matches
    for (auto &ttracks_in_frame : _matches_in_frame) {

        // Init all tracked feature in frame
        vec_match to_init;
        int nb_created = 0;
        for (auto &tframe : ttracks_in_frame.second) {

            // Check if the feat has enough parallax
            if ((tframe.first->getPoints().at(0) - tframe.second->getPoints().at(0)).norm() < 4) {
                continue;
            }

            // If this frame match has already been used before (should be associated to a ldmk) continue
            if (tframe.first->getLandmark().lock())
                continue;
            else {
                to_init.push_back(tframe);
                nb_created++;
            }
        }
        _slam_param->getLandmarksInitializer()[ttracks_in_frame.first]->initFromMatches(to_init);
    }
}

typed_vec_features SLAMCore::detectFeatures(std::shared_ptr<ImageSensor> &sensor) {

    typed_vec_features new_features;

    // For each detector launch an adaptive detector with bucketting
    for (auto &typed_detector : _slam_param->getFeatureDetectors()) {
        std::vector<std::shared_ptr<AFeature>> features;

        features = typed_detector.second->detectAndComputeGrid(
            sensor->getRawData(), sensor->getMask(), sensor->getFeatures()[typed_detector.first]);
        sensor->addFeatures(typed_detector.first, features);
        new_features[typed_detector.first] = features;
    }

    return new_features;
}

typed_vec_match SLAMCore::epipolarFiltering(std::shared_ptr<ImageSensor> &cam0,
                                            std::shared_ptr<ImageSensor> &cam1,
                                            typed_vec_match matches) {
    typed_vec_match valid_matches;
    Eigen::Affine3d T_c0_c1  = cam0->getWorld2SensorTransform() * cam1->getSensor2WorldTransform();
    Eigen::Vector3d epi_line = -T_c0_c1.translation();
    epi_line /= epi_line.norm();

    // Epipolar filtering for tracks_in_time, only for punctual landmarks
    for (auto &m : matches["pointxd"]) {

        // Check the angle with the epipolar plane
        Eigen::Vector3d ray1 = m.first->getBearingVectors().at(0);
        Eigen::Vector3d ray2 = T_c0_c1.rotation() * m.second->getBearingVectors().at(0);

        Eigen::Vector3d epiplane_normal = ray1.cross(epi_line);
        epiplane_normal /= epiplane_normal.norm();
        double residual = std::abs(epiplane_normal.dot(ray2));

        // The angular threshold is set to 1 degree
        if (90 - std::acos(residual) * 180 / M_PI < 1)
            valid_matches["pointxd"].push_back(m);
        else
            m.second->setOutlier();
    }

    return valid_matches;
}

uint SLAMCore::recoverFeatureFromMapLandmarks(std::shared_ptr<ImageSensor> &sensor) {
    uint nb_resurected = 0;

    _map_mutex.lock();
    for (auto typed_ldmk : _local_map->getLandmarks()) {
        nb_resurected += _slam_param->getFeatureMatchers()[typed_ldmk.first].feature_matcher->ldmk_match(
            sensor, typed_ldmk.second, 5, 5);
    }
    _map_mutex.unlock();

    return nb_resurected;
}

void SLAMCore::predictFeature(std::vector<std::shared_ptr<AFeature>> features,
                              std::shared_ptr<ImageSensor> sensor,
                              std::vector<std::shared_ptr<AFeature>> &features_init,
                              vec_match previous_matches = {}) {
    bool is_init;

    for (auto feature : features) {

        is_init = false;

        if (feature->getLandmark().lock()) {

            // Let's project the landmark with the predicted frame pose
            Eigen::Affine3d T_w_lmk = feature->getLandmark().lock()->getPose();
            std::vector<Eigen::Vector2d> predicted_p2ds;

            bool success;
            success = sensor->project(T_w_lmk, feature->getLandmark().lock()->getModel(), predicted_p2ds);

            if (success && std::isfinite(predicted_p2ds.at(0).x()) && std::isfinite(predicted_p2ds.at(0).y())) {
                features_init.push_back(std::make_shared<AFeature>(predicted_p2ds));
                is_init = true;
            }
        }

        if (is_init)
            continue;

        for (auto match : previous_matches) {

            if (feature == match.first) {
                features_init.push_back(match.second);
                is_init = true;
                break;
            }
        }

        if (is_init)
            continue;
        else
            features_init.push_back(feature);
    }
}

void SLAMCore::computeFeatureVelocity(typed_vec_match &matches) {

    for (auto &match_vec : matches) {

        if (match_vec.second.empty())
            continue;

        if (!match_vec.second.at(0).second->getSensor() || !match_vec.second.at(0).first->getSensor())
            continue;

        double dt = (double)(match_vec.second.at(0).second->getSensor()->getFrame()->getTimestamp() -
                             match_vec.second.at(0).first->getSensor()->getFrame()->getTimestamp()) *
                    1e-9;
        for (auto &match : match_vec.second) {
            std::vector<Eigen::Vector3d> vel_vec;
            for (uint i = 0; i < match.first->getBearingVectors().size(); i++) {
                Eigen::Vector3d velocity =
                    (match.first->getBearingVectors().at(i) - match.second->getBearingVectors().at(i)) / dt;
                vel_vec.push_back(velocity);
            }
            match.first->setVelocity(vel_vec);
        }
    }
}

uint SLAMCore::matchFeatures(std::shared_ptr<ImageSensor> &sensor0,
                             std::shared_ptr<ImageSensor> &sensor1,
                             typed_vec_match &matches,
                             typed_vec_match &matches_lmk,
                             typed_vec_features features_to_track) {

    uint nb_matches = 0;

    for (const auto &typed_matcher : _slam_param->getFeatureMatchers()) {

        // Build features_init vector
        std::vector<std::shared_ptr<AFeature>> features_init;
        predictFeature(
            sensor0->getFeatures(typed_matcher.first), sensor1, features_init, _matches_in_time[typed_matcher.first]);

        matches[typed_matcher.first].clear();
        matches_lmk[typed_matcher.first].clear();

        nb_matches += typed_matcher.second.feature_matcher->match(sensor0->getFeatures(typed_matcher.first),
                                                                  sensor1->getFeatures(typed_matcher.first),
                                                                  features_init,
                                                                  matches[typed_matcher.first],
                                                                  matches_lmk[typed_matcher.first],
                                                                  typed_matcher.second.matcher_width,
                                                                  typed_matcher.second.matcher_height);
    }

    return nb_matches;
}

uint SLAMCore::trackFeatures(std::shared_ptr<ImageSensor> &sensor0,
                             std::shared_ptr<ImageSensor> &sensor1,
                             typed_vec_match &matches,
                             typed_vec_match &matches_lmk,
                             typed_vec_features features_to_track) {

    uint nb_tracks = 0;
    for (const auto &typed_tracker : _slam_param->getFeatureTrackers()) {

        // Build features_init vector
        std::vector<std::shared_ptr<AFeature>> features_init;
        predictFeature(features_to_track[typed_tracker.first], sensor1, features_init, matches[typed_tracker.first]);

        // Clear matches typed vec for update
        matches[typed_tracker.first].clear();
        matches_lmk[typed_tracker.first].clear();

        nb_tracks += typed_tracker.second.feature_tracker->track(sensor0,
                                                                 sensor1,
                                                                 features_to_track[typed_tracker.first],
                                                                 features_init,
                                                                 matches[typed_tracker.first],
                                                                 matches_lmk[typed_tracker.first],
                                                                 typed_tracker.second.tracker_width,
                                                                 typed_tracker.second.tracker_height,
                                                                 typed_tracker.second.tracker_nlvls_pyramids,
                                                                 typed_tracker.second.tracker_max_err,
                                                                 true);
    }

    return nb_tracks;
}

bool SLAMCore::shouldInsertKeyframe(std::shared_ptr<Frame> &f) {

    // Compute average translationnal parallax of matches
    double avg_parallax  = 0;
    double n_matches     = 0;
    double n_matches_lmk = 0;

    // Compute parallax
    Eigen::Affine3d T_lc_c =
        getLastKF()->getSensors().at(0)->getWorld2SensorTransform() * f->getSensors().at(0)->getSensor2WorldTransform();
    for (auto tmatch : _matches_in_time_lmk) {
        n_matches += (_matches_in_time[tmatch.first].size() + _matches_in_time_lmk[tmatch.first].size());
        n_matches_lmk += _matches_in_time_lmk[tmatch.first].size();
    }

    for (auto tmatch : _matches_in_time) {
        for (auto match : tmatch.second) {
            avg_parallax += std::acos(match.first->getBearingVectors().at(0).transpose() * T_lc_c.rotation() *
                                      match.second->getBearingVectors().at(0));
        }
    }

    for (auto tmatch : _matches_in_time_lmk) {
        for (auto match : tmatch.second) {
            avg_parallax += std::acos(match.first->getBearingVectors().at(0).transpose() * T_lc_c.rotation() *
                                      match.second->getBearingVectors().at(0));
        }
    }

    avg_parallax /= n_matches; // n_matches already counts the landmark matches
    avg_parallax *= 180 / M_PI;
    _parallax = avg_parallax;

    // Check conditions to vote for a KF or not

    // Case when it is already a KF
    if (f->isKeyFrame()) {
        logKfVote(f, n_matches, n_matches_lmk, "forced");
        return true;
    }

    // Case when the parallax fall under the parallax noise condition => KF not voted
    if (avg_parallax < _min_movement_parallax) {
        logKfVote(f, n_matches, n_matches_lmk, "no_motion");
        return false;
    }

    // Case when the parallax in degree is over the threshold => KF voted
    if (avg_parallax > _max_movement_parallax) {
        f->setKeyFrame();
        logKfVote(f, n_matches, n_matches_lmk, "parallax");
        return true;
    }

    // Case when many landmarks has been lost => KF voted
    // In mono mode we include also the non triangulated features to avoid poor triangulation
    // Else we just consider the triangulated landmarks
    if (_slam_param->_config.slam_mode == "mono" || _slam_param->_config.slam_mode == "monovio") {
        if (n_matches < _min_lmk_number) { // n_matches already counts the landmark matches
            f->setKeyFrame();
            logKfVote(f, n_matches, n_matches_lmk, "few_lmk");
            return true;
        }
    } else {
        if (n_matches_lmk < _min_lmk_number) {
            f->setKeyFrame();
            logKfVote(f, n_matches, n_matches_lmk, "few_lmk");
            return true;
        }
    }

    logKfVote(f, n_matches, n_matches_lmk, "none");
    return false;
}

void SLAMCore::logKfVote(const std::shared_ptr<Frame> &f, double n_matches, double n_matches_lmk, const char *reason) {
    // Diagnostics of the KF voting (log_slam/kf_votes.csv): one row per tracked frame
    if (!std::filesystem::is_directory("log_slam"))
        std::filesystem::create_directory("log_slam");
    std::ofstream fw("log_slam/kf_votes.csv", _kf_votes_started ? std::ofstream::app : std::ofstream::trunc);
    if (!_kf_votes_started) {
        fw << "timestamp (ns), last_kf_timestamp (ns), parallax_deg, n_matches, n_matches_lmk, reason\n";
        _kf_votes_started = true;
    }
    fw << f->getTimestamp() << "," << getLastKF()->getTimestamp() << "," << _parallax << "," << n_matches << ","
       << n_matches_lmk << "," << reason << "\n";
}

void SLAMCore::logKfFeatures(const std::shared_ptr<Frame> &kf) {
    static const bool enabled = (std::getenv("EXECO_KF_FEATURES_LOG") != nullptr);
    if (!enabled || kf->getSensors().empty())
        return;
    if (!std::filesystem::is_directory("log_slam"))
        std::filesystem::create_directory("log_slam");
    std::ofstream fw("log_slam/kf_features.csv", _kf_features_started ? std::ofstream::app : std::ofstream::trunc);
    if (!_kf_features_started) {
        // status: 0 no landmark, 1 initialized landmark, 2 resurrected landmark, 3 landmark not initialized
        fw << "kf_timestamp (ns), u, v, status\n";
        _kf_features_started = true;
    }
    fw << std::fixed << std::setprecision(1);
    for (const auto &feat : kf->getSensors().at(0)->getFeatures()["pointxd"]) {
        const auto lmk = feat->getLandmark().lock();
        const int status = !lmk ? 0 : lmk->isResurected() ? 2 : lmk->isInitialized() ? 1 : 3;
        const Eigen::Vector2d pt = feat->getPoints().at(0);
        fw << kf->getTimestamp() << "," << pt.x() << "," << pt.y() << "," << status << "\n";
    }
}

bool SLAMCore::plausibleUpdate(const Eigen::Affine3d &T_ref, const Eigen::Affine3d &T_new) {
    const double ref = std::max(geometry::se3_RTtoVec6d(T_ref).norm(), 0.1);
    return (geometry::se3_RTtoVec6d(T_new) - geometry::se3_RTtoVec6d(T_ref)).norm() <= 10 * ref;
}

bool SLAMCore::predict(std::shared_ptr<Frame> &f) {
    Eigen::MatrixXd covdT = 100 * Eigen::MatrixXd::Identity(6, 6);

    // Predict pose with constant velocity model dT
    Eigen::Affine3d T_last_curr = (getLastKF()->getWorld2FrameTransform() * f->getFrame2WorldTransform());
    Eigen::Affine3d T_const     = T_last_curr;

    // False if the prediction failed and constant velocity applies
    if (!_slam_param->getPoseEstimator()->estimateTransformBetween(
            getLastKF(), f, _matches_in_time_lmk["pointxd"], T_last_curr, covdT)) {
        std::cerr << "Predict fails, PnP failed" << std::endl;
        return false;
    } else {

        // Check if the pose is valid: its deviation from the constant velocity prediction must stay below 10 times
        // the predicted motion. The reference has a floor (it was skipped below 0.1 m of predicted translation,
        // which let PnP solutions with a flipped rotation (85 deg) through during slow motion)
        if (!plausibleUpdate(T_const, T_last_curr)) {
            std::cerr << "Predict fails, PnP pose is not valid" << std::endl;
            std::cout << "T_last_curr: " << T_last_curr.translation().transpose() << std::endl;
            std::cout << "T_const: " << T_const.translation().transpose() << std::endl;
            _matches_in_time_lmk["pointxd"].clear(); // The matches are not valid, clear them
            return false;
        }

        // Update the pose only for pnp
        if (_slam_param->_config.pose_estimator != "pnp")
            T_last_curr = T_const;

        f->setWorld2FrameTransform(T_last_curr.inverse() * getLastKF()->getWorld2FrameTransform());

        return true;
    }
}

void SLAMCore::bridgePreintegration(const std::shared_ptr<Frame> &kf, const std::shared_ptr<Frame> &kf_prev) {
    std::shared_ptr<IMU> imu = kf->getIMU();
    if (!imu)
        return;
    if (kf_prev && kf_prev->getIMU()) {
        imu->setLastKF(kf_prev);
        if (imu->repropagate(kf_prev->getIMU()->getBa(), kf_prev->getIMU()->getBg()))
            return;
    }
    imu->setLastKF(nullptr);
}

bool SLAMCore::canBridge(const std::shared_ptr<Frame> &kf, const std::shared_ptr<Frame> &kf_prev) const {
    return kf && kf_prev && kf->getIMU() && kf_prev->getIMU() &&
           (kf->getTimestamp() - kf_prev->getTimestamp()) * 1e-9 <= AOptimizer::kMaxImuFactorDt;
}

bool SLAMCore::inertialInitAccepted(double scale) {
    if (!(scale > 0) || !std::isfinite(scale)) {
        std::cout << "Inertial initialization rejected: scale " << scale << std::endl;
        return false;
    }
    for (auto &f : _local_map->getFrames()) {
        if (!f->getWorld2FrameTransform().matrix().allFinite()) {
            std::cout << "Inertial initialization rejected: non-finite pose" << std::endl;
            return false;
        }
        if (!f->getIMU())
            continue;
        std::shared_ptr<IMU> imu = f->getIMU();
        if (!imu->getVelocity().allFinite() || !imu->getBa().allFinite() || !imu->getBg().allFinite() ||
            imu->getBa().norm() > 2 || imu->getBg().norm() > 0.5) {
            std::cout << "Inertial initialization rejected: v " << imu->getVelocity().transpose() << ", ba "
                      << imu->getBa().transpose() << ", bg " << imu->getBg().transpose() << std::endl;
            return false;
        }
    }
    return true;
}

void SLAMCore::logVIODiag(const std::shared_ptr<Frame> &kf, const VIOptimStats &stats) {

    if (!kf || !kf->getIMU())
        return;

    if (!std::filesystem::is_directory("log_slam"))
        std::filesystem::create_directory("log_slam");

    std::ofstream fw("log_slam/vio_diag.csv", _vio_diag_started ? std::ofstream::app : std::ofstream::trunc);
    if (!_vio_diag_started) {
        fw << "kf_timestamp (ns), last_kf_timestamp (ns), factor_dt, imu_integrated_dt, imu_gap_steps, "
           << "rejected_imu_total, n_imu_factors, solver_usable, termination, initial_cost, final_cost, iterations, "
           << "nonfinite_states, v_x, v_y, v_z, ba_x, ba_y, ba_z, bg_x, bg_y, bg_z, n_prior_factors, n_other_factors, "
           << "prior_cost0, prior_lmk_offset, prior_frame_rot_offset, prior_frame_t_offset, prior_v_offset, "
           << "prior_ba_offset, cost_visual, cost_imu, cost_bias, cost_prior\n";
        _vio_diag_started = true;
    }

    // Count window states with non-finite pose, velocity or biases
    int nonfinite = 0;
    for (auto &f : _local_map->getFrames()) {
        bool ok = f->getWorld2FrameTransform().matrix().allFinite();
        if (f->getIMU())
            ok = ok && f->getIMU()->getVelocity().allFinite() && f->getIMU()->getBa().allFinite() &&
                 f->getIMU()->getBg().allFinite();
        nonfinite += !ok;
    }

    std::shared_ptr<IMU> imu      = kf->getIMU();
    std::shared_ptr<Frame> lastkf = imu->getLastKF();
    double factor_dt = lastkf ? (kf->getTimestamp() - lastkf->getTimestamp()) * 1e-9 : -1;
    Eigen::Vector3d v = imu->getVelocity(), ba = imu->getBa(), bg = imu->getBg();
    fw << kf->getTimestamp() << "," << (lastkf ? lastkf->getTimestamp() : 0) << "," << factor_dt << ","
       << imu->getIntegratedDt() << "," << imu->getGapSteps() << "," << _n_rejected_imu << ","
       << stats.n_imu_factors << "," << stats.usable << "," << stats.termination << "," << stats.initial_cost << ","
       << stats.final_cost << "," << stats.iterations << "," << nonfinite << "," << v.x() << "," << v.y() << ","
       << v.z() << "," << ba.x() << "," << ba.y() << "," << ba.z() << "," << bg.x() << "," << bg.y() << ","
       << bg.z() << "," << stats.n_prior_factors << "," << stats.n_other_factors << "," << stats.prior_cost0 << ","
       << stats.prior_lmk_offset << "," << stats.prior_frame_rot_off << "," << stats.prior_frame_t_off << ","
       << stats.prior_v_off << "," << stats.prior_ba_off << "," << stats.cost_visual << "," << stats.cost_imu << ","
       << stats.cost_bias << "," << stats.cost_prior << "\n";
}

void SLAMCore::profiling() {

    if (!std::filesystem::is_directory("log_slam"))
        std::filesystem::create_directory("log_slam");

    if (!_is_init) {
        // Clean the result files once per run: a re-initialization used to truncate them too, so the evaluation only
        // saw the segment after the last one (a failing run looked like a short, accurate one)
        if (_results_started && _segment_has_rows) {
            _segment++; // a re-initialization: the next rows belong to a new segment
            _segment_has_rows = false;
        }
        if (!_results_started) {
            _results_started = true;

            // Clean the result file
            std::ofstream fw_res("log_slam/results.csv", std::ofstream::out | std::ofstream::trunc);
            fw_res << "timestamp (ns), nframes, T_wf(00), T_wf(01), T_wf(02), T_wf(03), T_wf(10), T_wf(11), T_wf(12), "
                   << "T_wf(13), T_wf(20), T_wf(21), T_wf(22), T_wf(23), segment\n";
            fw_res.close();

            std::ofstream fw_res1("log_slam/cov_mat.csv", std::ofstream::out | std::ofstream::trunc);
            fw_res1 << "timestamp (ns), timestamp previous (ns), cov(00), cov(11), cov(22), cov(33), "
                    << "cov(44), cov(55), parallax, nb_tracks, nb_outliers, t(0), t(1), t(3), r(0), r(1), r(2), "
                    << "vel_norm \n";
            fw_res1.close();
        }

        // For timing statistics
        // // Clean profiling file
        // std::ofstream fw_prof_fefr("log_slam/timing_fe_fr.csv",
        //                            std::ofstream::out | std::ofstream::trunc);
        // fw_prof_fefr << "tracking_dt, predict_dt, epi_dt, filter_dt, optim_dt\n";
        // fw_prof_fefr.close();

        // // Clean profiling file
        // std::ofstream fw_prof_fekfr("log_slam/timing_fe_kfr.csv",
        //                             std::ofstream::out | std::ofstream::trunc);
        // fw_prof_fekfr
        //     << "tracking_dt, predict_dt, epi_dt, filter_dt, optim_dt, detect_dt, reover_dt, track_f_dt, init_dt\n";
        // fw_prof_fekfr.close();

        // // Clean profiling file
        // std::ofstream fw_prof_be("log_slam/timing_be.csv",
        //                             std::ofstream::out | std::ofstream::trunc);
        // fw_prof_be
        //     << "marg_dt, optim_dt, epi_dt\n";
        // fw_prof_be.close();
    } else {

        // Write in a txt file for evaluation: each KF once, when it leaves the window (its final estimate). The KFs
        // still in the window at a re-initialization are written before it (logWindowBeforeReset)
        if (getLastKF()) {
            std::shared_ptr<Frame> front = _local_map->getFrames().front();
            if (_results_front && _results_front != front)
                writeResultRow(_results_front);
            _results_front = front;
        }

        if (_local_map->getFrames().size() > 2) {
            std::ofstream fw_res("log_slam/cov_mat.csv", std::ofstream::out | std::ofstream::app);
            std::shared_ptr<Frame> f  = _frame_to_optim;
            std::shared_ptr<Frame> fp = _local_map->getFrames().at(_local_map->getFrames().size() - 2);
            Eigen::MatrixXd cov       = f->getdTCov();
            Eigen::Affine3d T_fp_f    = fp->getWorld2FrameTransform() * f->getFrame2WorldTransform();
            Eigen::Vector3d t         = T_fp_f.translation();
            Eigen::Vector3d r         = geometry::log_so3(T_fp_f.rotation());
            uint nb_lmk               = _matches_in_time_lmk["pointxd"].size() + _matches_in_time["pointxd"].size();
            double vel_norm           = _6d_velocity.block(2, 0, 3, 1).norm();
            fw_res << f->getTimestamp() << "," << fp->getTimestamp() << "," << cov(0, 0) << "," << cov(1, 1) << ","
                   << cov(2, 2) << "," << cov(3, 3) << "," << cov(4, 4) << "," << cov(5, 5) << "," << _parallax << ","
                   << nb_lmk << "," << t(0) << "," << t(1) << "," << t(2) << "," << r(0) << "," << r(1) << "," << r(2)
                   << "," << vel_norm << "\n";
            fw_res.close();
        }

        // For timing statistics
        // if (!_timings_frate.empty()) {

        //     if (!_frame->isKeyFrame()) {
        //         if (_timings_frate.back().empty())
        //             return;
        //         std::ofstream fw_prof_fefr("log_slam/timing_fe_fr.csv",
        //                                    std::ofstream::out | std::ofstream::app);
        //         fw_prof_fefr << _timings_frate.back().at(0) << "," << _timings_frate.back().at(1) << ","
        //                      << _timings_frate.back().at(2) << "," << _timings_frate.back().at(3) << ","
        //                      << _timings_frate.back().at(4) << "\n";
        //         fw_prof_fefr.close();
        //     } else {
        //         if (_timings_kfrate_fe.back().size() < 9)
        //             return;
        //         std::ofstream fw_prof_fekfr("log_slam/timing_fe_kfr.csv",
        //                                     std::ofstream::out | std::ofstream::app);
        //         fw_prof_fekfr << _timings_kfrate_fe.back().at(0) << "," << _timings_kfrate_fe.back().at(1) << ","
        //                       << _timings_kfrate_fe.back().at(2) << "," << _timings_kfrate_fe.back().at(3) << ","
        //                       << _timings_kfrate_fe.back().at(4) << "," << _timings_kfrate_fe.back().at(5) << ","
        //                       << _timings_kfrate_fe.back().at(6) << "," << _timings_kfrate_fe.back().at(7) << ","
        //                       << _timings_kfrate_fe.back().at(8) << "\n";
        //         fw_prof_fekfr.close();

        //         std::ofstream fw_prof_be("log_slam/timing_be.csv",
        //                                     std::ofstream::out | std::ofstream::app);
        //         fw_prof_be << _timings_kfrate_be.back().at(0) << "," << _timings_kfrate_be.back().at(1) << "\n";
        //         fw_prof_be.close();
        //     }
        // }
    }

    // Write a txt file for profiling
    std::ofstream fw("log_slam/slam_profiler.txt", std::ofstream::out);
    fw << "===== SLAM profiler ======= \n";
    fw << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
    fw << "Dataset: " << _slam_param->_config.dataset_id << "\n";
    fw << "Number of frames: " << _nframes << "\n";
    fw << "Number of keyframes: " << _nkeyframes << "\n";
    fw << "Img process dt: " << _avg_processing_t << "\n";
    fw << "Avg number of matches in frame: " << _avg_matches_frame << "\n";
    fw << "Avg number of matches in time: " << _avg_matches_time << "\n";
    fw << "Removed matches: " << _removed_feat << "\n";
    fw << "Avg number of lmks resurected: " << _avg_resur_lmk << "\n";
    fw << "Avg landmarks matched in map: " << _lmk_inmap << "\n";
    fw << "Detection dt: " << _avg_detect_t << "\n";
    fw << "Prediction " + _slam_param->_config.pose_estimator + "RANSAC dt: " << _avg_predict_t << "\n";
    fw << "Matching in frame dt: " << _avg_match_frame_t << "\n";
    fw << "Matching in time dt: " << _avg_match_time_t << "\n";
    fw << "Average filter time dt: " << _avg_filter_t << "\n";
    fw << "Average cleaning time dt: " << _avg_clean_t << "\n";
    fw << "Landmark init dt: " << _avg_lmk_init_t << "\n";
    fw << "Optimize frame dt: " << _avg_frame_opt_t << "\n";
    fw << "Marginalization dt: " << _avg_marg_t << "\n";
    if (_slam_param->_config.mesh3D)
        fw << "Mesh dt: " << _mesher->_avg_mesh_t << "\n";
    fw << "Optimize window dt: " << _avg_wdw_opt_t << "\n";
    float frontend_dt = _avg_detect_t + _avg_predict_t + _avg_match_frame_t * _nkeyframes / _nframes +
                        _avg_match_time_t + _avg_filter_t + _avg_clean_t + _avg_lmk_init_t * _nkeyframes / _nframes +
                        _avg_frame_opt_t + _avg_lmk_resur_t * _nkeyframes / _nframes;
    fw << "Front end dt: " << frontend_dt << "\n";
    float backend_dt = _avg_wdw_opt_t + _avg_marg_t;
    fw << "Back end dt: " << backend_dt << "\n";
    if (_loop_closure) {
        const LoopClosure::Stats ls = _loop_closure->stats();
        fw << "Loop closure dt: " << ls.avg_ms_keyframe << "\n";
        fw << "Loop descriptor dt: " << ls.avg_ms_describe << "\n";
        fw << "Loop closure KFs: " << ls.n_keyframes << "\n";
        fw << "Loop candidates: " << ls.n_candidates << "\n";
        fw << "Loops verified: " << ls.n_verified << "\n";
        fw << "Loops rejected by the pose graph: " << ls.n_rejected_by_graph << "\n";
        fw << "Pose graph dt: " << ls.avg_ms_optimize << "\n";
    }
}

void SLAMCore::writeResultRow(const std::shared_ptr<Frame> &f) {
    if (!f || f->getTimestamp() <= _last_logged_ts)
        return;
    std::ofstream fw_res("log_slam/results.csv", std::ofstream::out | std::ofstream::app);
    Eigen::Affine3d T_w_f   = f->getFrame2WorldTransform();
    const Eigen::Matrix3d R = T_w_f.linear();
    Eigen::Vector3d twc     = T_w_f.translation();
    fw_res << f->getTimestamp() << "," << _nframes << "," << R(0, 0) << "," << R(0, 1) << "," << R(0, 2) << ","
           << twc.x() << "," << R(1, 0) << "," << R(1, 1) << "," << R(1, 2) << "," << twc.y() << "," << R(2, 0) << ","
           << R(2, 1) << "," << R(2, 2) << "," << twc.z() << "," << _segment << "\n";
    _last_logged_ts   = f->getTimestamp();
    _segment_has_rows = true;
}

void SLAMCore::applyLoopCorrection() {
    Eigen::Affine3d C;
    if (!_loop_closure || !_loop_closure->takeCorrection(_segment, C))
        return;
    std::unordered_set<Frame *> frames_done;
    std::unordered_set<ALandmark *> lmks_done;
    auto moveLandmark = [&](const std::shared_ptr<ALandmark> &l) {
        if (!l || !lmks_done.insert(l.get()).second)
            return;
        Eigen::Affine3d T = l->getPose();
        if (l->_label == "pointxd")
            T.translation() = C * T.translation(); // points only translate (no orientation)
        else
            T = C * T;
        l->setPose(T);
    };
    auto moveFrame = [&](const std::shared_ptr<Frame> &f) {
        if (!f || !frames_done.insert(f.get()).second)
            return;
        f->setWorld2FrameTransform((C * f->getFrame2WorldTransform()).inverse());
        if (f->getIMU())
            f->getIMU()->setVelocity(C.rotation() * f->getIMU()->getVelocity());
        for (auto &tl : f->getLandmarks())
            for (auto &l : tl.second)
                moveLandmark(l);
    };
    _map_mutex.lock();
    for (auto &tl : _local_map->getLandmarks())
        for (auto &l : tl.second)
            moveLandmark(l);
    for (auto &f : _local_map->getFrames())
        moveFrame(f);
    moveFrame(_frame_to_optim);
    moveFrame(_frame);
    _map_mutex.unlock();
    // The marginalization priors move with the window: their cost for the moved states is the one they had
    _slam_param->getOptimizerBack()->transformPriors(C);
    std::cout << "Loop closure: window moved by " << C.translation().norm() << " m, "
              << Eigen::AngleAxisd(C.rotation()).angle() * 180 / M_PI << " deg" << std::endl;
}

void SLAMCore::handToLoopClosure(const std::shared_ptr<Frame> &f) {
    const Config &cfg = _slam_param->_config;
    if (cfg.loop_closure != 1 || cfg.slam_mode == "mono" || !f || f->getTimestamp() <= _last_loop_ts)
        return;
    if (!_loop_closure) {
        LoopClosure::Options opt;
        opt.detector    = cfg.loop_detector;
        opt.vocabulary  = cfg.loop_vocabulary;
        opt.model         = cfg.loop_model;
        opt.model_device  = cfg.loop_model_device;
        opt.model_threads = cfg.loop_model_threads;
        opt.min_gap     = cfg.loop_min_gap;
        opt.min_inliers = cfg.loop_min_inliers;
        opt.gate_radius = cfg.loop_gate_radius;
        // 0 (automatic) is 6-DoF in every mode: with an IMU too it was as good or better than 4-DoF on TUM-VI and
        // EuRoC (VIO's roll and pitch are not exact; doc/loop_closure ledger, 2026-10-09)
        opt.four_dof    = cfg.loop_graph_dof == 4;
        opt.async       = cfg.multithreading;
        // Automatic window correction (-1): where it measured best (doc/loop_closure ledger): VIO with a dense prior
        // (better than output-only and than dropping the prior) and mono VIO (repairs failing runs); elsewhere the
        // output is corrected and the window left as it is
        const bool vio_dense_prior = cfg.slam_mode == "bimonovio" && cfg.marginalization == 1 && !cfg.sparsification;
        opt.correct_window         = cfg.loop_correct_window == 1 ||
                             (cfg.loop_correct_window == -1 && (vio_dense_prior || cfg.slam_mode == "monovio"));
        opt.window_gravity = cfg.slam_mode == "monovio" || cfg.slam_mode == "bimonovio";
        std::atomic_store(&_loop_closure, std::make_shared<LoopClosure>(opt)); // read by viewer threads
    }
    _loop_closure->addKeyframe(f, _segment);
    _last_loop_ts = f->getTimestamp();
}

void SLAMCore::logWindowBeforeReset() {
    if (!_results_started)
        return;
    writeResultRow(_results_front);
    for (auto &f : _local_map->getFrames()) {
        writeResultRow(f);
        handToLoopClosure(f); // the window's KFs are not discarded one by one before a reset
    }
    _results_front = nullptr;
}

void SLAMCore::captureCarriedState(const std::shared_ptr<IMU> &imu) {
    _carried.valid = false;
    if (!_slam_param->_config.reinit_carry_state || !imu || !imu->getFrame())
        return;
    const std::shared_ptr<Frame> f = imu->getFrame();
    if (_carried.capture(f->getFrame2WorldTransform(),
                         imu->getVelocity(),
                         imu->getBa(),
                         imu->getBg(),
                         imu->getAcc(),
                         imu->getGyr(),
                         f->getTimestamp(),
                         _last_visual_ts,
                         imu->maxStepDt(),
                         _slam_param->_config.reinit_carry_max_age_vio))
        std::cout << "Re-initialization will start from the last state (" << _carried.age(f->getTimestamp())
                  << " s after the last visual pose)" << std::endl;
    else
        std::cout << "Re-initialization from scratch: last state not usable (" << _carried.age(f->getTimestamp())
                  << " s after the last visual pose, |ba| " << imu->getBa().norm() << ", |bg| "
                  << imu->getBg().norm() << ")" << std::endl;
}

std::shared_ptr<Frame> SLAMCore::nextFrame() {
    std::shared_ptr<Frame> f;
    if (!_frontend_lock) {
        f = _slam_param->getDataProvider()->next();
    } else {
        _frontend_lock->unlock();
        _step_cv.notify_all();
        f = _slam_param->getDataProvider()->next();
        _frontend_lock->lock();

        // The front end must not process a frame before the back end has added the last voted KF to the local map:
        // getLastKF() would still be the KF before it, and the frame (IMU preintegration, tracking) would be anchored
        // to that older KF, e.g. an IMU factor skipping a KF of the window
        waitBackEnd();
    }

    // A state carried to the next initialization follows every IMU measurement, whichever code reads the frame
    if (_carried.valid && f && f->getIMU())
        _carried.propagate(f->getIMU()->getAcc(), f->getIMU()->getGyr(), f->getTimestamp(), f->getIMU()->rawGap());
    return f;
}

void SLAMCore::waitBackEnd() {
    if (!_frontend_lock)
        return;
    _step_cv.notify_all();
    _step_cv.wait(*_frontend_lock, [this] { return _frame_to_optim == nullptr; });
}

void SLAMCore::runFrontEnd() {

    std::unique_lock<std::mutex> lock(_step_mutex);
    _frontend_lock = &lock;

    while (true) {

        if (!_is_init) {
            // A re-initialization starts once the back end is done with the last KF
            waitBackEnd();
            bool init_success = this->init();
            while (!init_success)
                init_success = this->init();
        } else
            this->frontEndStep();

        // Let the back end take the KF that may have been voted
        lock.unlock();
        _step_cv.notify_all();
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        lock.lock();
    }
}

void SLAMCore::runBackEnd() {

    while (true) {

        {
            std::unique_lock<std::mutex> lock(_step_mutex);
            _step_cv.wait(lock, [this] { return _frame_to_optim != nullptr; });
            this->backEndStep();
        }
        _step_cv.notify_all();
    }
}

void SLAMCore::runFullOdom() {

    while (true) {

        if (!_is_init) {
            bool init_success = this->init();
            while (!init_success)
                init_success = this->init();
        } else
            this->frontEndStep();

        this->backEndStep();

        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
}

} // namespace isae