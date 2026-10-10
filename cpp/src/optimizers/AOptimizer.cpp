#include "isaeslam/optimizers/AOptimizer.h"

namespace {

// Absolute prior on a bias: (b + db) / sigma, with b the current bias and db the window's increment
struct BiasPriorError {
    BiasPriorError(const Eigen::Vector3d &b, double sigma) : _b(b), _w(1 / sigma) {}
    template <typename T> bool operator()(const T *db, T *r) const {
        for (int k = 0; k < 3; k++)
            r[k] = (T(_b(k)) + db[k]) * T(_w);
        return true;
    }
    Eigen::Vector3d _b;
    double _w;
};

} // namespace

namespace isae {

void AOptimizer::addSparsePriorResiduals(ceres::Problem &problem,
                                         ceres::LossFunction *loss_function,
                                         ceres::ParameterBlockOrdering *ordering) {

    // Explicit checks on the retained variables (the previous gating counted landmark *types*, or required a
    // frame to keep even in VO)
    if (!_marginalization->_has_prior)
        return;
    std::vector<std::shared_ptr<ALandmark>> kept;
    if (_marginalization->_lmk_to_keep.count("pointxd"))
        kept = _marginalization->_lmk_to_keep.at("pointxd");

    // Landmarks of the prior are supposed to be in the problem, add them otherwise
    auto lmk_block = [&](const std::shared_ptr<ALandmark> &lmk) {
        if (_map_lmk_ptpar.find(lmk) == _map_lmk_ptpar.end()) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            problem.AddParameterBlock(_map_lmk_ptpar.at(lmk).values(), 3);
            ordering->AddElementToGroup(_map_lmk_ptpar.at(lmk).values(), 0);
        }
        return _map_lmk_ptpar.at(lmk).values();
    };

    std::shared_ptr<Frame> frame_to_keep = _marginalization->_frame_to_keep;

    /// CASE 1 VIO ///
    if (frame_to_keep && frame_to_keep->getIMU()) {

        // Ignore if the frame to keep is not in the problem
        if (_map_frame_posepar.find(frame_to_keep) == _map_frame_posepar.end() ||
            _map_frame_velpar.find(frame_to_keep) == _map_frame_velpar.end() ||
            _marginalization->_map_frame_inf.find(frame_to_keep) == _marginalization->_map_frame_inf.end())
            return;

        Eigen::Affine3d T_f_w = frame_to_keep->getWorld2FrameTransform();
        Eigen::Vector3d v     = frame_to_keep->getIMU()->getVelocity();
        Eigen::Vector3d ba    = frame_to_keep->getIMU()->getBa();
        Eigen::Vector3d bg    = frame_to_keep->getIMU()->getBg();
        // The prior is centred on its linearization point (not on the current state, which made it pull towards
        // wherever the window had drifted to)
        // Values in the current world (a loop closure may have moved the window since the prior was made): the pose
        // error is relative to the linearization pose and does not change, the velocity error rotates
        Eigen::MatrixXd S_frame = _marginalization->_map_frame_inf.at(frame_to_keep);
        if (S_frame.cols() == 15)
            S_frame.middleCols(6, 3) = S_frame.middleCols(6, 3) * _marginalization->_W_prior.rotation();
        ceres::CostFunction *cost_fct0 = new IMUPriordx(T_f_w,
                                                        _marginalization->linPoseCurrent(),
                                                        v,
                                                        _marginalization->linVelocityCurrent(),
                                                        ba,
                                                        _marginalization->_ba_lin,
                                                        bg,
                                                        _marginalization->_bg_lin,
                                                        S_frame);
        problem.AddResidualBlock(cost_fct0,
                                 loss_function,
                                 _map_frame_posepar.at(frame_to_keep).values(),
                                 _map_frame_velpar.at(frame_to_keep).values(),
                                 _map_frame_dbapar.at(frame_to_keep).values(),
                                 _map_frame_dbgpar.at(frame_to_keep).values());

        // Relative factors for the landmarks
        for (auto &lmk : kept) {
            if (_marginalization->_map_lmk_inf.find(lmk) == _marginalization->_map_lmk_inf.end())
                continue;
            ceres::CostFunction *cost_fct = new PoseToLandmarkFactor(_marginalization->_map_lmk_prior.at(lmk),
                                                                     T_f_w,
                                                                     lmk->getPose().translation(),
                                                                     _marginalization->_map_lmk_inf.at(lmk));
            problem.AddResidualBlock(
                cost_fct, loss_function, _map_frame_posepar.at(frame_to_keep).values(), lmk_block(lmk));
        }
        return;
    }

    /// CASE 2 VO ///
    if (kept.empty() || !_marginalization->_lmk_with_prior)
        return;

    // Unary factor for the landmark with a prior
    double *prior_block = lmk_block(_marginalization->_lmk_with_prior);
    // The VO factors are along world axes: their values in the current world
    const Eigen::Affine3d W_cur = _marginalization->_W_prior.inverse();
    const Eigen::Matrix3d R_W   = _marginalization->_W_prior.rotation();
    ceres::CostFunction *cost_fct_0 = new Landmark3DPrior(W_cur * _marginalization->_prior_lmk,
                                                          _marginalization->_lmk_with_prior->getPose().translation(),
                                                          Eigen::Matrix3d(_marginalization->_info_lmk * R_W));
    problem.AddResidualBlock(cost_fct_0, loss_function, prior_block);
    ordering->Remove(prior_block);
    ordering->AddElementToGroup(prior_block, 2);

    // Relative factors along the landmark chain
    for (size_t k = 0; k + 1 < kept.size(); k++) {
        std::shared_ptr<ALandmark> lmk_k = kept.at(k), lmk_kp1 = kept.at(k + 1);
        if (lmk_k == lmk_kp1 || _marginalization->_map_lmk_inf.find(lmk_kp1) == _marginalization->_map_lmk_inf.end())
            continue;
        double *block_k = lmk_block(lmk_k), *block_kp1 = lmk_block(lmk_kp1);
        ordering->Remove(block_kp1);
        ordering->AddElementToGroup(block_kp1, 2);
        ceres::CostFunction *cost_fct = new LandmarkToLandmarkFactor(R_W.transpose() * _marginalization->_map_lmk_prior.at(lmk_kp1),
                                                                     lmk_k->getPose().translation(),
                                                                     lmk_kp1->getPose().translation(),
                                                                     Eigen::Matrix3d(_marginalization->_map_lmk_inf.at(lmk_kp1) * R_W));
        problem.AddResidualBlock(cost_fct, loss_function, block_k, block_kp1);
    }
}

bool AOptimizer::imuFactorUsable(const std::shared_ptr<Frame> &fi, const std::shared_ptr<Frame> &fj) {
    if (!fi || !fj || fi == fj || !fi->getIMU() || !fj->getIMU() || fj->getIMU()->getLastKF() != fi)
        return false;
    const double span = (fj->getTimestamp() - fi->getTimestamp()) * 1e-9;
    if (span > kMaxImuFactorDt)
        return false;
    // The preintegration must cover the interval between the two KFs. With an intact chain the integrated steps
    // telescope to exactly this span (every frame has an IMU sample at its own stamp, whatever the dataset), so
    // any difference beyond rounding means the preintegration does not start at fi
    if (std::abs(fj->getIMU()->getIntegratedDt() - span) > 1e-6)
        return false;
    return fj->getIMU()->getGapSteps() == 0 && fj->getIMU()->hasValidCovariance();
}

void AOptimizer::repropagateIfNeeded(std::vector<std::shared_ptr<Frame>> &frame_vector) {
    for (auto &frame : frame_vector) {
        if (!frame->getIMU())
            continue;
        std::shared_ptr<Frame> frame_i = frame->getIMU()->getLastKF();
        if (!frame_i || !frame_i->getIMU())
            continue;
        Eigen::Vector3d ba = frame_i->getIMU()->getBa(), bg = frame_i->getIMU()->getBg();
        if ((ba - frame->getIMU()->getBaLin()).norm() > 0.1 || (bg - frame->getIMU()->getBgLin()).norm() > 0.01)
            frame->getIMU()->repropagate(ba, bg);
    }
}

uint AOptimizer::addIMUResiduals(ceres::Problem &problem,
                                 ceres::LossFunction *loss_function,
                                 ceres::ParameterBlockOrdering *ordering,
                                 std::vector<std::shared_ptr<Frame>> &frame_vector,
                                 size_t fixed_frame_number) {

    uint n_imu_factors = 0;

    // Add parameter blocks specific to IMU (we suppose that the parameter blocks for pose were already added)
    for (size_t i = 0; i < frame_vector.size(); i++) {

        if (frame_vector.at(i)->getIMU()) {
            _map_frame_velpar.emplace(frame_vector.at(i), PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            _map_frame_dbapar.emplace(frame_vector.at(i), PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            _map_frame_dbgpar.emplace(frame_vector.at(i), PointXYZParametersBlock(Eigen::Vector3d::Zero()));

            problem.AddParameterBlock(_map_frame_velpar.at(frame_vector.at(i)).values(), 3);
            ordering->AddElementToGroup(_map_frame_velpar.at(frame_vector.at(i)).values(), 1);

            problem.AddParameterBlock(_map_frame_dbapar.at(frame_vector.at(i)).values(), 3);
            ordering->AddElementToGroup(_map_frame_dbapar.at(frame_vector.at(i)).values(), 1);

            problem.AddParameterBlock(_map_frame_dbgpar.at(frame_vector.at(i)).values(), 3);
            ordering->AddElementToGroup(_map_frame_dbgpar.at(frame_vector.at(i)).values(), 1);

            // Without a marginalization prior the biases are held only by the window's data and their random walk:
            // when vision is weak they drift freely (V2_03, no prior: |ba| up to 2.6 m/s^2, then the velocity
            // integrates it). A weak absolute prior keeps them physical
            const std::shared_ptr<IMU> imu = frame_vector.at(i)->getIMU();
            if (_bias_prior_acc > 0)
                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<BiasPriorError, 3, 3>(
                                             new BiasPriorError(imu->getBa(), _bias_prior_acc)),
                                         nullptr, _map_frame_dbapar.at(frame_vector.at(i)).values());
            if (_bias_prior_gyr > 0)
                problem.AddResidualBlock(new ceres::AutoDiffCostFunction<BiasPriorError, 3, 3>(
                                             new BiasPriorError(imu->getBg(), _bias_prior_gyr)),
                                         nullptr, _map_frame_dbgpar.at(frame_vector.at(i)).values());
        }
    }

    for (size_t i = 0; i < frame_vector.size(); i++) {

        std::shared_ptr<Frame> framej = frame_vector.at(i);

        // Skip if no IMU
        if (!framej->getIMU())
            continue;
        std::shared_ptr<Frame> framei = framej->getIMU()->getLastKF();

        // Skip if framei == nullptr
        if (!framei)
            continue;

        if (_map_frame_velpar.find(framei) != _map_frame_velpar.end() && framei != framej) {

            // add IMU factor, unless the interval is longer than kMaxImuFactorDt, IMU data is missing in it or its
            // covariance is unusable (the bias random walk, weighted by the interval, still holds in every case)
            if (imuFactorUsable(framei, framej)) {
                ceres::CostFunction *cost_fct = new IMUFactor(framei->getIMU(), framej->getIMU());
                problem.AddResidualBlock(cost_fct,
                                         nullptr,
                                         _map_frame_posepar.at(framei).values(),
                                         _map_frame_posepar.at(framej).values(),
                                         _map_frame_velpar.at(framei).values(),
                                         _map_frame_velpar.at(framej).values(),
                                         _map_frame_dbapar.at(framei).values(),
                                         _map_frame_dbgpar.at(framei).values());
                n_imu_factors++;
            }

            // add Bias random walk factor
            ceres::CostFunction *cost_fct_bias = new IMUBiasFactor(framei->getIMU(), framej->getIMU());
            problem.AddResidualBlock(cost_fct_bias,
                                     nullptr,
                                     _map_frame_dbapar.at(framei).values(),
                                     _map_frame_dbgpar.at(framei).values(),
                                     _map_frame_dbapar.at(framej).values(),
                                     _map_frame_dbgpar.at(framej).values());
        }
    }
    return n_imu_factors;
}

size_t AOptimizer::fixedFramesGivenPrior(const std::vector<std::shared_ptr<Frame>> &frame_vector,
                                         size_t requested) const {
    const std::shared_ptr<Frame> &kept = _marginalization->_frame_to_keep;
    if (!kept || !kept->getIMU() || !_marginalization->_has_prior)
        return requested;
    if (std::find(frame_vector.begin(), frame_vector.end(), kept) == frame_vector.end())
        return requested;
    return 0;
}

void AOptimizer::restoreGauge(const std::shared_ptr<Frame> &anchor, const Eigen::Affine3d &T_w_anchor_before) {
    const Eigen::Affine3d T_w_anchor_after = anchor->getFrame2WorldTransform();

    // Rotation about z closest to R_before R_after^T, then the translation that puts the anchor back
    const Eigen::Matrix3d M = T_w_anchor_before.rotation() * T_w_anchor_after.rotation().transpose();
    const double yaw        = std::atan2(M(1, 0) - M(0, 1), M(0, 0) + M(1, 1));
    Eigen::Affine3d G       = Eigen::Affine3d::Identity();
    G.linear()              = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    G.translation()         = T_w_anchor_before.translation() - G.linear() * T_w_anchor_after.translation();

    for (auto &frame_posepar : _map_frame_posepar) {
        std::shared_ptr<Frame> f = frame_posepar.first;
        f->setWorld2FrameTransform((G * f->getFrame2WorldTransform()).inverse());
        if (f->getIMU() && _map_frame_velpar.count(f))
            f->getIMU()->setVelocity(G.linear() * f->getIMU()->getVelocity());
    }
    // Points only translate: they have no orientation, and the sparse prior's landmark factors take their increments
    // along world axes (a rotated point pose made the reprojection factors take them along that rotation instead)
    for (auto &lmk_ptpar : _map_lmk_ptpar) {
        Eigen::Affine3d T = lmk_ptpar.first->getPose();
        T.translation()   = G * T.translation();
        lmk_ptpar.first->setPose(T);
    }
    for (auto &lmk_posepar : _map_lmk_posepar)
        lmk_posepar.first->setPose(G * lmk_posepar.first->getPose());
}

bool AOptimizer::landmarkOptimization(std::shared_ptr<Frame> &frame) {

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(std::sqrt(1.345));

    // Get point cloud to be optimized
    typed_vec_landmarks cloud_to_optimize = frame->getLandmarks();
    addLandmarkResiduals(problem, loss_function, cloud_to_optimize);

    // Landmarks tied to a marginalization prior are estimated by the window, together with that prior. Moving them
    // here (poses fixed, prior ignored) would leave them in conflict with the prior, which measures them from its
    // linearization point: they are kept constant
    for (auto &ldmk_list : cloud_to_optimize) {
        for (auto &ldmk : ldmk_list.second) {
            if (!ldmk->hasPrior())
                continue;
            auto it_pt = _map_lmk_ptpar.find(ldmk);
            if (it_pt != _map_lmk_ptpar.end())
                problem.SetParameterBlockConstant(it_pt->second.values());
            auto it_pose = _map_lmk_posepar.find(ldmk);
            if (it_pose != _map_lmk_posepar.end())
                problem.SetParameterBlockConstant(it_pose->second.values());
        }
    }

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 10;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.function_tolerance                 = 1e-3;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.num_threads                        = 4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::mutex opti_mtx;
    std::lock_guard<std::mutex> lock(opti_mtx);

    // Chi2 test
    for (auto &ldmk_list : cloud_to_optimize) {
        for (auto &ldmk : ldmk_list.second) {

            // Same check as in AddResiduals
            if (!ldmk->isInitialized() || ldmk->isOutlier())
                continue;

            bool can_be_updated = ldmk->sanityCheck(); // Only inliers can be updated

            if (can_be_updated) {
                if (ldmk_list.first == "pointxd") {
                    ldmk->setPose(ldmk->getPose() * _map_lmk_ptpar.at(ldmk).getPose());
                } else {
                    ldmk->setPose(ldmk->getPose() * _map_lmk_posepar.at(ldmk).getPose());
                }
            }
        }
    }

    // Clear maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();

    return true;
}

bool AOptimizer::singleFrameOptimization(std::shared_ptr<isae::Frame> &moving_frame) {

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = nullptr;

    // Get point cloud to be optimized
    typed_vec_landmarks cloud_to_optimize = moving_frame->getLandmarks();

    // Add residuals
    addSingleFrameResiduals(problem, loss_function, moving_frame, cloud_to_optimize);

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY; // SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 5;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 1;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Update state
    for (auto &frame_posepar : _map_frame_posepar) {
        frame_posepar.first->setWorld2FrameTransform(frame_posepar.first->getWorld2FrameTransform() *
                                                     frame_posepar.second.getPose());
    }

    // Set maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();

    // std::cout << summary.FullReport() << std::endl;

    return true;
}

bool AOptimizer::singleFrameVIOptimization(std::shared_ptr<isae::Frame> &moving_frame) {

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(std::sqrt(1.345));

    // Get point cloud to be optimized
    typed_vec_landmarks cloud_to_optimize = moving_frame->getLandmarks();

    // Add visual residuals
    addSingleFrameResiduals(problem, loss_function, moving_frame, cloud_to_optimize);

    // Add IMU residuals
    std::vector<std::shared_ptr<Frame>> frame_vec;
    if (moving_frame->getIMU() && moving_frame->getIMU()->getLastKF()->getIMU()) {
        frame_vec.push_back(moving_frame);
        frame_vec.push_back(moving_frame->getIMU()->getLastKF());
        std::shared_ptr<Frame> frame = moving_frame->getIMU()->getLastKF();

        addSingleFrameResiduals(problem, loss_function, frame, cloud_to_optimize);

        auto ordering = new ceres::ParameterBlockOrdering;
        addIMUResiduals(problem, nullptr, ordering, frame_vec, 0);
    }

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 5;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 1;
    options.max_solver_time_in_seconds         = 0.005;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (!summary.IsSolutionUsable())
        return false;

    // Update state
    for (auto &frame_posepar : _map_frame_posepar) {
        frame_posepar.first->setWorld2FrameTransform(frame_posepar.first->getWorld2FrameTransform() *
                                                     frame_posepar.second.getPose());
    }

    // For IMU
    if (moving_frame->getIMU() && moving_frame->getIMU()->getLastKF()->getIMU()) {
        for (auto &frame_velpar : _map_frame_velpar) {
            frame_velpar.first->getIMU()->setVelocity(frame_velpar.first->getIMU()->getVelocity() +
                                                      frame_velpar.second.getPose().translation());
        }

        for (auto &frame_dbapar : _map_frame_dbapar) {
            frame_dbapar.first->getIMU()->setBa(frame_dbapar.first->getIMU()->getBa() +
                                                frame_dbapar.second.getPose().translation());
        }

        for (auto &frame_dbgpar : _map_frame_dbgpar) {
            frame_dbgpar.first->getIMU()->setBg(frame_dbgpar.first->getIMU()->getBg() +
                                                frame_dbgpar.second.getPose().translation());
        }
    }

    // Set maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // std::cout << summary.FullReport() << std::endl;

    return true;
}

bool AOptimizer::localMapBA(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_sized_number) {

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = _robust_visual_vo ? newVisualLoss() : nullptr;

    // Get all moving frames
    std::vector<std::shared_ptr<isae::Frame>> frame_vector;
    local_map->getLastNFramesIn(local_map->getMapSize(), frame_vector);

    // Add residuals
    auto ordering = new ceres::ParameterBlockOrdering;
    addResidualsLocalMap(problem, loss_function, ordering, frame_vector, fixed_sized_number, local_map);
    addMarginalizationResiduals(problem, nullptr, ordering); // the prior is never robustified

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 20;
    options.minimizer_progress_to_stdout       = false;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (!summary.IsSolutionUsable())
        return discardFailedSolve(summary);

    // Update states
    for (auto &frame_posepar : _map_frame_posepar) {
        frame_posepar.first->setWorld2FrameTransform(frame_posepar.first->getWorld2FrameTransform() *
                                                     frame_posepar.second.getPose());
    }

    for (auto &lmk_posepar : _map_lmk_posepar) {
        lmk_posepar.first->setPose(lmk_posepar.first->getPose() * lmk_posepar.second.getPose());
    }

    for (auto &lmk_ptpar : _map_lmk_ptpar) {
        lmk_ptpar.first->setPose(lmk_ptpar.first->getPose() * lmk_ptpar.second.getPose());
    }

    // Clear maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();

    // std::cout << summary.FullReport() << std::endl;

    return true;
}

bool AOptimizer::localMapVIOptimization(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_sized_number) {

    // Set maps for bookeeping;
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = newVisualLoss(); // visual factors only
    // Get all moving frames
    std::vector<std::shared_ptr<isae::Frame>> frame_vector;
    local_map->getLastNFramesIn(local_map->getMapSize(), frame_vector);
    const size_t n_fixed = fixedFramesGivenPrior(frame_vector, fixed_sized_number);
    // Without a fixed frame (prior-anchored window), the oldest frame keeps its yaw and position (gauge)
    std::shared_ptr<Frame> gauge_anchor =
        (n_fixed == 0 && fixed_sized_number > 0 && !frame_vector.empty()) ? frame_vector.back() : nullptr;
    const Eigen::Affine3d T_w_anchor_before =
        gauge_anchor ? gauge_anchor->getFrame2WorldTransform() : Eigen::Affine3d::Identity();

    // Add residuals
    auto ordering = new ceres::ParameterBlockOrdering;
    addResidualsLocalMap(problem, loss_function, ordering, frame_vector, n_fixed, local_map);
    uint n_imu_factors   = addIMUResiduals(problem, nullptr, ordering, frame_vector, n_fixed);
    const int n_blocks   = problem.NumResidualBlocks();
    addMarginalizationResiduals(problem, nullptr, ordering);
    uint n_prior_factors = problem.NumResidualBlocks() - n_blocks;

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.linear_solver_ordering.reset(ordering);
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = 20;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    recordVIStats(summary, n_imu_factors, n_prior_factors, problem.NumResidualBlocks());
    recordCostsPerType(problem);
    if (!summary.IsSolutionUsable())
        return discardFailedSolve(summary);

    // Update state
    for (auto &frame_posepar : _map_frame_posepar) {
        frame_posepar.first->setWorld2FrameTransform(frame_posepar.first->getWorld2FrameTransform() *
                                                     frame_posepar.second.getPose());
    }

    for (auto &lmk_posepar : _map_lmk_posepar) {
        lmk_posepar.first->setPose(lmk_posepar.first->getPose() * lmk_posepar.second.getPose());
    }

    for (auto &lmk_ptpar : _map_lmk_ptpar) {
        lmk_ptpar.first->setPose(lmk_ptpar.first->getPose() * lmk_ptpar.second.getPose());
    }

    // For IMU
    for (auto &frame_velpar : _map_frame_velpar) {
        frame_velpar.first->getIMU()->setVelocity(frame_velpar.first->getIMU()->getVelocity() +
                                                  frame_velpar.second.getPose().translation());
    }

    for (auto &frame_dbapar : _map_frame_dbapar) {
        frame_dbapar.first->getIMU()->setBa(frame_dbapar.first->getIMU()->getBa() +
                                            frame_dbapar.second.getPose().translation());
    }

    for (auto &frame_dbgpar : _map_frame_dbgpar) {
        frame_dbgpar.first->getIMU()->setBg(frame_dbgpar.first->getIMU()->getBg() +
                                            frame_dbgpar.second.getPose().translation());
    }

    if (gauge_anchor)
        restoreGauge(gauge_anchor, T_w_anchor_before);

    // Re-integrate the preintegrations whose linearization bias is now too far
    repropagateIfNeeded(frame_vector);

    // Set maps for bookeeping;
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // std::cout << summary.FullReport() << std::endl;

    return true;
}

bool AOptimizer::discardFailedSolve(const ceres::Solver::Summary &summary) {
    std::cerr << "Window optimization failed, states kept: " << summary.message << std::endl;
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();
    return false;
}

double AOptimizer::VIInit(std::shared_ptr<isae::LocalMap> &local_map, Eigen::Matrix3d &R_w_i, bool optim_scale) {
    int steps = 50;

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = nullptr;

    // Get all moving frames
    std::vector<std::shared_ptr<isae::Frame>> frame_vector;
    local_map->getLastNFramesIn(local_map->getMapSize(), frame_vector);

    // Add parameter blocks for the velocity of the IMU
    for (auto frame : frame_vector) {
        if (frame->getIMU()) {
            _map_frame_velpar.emplace(frame, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        }
    }

    // Parameter block of the gravity direction
    double r_wi_par[2] = {0.0, 0.0};
    problem.AddParameterBlock(r_wi_par, 2);

    // Parameter blocks of the delta bias (assumed constant on this sliding window)
    PointXYZParametersBlock dba_par = PointXYZParametersBlock(Eigen::Vector3d(0, 0, 0));
    PointXYZParametersBlock dbg_par = PointXYZParametersBlock(Eigen::Vector3d(0, 0, 0));
    problem.AddParameterBlock(dba_par.values(), 3);
    problem.AddParameterBlock(dbg_par.values(), 3);

    // Parameter block of the scale, that is set to 0 as it goes in an exponential
    double lambda[1] = {0.0};
    problem.AddParameterBlock(lambda, 1);
    if (!optim_scale)
        problem.SetParameterBlockConstant(lambda);

    for (auto framej : frame_vector) {

        std::shared_ptr<Frame> framei = framej->getIMU()->getLastKF();

        // Skip intervals with missing IMU data
        if (_map_frame_velpar.find(framei) != _map_frame_velpar.end() && imuFactorUsable(framei, framej)) {

            // add IMU factor
            ceres::CostFunction *cost_fct = new IMUFactorInit(framei->getIMU(), framej->getIMU());
            problem.AddResidualBlock(cost_fct,
                                     loss_function,
                                     r_wi_par,
                                     _map_frame_velpar.at(framei).values(),
                                     _map_frame_velpar.at(framej).values(),
                                     dba_par.values(),
                                     dbg_par.values(),
                                     lambda);
        }
    }

    // Priors on the bias changes w.r.t. the static initial estimate (as in ORB-SLAM3's inertial-only
    // initialization): the gyroscope bias is well observed through the rotations, so its prior is weak;
    // the accelerometer bias is barely observable over a short window, so its change is kept small.
    const double sigma_dba      = 0.01; // m/s^2
    const double sigma_dbg      = 0.1;  // rad/s
    Eigen::Matrix3d sqrt_inf_ba = Eigen::Matrix3d::Identity() / sigma_dba;
    ceres::CostFunction *cost_fct_ba =
        new Landmark3DPrior(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), sqrt_inf_ba);
    problem.AddResidualBlock(cost_fct_ba, loss_function, dba_par.values());
    Eigen::Matrix3d sqrt_inf_bg = Eigen::Matrix3d::Identity() / sigma_dbg;
    ceres::CostFunction *cost_fct_bg =
        new Landmark3DPrior(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), sqrt_inf_bg);
    problem.AddResidualBlock(cost_fct_bg, loss_function, dbg_par.values());

    // Solve the problem we just built
    ceres::Solver::Options options;
    options.trust_region_strategy_type         = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type                 = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations                 = steps;
    options.minimizer_progress_to_stdout       = false;
    options.use_explicit_schur_complement      = true;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.function_tolerance                 = 1.e-3;
    options.num_threads                        = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Do not apply an unusable solution: the caller restarts the initialization
    if (!summary.IsSolutionUsable()) {
        std::cout << "Inertial initialization failed: " << summary.message << std::endl;
        _map_frame_velpar.clear();
        return -1;
    }

    // Mono: the motion so far must determine the scale. At constant velocity without rotation (or with too little
    // acceleration) any scale fits with a matching velocity, and the solver stops anywhere (scale -> 0 in practice).
    // The uncertainty of the log scale at the solution decides; it cannot be computed when the scale is free.
    if (optim_scale) {
        ceres::Covariance::Options cov_options;
        cov_options.algorithm_type = ceres::DENSE_SVD;
        ceres::Covariance covariance(cov_options);
        const std::vector<std::pair<const double *, const double *>> blocks = {{lambda, lambda}};
        double var_log_s = std::numeric_limits<double>::infinity();
        if (covariance.Compute(blocks, &problem))
            covariance.GetCovarianceBlock(lambda, lambda, &var_log_s);
        const double sigma_log_s = std::sqrt(var_log_s);
        if (!(sigma_log_s < kMaxInitScaleSigma)) {
            std::cout << "Inertial initialization rejected: scale " << std::exp(lambda[0])
                      << " not determined by the motion (sigma log scale " << sigma_log_s << ")" << std::endl;
            _map_frame_velpar.clear();
            return -1;
        }
        std::cout << "Inertial initialization: scale " << std::exp(lambda[0]) << ", sigma log scale " << sigma_log_s
                  << std::endl;
    }

    // Update IMU velocity and biases (the bias change is shared by the whole window)
    const Eigen::Vector3d dba = dba_par.getPose().translation();
    const Eigen::Vector3d dbg = dbg_par.getPose().translation();
    for (auto &frame_velpar : _map_frame_velpar) {
        std::shared_ptr<IMU> imu = frame_velpar.first->getIMU();
        imu->setVelocity(imu->getVelocity() + frame_velpar.second.getPose().translation());
        imu->setBa(imu->getBa() + dba);
        imu->setBg(imu->getBg() + dbg);
    }

    // The factors correct the deltas for the new biases; re-integrate where the change is too large
    repropagateIfNeeded(frame_vector);

    // Update gravity direction with only 2DoF
    R_w_i = geometry::exp_so3(Eigen::Vector3d(r_wi_par[0], r_wi_par[1], 0));

    // Update frame poses
    Eigen::Affine3d T_w_i            = Eigen::Affine3d::Identity();
    T_w_i.affine().block(0, 0, 3, 3) = R_w_i;
    for (auto &frame : frame_vector) {

        // Apply scale + rotate on inertial frame
        Eigen::Affine3d T_f_w = frame->getWorld2FrameTransform();
        T_f_w.translation() *= std::exp(lambda[0]);
        T_f_w = T_f_w * T_w_i;
        frame->setWorld2FrameTransform(T_f_w);

        if (frame->hasPrior()) {
            frame->setPrior(frame->getWorld2FrameTransform(), 100 * Vector6d::Ones());
        }
    }

    // Update landmarks
    for (auto &tlandmark : local_map->getLandmarks()) {
        for (auto landmark : tlandmark.second) {
            if (!landmark->isOutlier()) {
                Eigen::Affine3d T_w_lmk = T_w_i.inverse() * landmark->getPose();
                T_w_lmk.translation() *= std::exp(lambda[0]);
                landmark->setPose(T_w_lmk);
            }
        }
    }

    std::cout << summary.FullReport() << std::endl;
    std::cout << "Scale : " << std::exp(lambda[0]) << std::endl;

    // Set maps for bookeeping
    _map_frame_posepar.clear();
    _map_lmk_ptpar.clear();
    _map_lmk_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    return std::exp(lambda[0]);
}

// To be implemented in dedicated solvers
bool AOptimizer::landmarkOptimizationNoFov(std::shared_ptr<Frame> &f,
                                           std::shared_ptr<Frame> &fp,
                                           Eigen::Affine3d &T_cam0_cam0p,
                                           double info_scale) {
    return true;
}

} // namespace isae