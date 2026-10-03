#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"

namespace isae {

bool BundleAdjustmentCERESAnalytic::localMapVIOptimizationTd(std::shared_ptr<isae::LocalMap> &local_map,
                                                             double &td,
                                                             const size_t fixed_frame_number) {
    // Set maps for bookeeping;
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // Build the Bundle Adjustement Problem
    ceres::Problem problem;
    ceres::LossFunction *loss_function = newVisualLoss(); // visual factors only
    auto ordering                      = new ceres::ParameterBlockOrdering;

    // Get all moving frames
    std::vector<std::shared_ptr<isae::Frame>> frame_vector;
    local_map->getLastNFramesIn(local_map->getMapSize(), frame_vector);
    const size_t n_fixed = fixedFramesGivenPrior(frame_vector, fixed_frame_number);
    // Without a fixed frame (prior-anchored window), the oldest frame keeps its yaw and position (gauge)
    std::shared_ptr<Frame> gauge_anchor =
        (n_fixed == 0 && fixed_frame_number > 0 && !frame_vector.empty()) ? frame_vector.back() : nullptr;
    const Eigen::Affine3d T_w_anchor_before =
        gauge_anchor ? gauge_anchor->getFrame2WorldTransform() : Eigen::Affine3d::Identity();
    double t_delay[1] = {td};
    problem.AddParameterBlock(t_delay, 1);
    ordering->AddElementToGroup(t_delay, 1);

    // Add residuals
    for (size_t i = 0; i < frame_vector.size(); i++) {

        std::shared_ptr<Frame> frame = frame_vector.at(i);
        _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));

        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
        ordering->AddElementToGroup(_map_frame_posepar.at(frame).values(), 1);

        // Set parameter block constant for fixed frames
        if ((int)i > (int)(frame_vector.size() - n_fixed - 1)) {
            problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
        }

        // Had a prior factor if it exists
        if (frame->hasPrior()) {
            ceres::CostFunction *cost_fct =
                new PosePriordx(frame->getWorld2FrameTransform(), frame->getPrior(), frame->getInfPrior().asDiagonal());
            problem.AddResidualBlock(cost_fct, loss_function, _map_frame_posepar.at(frame).values());
        }
    }

    // For all the landmarks
    for (auto &ldmk_list : local_map->getLandmarks()) {

        // Deal with pointxd landmarks
        if (ldmk_list.first == "pointxd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // Add parameter block for each landmark
                _map_lmk_ptpar.emplace(landmark, PointXYZParametersBlock(Eigen::Vector3d::Zero()));

                problem.AddParameterBlock(_map_lmk_ptpar.at(landmark).values(), 3);
                ordering->AddElementToGroup(_map_lmk_ptpar.at(landmark).values(), 0);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();
                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> frame      = cam->getFrame();
                    std::shared_ptr<IMU> imu          = frame->getIMU();

                    // Check the consistency of the frame
                    if (!feature || !frame->isKeyFrame() ||
                        _map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        continue;
                    }

                    if (!feature->getVelocity().empty()) {

                        ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx_td(
                            feature->getPoints().at(0), cam, imu, landmark->getPose(), 1.0, frame->getTimeOffset());

                        problem.AddResidualBlock(cost_fct,
                                                 loss_function,
                                                 _map_frame_posepar.at(frame).values(),
                                                 _map_lmk_ptpar.at(landmark).values(),
                                                 t_delay);
                    } else {
                        ceres::CostFunction *cost_fct =
                            new ReprojectionErrCeres_pointxd_dx(feature->getPoints().at(0), cam, landmark->getPose());

                        problem.AddResidualBlock(cost_fct,
                                                 loss_function,
                                                 _map_frame_posepar.at(frame).values(),
                                                 _map_lmk_ptpar.at(landmark).values());
                    }
                }
            }
        }
    }
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

    std::cout << summary.FullReport() << std::endl;
    std::cout << t_delay[0] << std::endl;

    // Set maps for bookeeping;
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    td = t_delay[0];

    return true;
}

uint BundleAdjustmentCERESAnalytic::addSingleFrameResiduals(ceres::Problem &problem,
                                                            ceres::LossFunction *loss_function,
                                                            std::shared_ptr<Frame> &frame,
                                                            typed_vec_landmarks &cloud_to_optimize) {

    // ceres::Manifold *nullptr = new SE3RightParameterization();
    _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));
    problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);

    for (auto &ldmk_list : cloud_to_optimize) {

        // Deal with pointxd landmarks
        if (ldmk_list.first == "pointxd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();

                    if (!feature) {
                        continue;
                    }

                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> feat_frame = cam->getFrame();

                    if (frame == feat_frame) {

                        _map_lmk_ptpar.emplace(landmark, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                        problem.AddParameterBlock(_map_lmk_ptpar.at(landmark).values(), 3);
                        problem.SetParameterBlockConstant(_map_lmk_ptpar.at(landmark).values());

                        ceres::CostFunction *cost_fct =
                            new ReprojectionErrCeres_pointxd_dx(feature->getPoints().at(0), cam, landmark->getPose());

                        problem.AddResidualBlock(cost_fct,
                                                 loss_function,
                                                 _map_frame_posepar.at(frame).values(),
                                                 _map_lmk_ptpar.at(landmark).values());
                    }
                }
            }
        }

        // Deal with linexd landmarks
        if (ldmk_list.first == "linexd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();

                    if (!feature) {
                        continue;
                    }

                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> feat_frame = cam->getFrame();

                    if (frame == feat_frame) {

                        _map_lmk_posepar.emplace(landmark, PoseParametersBlock(Eigen::Affine3d::Identity()));
                        problem.AddParameterBlock(_map_lmk_posepar.at(landmark).values(), 6);
                        problem.SetParameterBlockConstant(_map_lmk_posepar.at(landmark).values());

                        ceres::CostFunction *cost_fct = new ReprojectionErrCeres_linexd_dx(
                            feature->getPoints(), cam, landmark->getPose(), landmark->getModel(), 1);

                        problem.AddResidualBlock(cost_fct,
                                                 loss_function,
                                                 _map_frame_posepar.at(frame).values(),
                                                 _map_lmk_posepar.at(landmark).values());
                    }
                }
            }
        }
    }

    return 0;
}

uint BundleAdjustmentCERESAnalytic::addLandmarkResiduals(ceres::Problem &problem,
                                                         ceres::LossFunction *loss_function,
                                                         typed_vec_landmarks &cloud_to_optimize) {

    // For all the landmarks
    for (auto &ldmk_list : cloud_to_optimize) {

        // Deal with pointxd landmarks
        if (ldmk_list.first == "pointxd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // Add parameter block for each landmark
                _map_lmk_ptpar.emplace(landmark, PointXYZParametersBlock(Eigen::Vector3d::Zero()));

                problem.AddParameterBlock(_map_lmk_ptpar.at(landmark).values(), 3);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();
                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> frame      = cam->getFrame();

                    // Check the consistency of the frame
                    if (!feature || !frame->isKeyFrame()) {
                        continue;
                    }

                    if (_map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));
                        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
                        problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
                    }

                    ceres::CostFunction *cost_fct =
                        new ReprojectionErrCeres_pointxd_dx(feature->getPoints().at(0), cam, landmark->getPose());
                    problem.AddResidualBlock(cost_fct,
                                             loss_function,
                                             _map_frame_posepar.at(frame).values(),
                                             _map_lmk_ptpar.at(landmark).values());
                }
            }
        }

        // Deal with linexd landmarks
        if (ldmk_list.first == "linexd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // Add parameter block for each landmark
                _map_lmk_posepar.emplace(landmark, PoseParametersBlock(Eigen::Affine3d::Identity()));

                problem.AddParameterBlock(_map_lmk_posepar.at(landmark).values(), 6);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();

                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();
                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> frame      = cam->getFrame();

                    // Check the consistency of the frame
                    if (!feature || !frame->isKeyFrame()) {
                        continue;
                    }

                    if (_map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        _map_frame_posepar.emplace(frame, PoseParametersBlock(Eigen::Affine3d::Identity()));
                        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
                        problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
                    }

                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_linexd_dx(
                        feature->getPoints(), cam, landmark->getPose(), landmark->getModel(), 1);
                    problem.AddResidualBlock(cost_fct,
                                             loss_function,
                                             _map_frame_posepar.at(frame).values(),
                                             _map_lmk_posepar.at(landmark).values());
                }
            }
        }
    }

    return 0;
}

uint BundleAdjustmentCERESAnalytic::addResidualsLocalMap(ceres::Problem &problem,
                                                         ceres::LossFunction *loss_function,
                                                         ceres::ParameterBlockOrdering *ordering,
                                                         std::vector<std::shared_ptr<Frame>> &frame_vector,
                                                         size_t fixed_frame_number,
                                                         std::shared_ptr<isae::LocalMap> &local_map) {

    uint nb_residuals = 0;

    // Add parameter block and ordering for each frame parameter
    for (size_t i = 0; i < frame_vector.size(); i++) {
        _map_frame_posepar.emplace(frame_vector.at(i), PoseParametersBlock(Eigen::Affine3d::Identity()));
    }

    for (size_t i = 0; i < frame_vector.size(); i++) {

        std::shared_ptr<Frame> frame = frame_vector.at(i);

        problem.AddParameterBlock(_map_frame_posepar.at(frame).values(), 6);
        ordering->AddElementToGroup(_map_frame_posepar.at(frame).values(), 1);

        // Set parameter block constant for fixed frames
        if ((int)i > (int)(frame_vector.size() - fixed_frame_number - 1)) {
            problem.SetParameterBlockConstant(_map_frame_posepar.at(frame).values());
        }

        // Had a prior factor if it exists
        if (frame->hasPrior()) {
            ceres::CostFunction *cost_fct =
                new PosePriordx(frame->getWorld2FrameTransform(), frame->getPrior(), frame->getInfPrior().asDiagonal());
            problem.AddResidualBlock(cost_fct, loss_function, _map_frame_posepar.at(frame).values());
        }
    }

    // For all the landmarks
    for (auto &ldmk_list : local_map->getLandmarks()) {

        // Deal with pointxd landmarks
        if (ldmk_list.first == "pointxd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // Add parameter block for each landmark
                _map_lmk_ptpar.emplace(landmark, PointXYZParametersBlock(Eigen::Vector3d::Zero()));

                problem.AddParameterBlock(_map_lmk_ptpar.at(landmark).values(), 3);
                ordering->AddElementToGroup(_map_lmk_ptpar.at(landmark).values(), 0);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();
                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();
                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> frame      = cam->getFrame();

                    // Check the consistency of the frame
                    if (!feature || !frame->isKeyFrame() ||
                        _map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        continue;
                    }

                    nb_residuals++;

                    ceres::CostFunction *cost_fct =
                        new ReprojectionErrCeres_pointxd_dx(feature->getPoints().at(0), cam, landmark->getPose());
                    problem.AddResidualBlock(cost_fct,
                                             loss_function,
                                             _map_frame_posepar.at(frame).values(),
                                             _map_lmk_ptpar.at(landmark).values());
                }
            }
        }

        // Deal with linexd landmarks
        if (ldmk_list.first == "linexd") {
            // For all landmark
            for (auto &landmark : ldmk_list.second) {

                if (!landmark->isInitialized() || landmark->isOutlier())
                    continue;

                // Add parameter block for each landmark
                _map_lmk_posepar.emplace(landmark, PoseParametersBlock(Eigen::Affine3d::Identity()));

                problem.AddParameterBlock(_map_lmk_posepar.at(landmark).values(), 6);
                ordering->AddElementToGroup(_map_lmk_posepar.at(landmark).values(), 2);

                // For all feature
                std::vector<std::weak_ptr<AFeature>> featuresAssociatedLandmarks = landmark->getFeatures();
                for (std::weak_ptr<AFeature> &wfeature : featuresAssociatedLandmarks) {
                    std::shared_ptr<AFeature> feature = wfeature.lock();
                    std::shared_ptr<ImageSensor> cam  = feature->getSensor();
                    std::shared_ptr<Frame> frame      = cam->getFrame();

                    // Check the consistency of the frame
                    if (!feature || !frame->isKeyFrame() ||
                        _map_frame_posepar.find(frame) == _map_frame_posepar.end()) {
                        continue;
                    }

                    nb_residuals++;

                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_linexd_dx(
                        feature->getPoints(), cam, landmark->getPose(), landmark->getModel(), 1);

                    problem.AddResidualBlock(cost_fct,
                                             loss_function,
                                             _map_frame_posepar.at(frame).values(),
                                             _map_lmk_posepar.at(landmark).values());
                }
            }
        }
    }
    return nb_residuals;
}

uint BundleAdjustmentCERESAnalytic::addMarginalizationResiduals(ceres::Problem &problem,
                                                                ceres::LossFunction *loss_function,
                                                                ceres::ParameterBlockOrdering *ordering) {

    // Add marginalization factor, dense case
    if (!_marginalization->_lmk_to_keep.empty() && !_enable_sparsif) {
        // Get parameter blocks for marginalization
        std::vector<double *> prior_parameter_blocks;

        // Add frame to keep blocks
        if (_marginalization->_frame_to_keep) {
            ordering->Remove(_map_frame_posepar.at(_marginalization->_frame_to_keep).values());
            prior_parameter_blocks.push_back(_map_frame_posepar.at(_marginalization->_frame_to_keep).values());
            ordering->AddElementToGroup(_map_frame_posepar.at(_marginalization->_frame_to_keep).values(), 2);

            ordering->Remove(_map_frame_velpar.at(_marginalization->_frame_to_keep).values());
            prior_parameter_blocks.push_back(_map_frame_velpar.at(_marginalization->_frame_to_keep).values());
            ordering->AddElementToGroup(_map_frame_velpar.at(_marginalization->_frame_to_keep).values(), 2);

            ordering->Remove(_map_frame_dbapar.at(_marginalization->_frame_to_keep).values());
            prior_parameter_blocks.push_back(_map_frame_dbapar.at(_marginalization->_frame_to_keep).values());
            ordering->AddElementToGroup(_map_frame_dbapar.at(_marginalization->_frame_to_keep).values(), 2);

            ordering->Remove(_map_frame_dbgpar.at(_marginalization->_frame_to_keep).values());
            prior_parameter_blocks.push_back(_map_frame_dbgpar.at(_marginalization->_frame_to_keep).values());
            ordering->AddElementToGroup(_map_frame_dbgpar.at(_marginalization->_frame_to_keep).values(), 2);
        }

        // Add lmk to keep blocks
        for (auto tlmk : _marginalization->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                // TODO: fix this, it is supposed to be in the map
                if (_map_lmk_ptpar.find(lmk) == _map_lmk_ptpar.end()) {
                    _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
                    problem.AddParameterBlock(_map_lmk_ptpar.at(lmk).values(), 3);
                    ordering->AddElementToGroup(_map_lmk_ptpar.at(lmk).values(), 0);
                }
                ordering->Remove(_map_lmk_ptpar.at(lmk).values());
                prior_parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());
                ordering->AddElementToGroup(_map_lmk_ptpar.at(lmk).values(), 2);
            }
        }

        MarginalizationFactor *cost_fct = new MarginalizationFactor(_marginalization);
        recordPriorDiag(cost_fct, prior_parameter_blocks);
        problem.AddResidualBlock(cost_fct, loss_function, prior_parameter_blocks);
    }

    // Add marginalization factor, sparse case
    if (_enable_sparsif)
        addSparsePriorResiduals(problem, loss_function, ordering);

    return 0;
}

bool BundleAdjustmentCERESAnalytic::marginalize(std::shared_ptr<Frame> &frame0,
                                                std::shared_ptr<Frame> &frame1,
                                                bool enable_sparsif) {
    if (frame0 == frame1)
        return false;
    _enable_sparsif = enable_sparsif;

    // Setup the maps for memory gestion
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // Select the nodes to marginalize / keep
    _marginalization->preMarginalize(frame0, frame1, _marginalization_last);

    // Visual factors are marginalized with the loss of the window they come from (robust in VIO, and in VO
    // when marginalization is enabled)
    std::shared_ptr<ceres::LossFunction> visual_loss((frame0->getIMU() || _robust_visual_vo) ? newVisualLoss()
                                                                                           : nullptr);

    // Create pose parameters for the frame to marginalize
    _map_frame_posepar.emplace(frame0, PoseParametersBlock(Eigen::Affine3d::Identity()));
    _map_frame_posepar.emplace(frame1, PoseParametersBlock(Eigen::Affine3d::Identity()));

    // Create Marginalization Blocks with pre-integration factors in the case of VIO
    if (frame0->getIMU() && frame1->getIMU()) {

        // Create velo and bias parameters for both frames
        _map_frame_velpar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbapar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbgpar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_velpar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbapar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbgpar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));

        // Parameters of marginalization blocks
        std::vector<double *> parameter_blocks;
        std::vector<int> parameter_idx;

        // For the frames
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));
        parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame1));
        parameter_blocks.push_back(_map_frame_posepar.at(frame1).values());

        // For the velocities
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0) + 6);
        parameter_blocks.push_back(_map_frame_velpar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame1) + 6);
        parameter_blocks.push_back(_map_frame_velpar.at(frame1).values());

        // For the biases
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0) + 9);
        parameter_blocks.push_back(_map_frame_dbapar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0) + 12);
        parameter_blocks.push_back(_map_frame_dbgpar.at(frame0).values());

        // Add the pre integration factor in the marginalization scheme, only if frame1's
        // preintegration starts at frame0 and is usable (the bias random walk always holds)
        if (imuFactorUsable(frame0, frame1)) {
            ceres::CostFunction *cost_fct = new IMUFactor(frame0->getIMU(), frame1->getIMU());
            _marginalization->_marginalization_blocks.push_back(
                std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
        }

        // Parameters of marginalization blocks
        std::vector<double *> parameter_blocks_b;
        std::vector<int> parameter_idx_b;

        // For the biases
        parameter_idx_b.push_back(_marginalization->_map_frame_idx.at(frame0) + 9);
        parameter_blocks_b.push_back(_map_frame_dbapar.at(frame0).values());
        parameter_idx_b.push_back(_marginalization->_map_frame_idx.at(frame0) + 12);
        parameter_blocks_b.push_back(_map_frame_dbgpar.at(frame0).values());
        parameter_idx_b.push_back(_marginalization->_map_frame_idx.at(frame1) + 9);
        parameter_blocks_b.push_back(_map_frame_dbapar.at(frame1).values());
        parameter_idx_b.push_back(_marginalization->_map_frame_idx.at(frame1) + 12);
        parameter_blocks_b.push_back(_map_frame_dbgpar.at(frame1).values());

        // Add the bias random walk factor in the marginalization scheme
        ceres::CostFunction *cost_fct_b = new IMUBiasFactor(frame0->getIMU(), frame1->getIMU());
        _marginalization->_marginalization_blocks.push_back(
            std::make_shared<MarginalizationBlockInfo>(cost_fct_b, parameter_idx_b, parameter_blocks_b));
    }

    // Create Marginalization Blocks with landmarks to keep
    for (auto tlmk : _marginalization->_lmk_to_keep) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto feature : lmk->getFeatures()) {
                if (feature.lock()->getSensor()->getFrame() == frame0) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));
                    parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());

                    // For the lmk
                    parameter_idx.push_back(_marginalization->_map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the reprojection factor in the marginalization scheme
                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx(
                        feature.lock()->getPoints().at(0), feature.lock()->getSensor(), lmk->getPose());

                    _marginalization->_marginalization_blocks.push_back(std::make_shared<MarginalizationBlockInfo>(
                        cost_fct, parameter_idx, parameter_blocks, visual_loss));
                }
            }
        }
    }

    // Create Marginalization Blocks with landmark to marginalize
    for (auto tlmk : _marginalization->_lmk_to_marg) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto feature : lmk->getFeatures()) {
                if (feature.lock()->getSensor()->getFrame() == frame0) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));
                    parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());

                    // For the lmk
                    parameter_idx.push_back(_marginalization->_map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the reprojection factor in the marginalization scheme
                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx(
                        feature.lock()->getPoints().at(0), feature.lock()->getSensor(), lmk->getPose());

                    _marginalization->_marginalization_blocks.push_back(std::make_shared<MarginalizationBlockInfo>(
                        cost_fct, parameter_idx, parameter_blocks, visual_loss));
                }
            }
        }
    }

    // Create a marginalization block with previous prior
    if (!_marginalization_last->_lmk_to_keep.empty()) {

        // Compute index and block vectors for marginalization factor
        std::vector<double *> parameter_blocks;
        std::vector<int> parameter_idx;

        // Fill the parameters with previous frame to keep
        if (_marginalization_last->_frame_to_keep) {
            parameter_blocks.push_back(_map_frame_posepar.at(_marginalization_last->_frame_to_keep).values());
            parameter_idx.push_back(_marginalization->_map_frame_idx.at(_marginalization_last->_frame_to_keep));

            parameter_blocks.push_back(_map_frame_velpar.at(_marginalization_last->_frame_to_keep).values());
            parameter_idx.push_back(_marginalization->_map_frame_idx.at(_marginalization_last->_frame_to_keep) + 6);

            parameter_blocks.push_back(_map_frame_dbapar.at(_marginalization_last->_frame_to_keep).values());
            parameter_idx.push_back(_marginalization->_map_frame_idx.at(_marginalization_last->_frame_to_keep) + 9);

            parameter_blocks.push_back(_map_frame_dbgpar.at(_marginalization_last->_frame_to_keep).values());
            parameter_idx.push_back(_marginalization->_map_frame_idx.at(_marginalization_last->_frame_to_keep) + 12);
        }

        // Fill the parameters with previous landmarks kept
        for (auto tlmk : _marginalization_last->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());
                parameter_idx.push_back(_marginalization->_map_lmk_idx.at(lmk));
            }
        }
        // Add the last prior in the marginalization scheme
        ceres::CostFunction *cost_fct = new MarginalizationFactor(_marginalization_last);
        _marginalization->_marginalization_blocks.push_back(
            std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
    }

    // Add frame prior if it exists
    if (frame0->hasPrior()) {
        // Compute index and block vectors for pose prior factor
        std::vector<double *> parameter_blocks;
        std::vector<int> parameter_idx;
        parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));

        ceres::CostFunction *cost_fct =
            new PosePriordx(frame0->getWorld2FrameTransform(), frame0->getPrior(), frame0->getInfPrior().asDiagonal());
        _marginalization->_marginalization_blocks.push_back(
            std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
    }

    // Reset the marginalization scheme if it failed
    if (!_marginalization->computeSchurComplement()) {
        _marginalization->_lmk_to_keep.clear();
        _marginalization->_marginalization_blocks.clear();
        _marginalization_last->_lmk_to_keep.clear();
        return false;
    }

    // Compute the sparse factors
    if (_enable_sparsif) {
        if (_marginalization->_frame_to_keep && _marginalization->_frame_to_keep->getIMU())
            _marginalization->sparsifyVIO();
        else
            _marginalization->sparsifyVO();
    }

    // The jacobians and residuals of the dense factor are computed in every case
    _marginalization->computeJacobiansAndResiduals();

    // Update the last marginalization variable
    _marginalization_last->_map_lmk_prior.clear();
    _marginalization_last->_map_lmk_inf.clear();
    _marginalization_last->_lmk_to_keep              = _marginalization->_lmk_to_keep;
    _marginalization_last->_frame_to_keep            = _marginalization->_frame_to_keep;
    _marginalization_last->_map_frame_idx            = _marginalization->_map_frame_idx;
    _marginalization_last->_map_lmk_idx              = _marginalization->_map_lmk_idx;
    _marginalization_last->_map_frame_inf            = _marginalization->_map_frame_inf;
    _marginalization_last->_map_lmk_inf              = _marginalization->_map_lmk_inf;
    _marginalization_last->_map_lmk_prior            = _marginalization->_map_lmk_prior;
    _marginalization_last->_lmk_with_prior           = _marginalization->_lmk_with_prior;
    _marginalization_last->_prior_lmk                = _marginalization->_prior_lmk;
    _marginalization_last->_info_lmk                 = _marginalization->_info_lmk;
    _marginalization_last->_Ak                       = _marginalization->_Ak;
    _marginalization_last->_bk                       = _marginalization->_bk;
    _marginalization_last->_marginalization_jacobian = _marginalization->_marginalization_jacobian;
    _marginalization_last->_marginalization_residual = _marginalization->_marginalization_residual;
    _marginalization_last->_m                        = _marginalization->_m;
    _marginalization_last->_n                        = _marginalization->_n;
    _marginalization_last->_n_full                   = _marginalization->_n_full;
    _marginalization_last->_U                        = _marginalization->_U;
    _marginalization_last->_Lambda                   = _marginalization->_Lambda;
    _marginalization_last->_Sigma                    = _marginalization->_Sigma;
    _marginalization_last->_T_f_w_lin                = _marginalization->_T_f_w_lin;
    _marginalization_last->_v_lin                    = _marginalization->_v_lin;
    _marginalization_last->_ba_lin                   = _marginalization->_ba_lin;
    _marginalization_last->_bg_lin                   = _marginalization->_bg_lin;
    _marginalization_last->_map_lmk_lin              = _marginalization->_map_lmk_lin;

    return true;
}

Eigen::MatrixXd BundleAdjustmentCERESAnalytic::marginalizeRelative(std::shared_ptr<Frame> &frame0,
                                                                   std::shared_ptr<Frame> &frame1) {

    // Setup the maps for memory gestion
    _map_lmk_ptpar.clear();
    _map_frame_posepar.clear();
    _map_frame_velpar.clear();
    _map_frame_dbapar.clear();
    _map_frame_dbgpar.clear();

    // Select the nodes to marginalize / keep
    _marginalization->preMarginalizeRelative(frame0, frame1);

    // Create pose parameters for the frames
    _map_frame_posepar.emplace(frame0, PoseParametersBlock(Eigen::Affine3d::Identity()));
    _map_frame_posepar.emplace(frame1, PoseParametersBlock(Eigen::Affine3d::Identity()));

    // Create Marginalization Blocks with pre-integration factors in the case of VIO
    if (frame0->getIMU() && frame1->getIMU()) {

        // Create velo and bias parameters for both frames
        _map_frame_velpar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbapar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbgpar.emplace(frame0, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_velpar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbapar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
        _map_frame_dbgpar.emplace(frame1, PointXYZParametersBlock(Eigen::Vector3d::Zero()));

        // Parameters of marginalization blocks (the variables are stored in the order v0, v1, ba0, bg0, ba1, bg0)
        std::vector<double *> parameter_blocks;
        std::vector<int> parameter_idx;

        // For the frames
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame0));
        parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame1));
        parameter_blocks.push_back(_map_frame_posepar.at(frame1).values());

        // For the velocities
        parameter_idx.push_back(_marginalization->_n);
        parameter_blocks.push_back(_map_frame_velpar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_n + 3);
        parameter_blocks.push_back(_map_frame_velpar.at(frame1).values());

        // For the biases
        parameter_idx.push_back(_marginalization->_n + 6);
        parameter_blocks.push_back(_map_frame_dbapar.at(frame0).values());
        parameter_idx.push_back(_marginalization->_n + 9);
        parameter_blocks.push_back(_map_frame_dbgpar.at(frame0).values());

        // Add the pre integration factor in the marginalization scheme, only if frame1's
        // preintegration starts at frame0 and is usable (the bias random walk always holds)
        if (imuFactorUsable(frame0, frame1)) {
            ceres::CostFunction *cost_fct = new IMUFactor(frame0->getIMU(), frame1->getIMU());
            _marginalization->_marginalization_blocks.push_back(
                std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
        }

        // Parameters of marginalization blocks
        std::vector<double *> parameter_blocks_b;
        std::vector<int> parameter_idx_b;

        // For the biases
        parameter_idx_b.push_back(_marginalization->_n + 6);
        parameter_blocks_b.push_back(_map_frame_dbapar.at(frame0).values());
        parameter_idx_b.push_back(_marginalization->_n + 9);
        parameter_blocks_b.push_back(_map_frame_dbgpar.at(frame0).values());
        parameter_idx_b.push_back(_marginalization->_n + 12);
        parameter_blocks_b.push_back(_map_frame_dbapar.at(frame1).values());
        parameter_idx_b.push_back(_marginalization->_n + 15);
        parameter_blocks_b.push_back(_map_frame_dbgpar.at(frame1).values());

        // Add the bias random walk factor in the marginalization scheme
        ceres::CostFunction *cost_fct_b = new IMUBiasFactor(frame0->getIMU(), frame1->getIMU());
        _marginalization->_marginalization_blocks.push_back(
            std::make_shared<MarginalizationBlockInfo>(cost_fct_b, parameter_idx_b, parameter_blocks_b));
    }

    // Create Marginalization Blocks with landmark to marginalize
    for (auto tlmk : _marginalization->_lmk_to_marg) {
        for (auto lmk : tlmk.second) {
            _map_lmk_ptpar.emplace(lmk, PointXYZParametersBlock(Eigen::Vector3d::Zero()));
            // For each feature on the frame
            for (auto &feature : lmk->getFeatures()) {
                std::shared_ptr<Frame> frame = feature.lock()->getSensor()->getFrame();
                if (frame == frame0 || frame == frame1) {

                    // Compute index and block vectors for reprojection factor
                    std::vector<double *> parameter_blocks;
                    std::vector<int> parameter_idx;

                    // For the frame
                    parameter_idx.push_back(_marginalization->_map_frame_idx.at(frame));
                    parameter_blocks.push_back(_map_frame_posepar.at(frame).values());

                    // For the lmk
                    parameter_idx.push_back(_marginalization->_map_lmk_idx.at(lmk));
                    parameter_blocks.push_back(_map_lmk_ptpar.at(lmk).values());

                    // Add the angular factor in the marginalization scheme
                    ceres::CostFunction *cost_fct = new ReprojectionErrCeres_pointxd_dx(
                        feature.lock()->getPoints().at(0), feature.lock()->getSensor(), lmk->getPose());
                    _marginalization->_marginalization_blocks.push_back(
                        std::make_shared<MarginalizationBlockInfo>(cost_fct, parameter_idx, parameter_blocks));
                }
            }
        }
    }

    // Reset the marginalization scheme if it failed
    if (!_marginalization->computeSchurComplement()) {
        _marginalization->_lmk_to_keep.clear();
        _marginalization->_marginalization_blocks.clear();
        _marginalization_last->_lmk_to_keep.clear();
        return Eigen::MatrixXd::Zero(12, 12);
    }

    // Compute the relative pose factor with NFR

    // Build a marginalization block to compute the jacobian
    Eigen::Affine3d T_w_a         = frame0->getFrame2WorldTransform();
    Eigen::Affine3d T_w_b         = frame1->getFrame2WorldTransform();
    Eigen::Affine3d T_a_b         = frame0->getWorld2FrameTransform() * T_w_b;
    ceres::CostFunction *cost_fct = new Relative6DPose(T_w_a, T_w_b, T_a_b, Vector6d::Ones().asDiagonal());
    std::vector<double *> parameter_blocks;
    std::vector<int> parameter_idx;
    parameter_blocks.push_back(_map_frame_posepar.at(frame0).values());
    parameter_idx.push_back(0);
    parameter_blocks.push_back(_map_frame_posepar.at(frame1).values());
    parameter_idx.push_back(0);
    MarginalizationBlockInfo block_relpose(cost_fct, parameter_idx, parameter_blocks);

    // Compute the covariance of the non linear factor
    block_relpose.Evaluate();
    Eigen::MatrixXd J   = Eigen::MatrixXd::Zero(6, 12);
    J.block(0, 0, 6, 6) = block_relpose._jacobians.at(0);
    J.block(0, 6, 6, 6) = block_relpose._jacobians.at(1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(_marginalization->_Ak);
    Eigen::MatrixXd Ak_inv =
        saes.eigenvectors() *
        Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse(), 0))
            .asDiagonal() *
        saes.eigenvectors().transpose();
    Eigen::MatrixXd cov = J * _marginalization->_Sigma_k * J.transpose();
    Eigen::MatrixXd inf = cov.inverse();

    return inf;
}

} // namespace isae
