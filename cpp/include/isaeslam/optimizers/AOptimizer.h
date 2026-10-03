#ifndef AOPTIMIZER_H
#define AOPTIMIZER_H

#include "isaeslam/data/frame.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/optimizers/marginalization.hpp"
#include "isaeslam/optimizers/parametersBlock.hpp"
#include "isaeslam/optimizers/residuals.hpp"
#include <ceres/ceres.h>

namespace isae {

/*!
 * @brief Diagnostics of the last sliding-window visual-inertial optimization (logging only)
 */
struct VIOptimStats {
    bool usable          = true; //!< ceres::Solver::Summary::IsSolutionUsable()
    std::string termination;     //!< Termination type of the solver
    double initial_cost  = 0;
    double final_cost    = 0;
    int iterations       = 0;
    uint n_imu_factors   = 0; //!< Preintegration factors actually inserted in the problem
    uint n_prior_factors = 0; //!< Marginalization prior factors (dense, or the sparse ones) inserted in the problem
    uint n_other_factors = 0; //!< All other residual blocks (visual factors, pose priors)

    // Dense marginalization prior at the start of the window (diagnostics; zero without a dense prior)
    double prior_cost0         = 0; //!< Cost of the prior at the window start
    double prior_lmk_offset    = 0; //!< Largest offset of a kept landmark from the prior's linearization point
    double prior_frame_rot_off = 0; //!< Rotation offset of the kept frame [rad]
    double prior_frame_t_off   = 0; //!< Translation offset of the kept frame
    double prior_v_off         = 0; //!< Velocity offset of the kept frame
    double prior_ba_off        = 0; //!< Accelerometer bias offset of the kept frame

    // Final cost per factor type (diagnostics)
    double cost_visual = 0; //!< Visual factors (and pose priors)
    double cost_imu    = 0; //!< Preintegration factors
    double cost_bias   = 0; //!< Bias random walk factors
    double cost_prior  = 0; //!< Marginalization prior (dense or sparse)
};

/*!
 * \brief Abstract class for optimization pipeline based on CERES.
 *
 * This class provides methods for optimizing frames, landmarks, and local maps.
 * It also implements methods involving factors such as initialization and marginalization.
 * This virtual formulation allows for different optimization scheme with different cost functions, different
 * parameterization or different CERES settings (e.g. numerical or analytic Jacobians).
 */
class AOptimizer {
  public:
    AOptimizer() {
        // Init marginalization variables
        _marginalization      = std::make_shared<Marginalization>();
        _marginalization_last = std::make_shared<Marginalization>();
    };

    void resetMarginalization() {
        _marginalization      = std::make_shared<Marginalization>();
        _marginalization_last = std::make_shared<Marginalization>();
    };

    /*!
     * @brief Diagnostics of the last visual-inertial window optimization
     */
    const VIOptimStats &getLastVIStats() const { return _last_vi_stats; }

    /*!
     * @brief True if a preintegration factor can link fi to fj: fj's preintegration starts at fi, spans at
     * most 1 s, has no missing IMU data and a usable covariance
     */
    static bool imuFactorUsable(const std::shared_ptr<Frame> &fi, const std::shared_ptr<Frame> &fj);

    //! Longest preintegration interval [s] used as a factor between two KFs
    static constexpr double kMaxImuFactorDt = 1.0;

    /*!
     * @brief Robust loss of the visual factors of the visual-inertial window and of their marginalization (the
     * Huber loss the front end uses). Bad associations would otherwise be marginalized into the prior at full
     * weight and could never be down-weighted afterwards.
     */
    static ceres::LossFunction *newVisualLoss() { return new ceres::HuberLoss(std::sqrt(1.345)); }

    /*!
     * @brief Use the robust visual loss in the VO window and its marginalization too (set when marginalization is
     * enabled: a marginalized bad association stays in the prior otherwise; default VO is left unchanged)
     */
    void setRobustVisualVO(bool robust) { _robust_visual_vo = robust; }

    /*!
     * @brief Structure only Bundle Adjustment for a frame.
     * @param frame The frame to optimize.
     *
     * This method optimize the landmarks associated to the frame, setting static the pose parameters.
     * It can be used to refine the triangulation of new landmarks.
     */
    bool landmarkOptimization(std::shared_ptr<Frame> &frame);

    /*!
     * @brief Motion only Bundle Adjustment for a frame.
     * @param moving_frame The frame to optimize.
     *
     * This method optimizes the frame's pose parameters only, setting static the landmarks.
     */
    bool singleFrameOptimization(std::shared_ptr<Frame> &moving_frame);

    /*!
     * @brief Visual-Inertial Motion only Bundle Adjustment for a frame.
     * @param moving_frame The frame to optimize.
     *
     * This method optimizes the frame's pose and velocity parameters.
     */
    bool singleFrameVIOptimization(std::shared_ptr<isae::Frame> &moving_frame);

    /*!
     * @brief Bundle Adjustment for a local map.
     * @param local_map The local map to optimize.
     * @param fixed_frame_number The frame number to fix during optimization (default is 0).
     */
    bool localMapBA(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_frame_number = 0);

    /*!
     * @brief Visual-Inertial Bundle Adjustment for a local map.
     * @param local_map The local map to optimize.
     * @param fixed_frame_number The frame number to fix during optimization (default is 0).
     */
    bool localMapVIOptimization(std::shared_ptr<isae::LocalMap> &local_map, const size_t fixed_frame_number = 0);

    /*!
     * @brief Visual-Inertial Bundle Adjustment for a local map with time delay.
     * @param local_map The local map to optimize.
     * @param fixed_frame_number The frame number to fix during optimization (default is 0).
     * @return The time delay used for optimization.
     */
    virtual bool localMapVIOptimizationTd(std::shared_ptr<isae::LocalMap> &local_map,
                                          double &td,
                                          const size_t fixed_frame_number = 0) {
        return true;
    }
    /*!
     * @brief Visual-Inertial Initialization for a local map.
     * @param local_map The local map to initialize.
     * @param R_w_i The initial rotation of the world frame with respect to the inertial frame.
     * @param optim_scale Whether to optimize the scale (default is false).
     * @return The scale factor used for initialization.
     */
    double VIInit(std::shared_ptr<isae::LocalMap> &local_map, Eigen::Matrix3d &R_w_i, bool optim_scale = false);

    /*!
     * @brief Structure only Bundle Adjustment for a frame with non overlapping fields of view sensors.
     * @param f The frame to optimize.
     * @param fp The previous frame to optimize.
     * @param T_cam0_cam0p The transformation from the current frame to the previous frame.
     * @param info_scale The scale factor for the information matrix.
     */
    virtual bool landmarkOptimizationNoFov(std::shared_ptr<Frame> &f,
                                           std::shared_ptr<Frame> &fp,
                                           Eigen::Affine3d &T_cam0_cam0p,
                                           double info_scale);

    /*!
     * @brief Marginalization of a frame and its landmarks.
     * @param frame0 The frame to marginalize.
     * @param frame1 The frame to keep connected to frame0.
     * @param enable_sparsif Whether to enable sparsification (default is false).
     *
     * Perform all the marginalization steps for a given frame. A marginalization object deals with all these steps
     */
    virtual bool marginalize(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1, bool enable_sparsif) {
        return true;
    };

    /*!
     * @brief Marginalization of all the landmarks linked to two frame and computation of the information matrix of the
     * relative pose.
     * @param frame0 The first frame to marginalize.
     * @param frame1 The second frame to marginalize.
     * @return The information matrix of the relative pose between frame0 and frame1.
     *
     * This method computes the information matrix of the relative pose between two frames and marginalizes the
     * landmarks linked to them. It uses NFR to extract a relative pose factor and to derive its information matrix.
     */
    virtual Eigen::MatrixXd marginalizeRelative(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1) {
        return Eigen::MatrixXd::Identity(6, 6);
    }

  protected:
    /*!
     * @brief Add visual residuals to a CERES problem for a local map.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param ordering The parameter block ordering for the problem.
     * @param frame_vector The vector of frames in the local map.
     * @param fixed_frame_number The number of frames to fix during optimization.
     * @param local_map The local map to which the residuals will be added.
     * @return The number of residuals added.
     */
    virtual uint addResidualsLocalMap(ceres::Problem &problem,
                                      ceres::LossFunction *loss_function,
                                      ceres::ParameterBlockOrdering *ordering,
                                      std::vector<std::shared_ptr<Frame>> &frame_vector,
                                      size_t fixed_frame_number,
                                      std::shared_ptr<isae::LocalMap> &local_map) = 0;

    /*!
     * @brief Add visual residuals to a CERES problem for structure only BA.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param cloud_to_optimize The landmarks to be optimized.
     * @return The number of residuals added.
     */
    virtual uint addLandmarkResiduals(ceres::Problem &problem,
                                      ceres::LossFunction *loss_function,
                                      typed_vec_landmarks &cloud_to_optimize) = 0;

    /*!
     * @brief Add visual residuals to a CERES problem for a motion only BA.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param frame The frame to which the residuals will be added.
     * @param cloud_to_optimize The landmarks belonging to the frame.
     * @return The number of residuals added.
     */
    virtual uint addSingleFrameResiduals(ceres::Problem &problem,
                                         ceres::LossFunction *loss_function,
                                         std::shared_ptr<Frame> &frame,
                                         typed_vec_landmarks &cloud_to_optimize) = 0;

    /*!
     * @brief Add marginalization residuals to a CERES problem.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param ordering The parameter block ordering for the problem.
     * @return The number of marginalization residuals added.
     */
    virtual uint addMarginalizationResiduals(ceres::Problem &problem,
                                             ceres::LossFunction *loss_function,
                                             ceres::ParameterBlockOrdering *ordering) {
        return 0;
    }

    /*!
     * @brief Add the sparse approximation of the marginalization prior (sparsification): an absolute factor on
     * the kept inertial state and pose-to-landmark factors (VIO), or a landmark chain (VO).
     */
    void addSparsePriorResiduals(ceres::Problem &problem,
                                 ceres::LossFunction *loss_function,
                                 ceres::ParameterBlockOrdering *ordering);

    /*!
     * @brief Add IMU residuals to a CERES problem.
     * @param problem The CERES problem to which the residuals will be added.
     * @param loss_function The loss function to be used for the residuals.
     * @param ordering The parameter block ordering for the problem.
     * @param frame_vector The vector of frames in the local map.
     * @param fixed_frame_number The number of frames to fix during optimization.
     * @return The number of preintegration factors added.
     */
    uint addIMUResiduals(ceres::Problem &problem,
                         ceres::LossFunction *loss_function,
                         ceres::ParameterBlockOrdering *ordering,
                         std::vector<std::shared_ptr<Frame>> &frame_vector,
                         size_t fixed_frame_number);

    std::unordered_map<std::shared_ptr<Frame>, PoseParametersBlock> _map_frame_posepar; //!< map for pose parameters
    std::unordered_map<std::shared_ptr<Frame>, PointXYZParametersBlock>
        _map_frame_velpar; //!< map for velocity parameters
    std::unordered_map<std::shared_ptr<Frame>, PointXYZParametersBlock>
        _map_frame_dbapar; //!< map for accelerometer bias parameters
    std::unordered_map<std::shared_ptr<Frame>, PointXYZParametersBlock>
        _map_frame_dbgpar; //!< map for gyroscope bias parameters
    std::unordered_map<std::shared_ptr<ALandmark>, PointXYZParametersBlock>
        _map_lmk_ptpar; //!< map for landmark point parameters
    std::unordered_map<std::shared_ptr<ALandmark>, PoseParametersBlock>
        _map_lmk_posepar; //!< map for landmark pose parameters

    /*!
     * @brief Number of window frames to hold fixed. A visual-inertial marginalization prior on a window frame
     * anchors the window (gauge) by itself; holding that frame's pose fixed as well leaves the prior's pull on it
     * unresolved, and that tension is folded into every following prior (the prior cost grew without bound and
     * the window failed). In that case no frame is fixed; otherwise the requested number.
     */
    size_t fixedFramesGivenPrior(const std::vector<std::shared_ptr<Frame>> &frame_vector, size_t requested) const;

    /*!
     * @brief Gauge of a window optimized without a fixed frame (VINS-Mono style): move the optimized states by the
     * rotation about gravity (world z) and the translation that give the anchor frame back its yaw and position
     * of before the optimization. These directions are unobservable in VIO; roll and pitch stay as optimized.
     */
    void restoreGauge(const std::shared_ptr<Frame> &anchor, const Eigen::Affine3d &T_w_anchor_before);

    /*!
     * @brief Re-integrate the preintegrations whose previous KF bias moved beyond the range of the first-order
     * bias correction (thresholds of VINS-Mono: 0.1 m/s^2, 0.01 rad/s)
     */
    void repropagateIfNeeded(std::vector<std::shared_ptr<Frame>> &frame_vector);

    /*!
     * @brief Store the diagnostics of a visual-inertial window optimization.
     */
    void recordVIStats(const ceres::Solver::Summary &summary,
                       uint n_imu_factors,
                       uint n_prior_factors,
                       uint n_residual_blocks) {
        _last_vi_stats.usable        = summary.IsSolutionUsable();
        _last_vi_stats.termination   = ceres::TerminationTypeToString(summary.termination_type);
        _last_vi_stats.initial_cost  = summary.initial_cost;
        _last_vi_stats.final_cost    = summary.final_cost;
        _last_vi_stats.iterations    = summary.iterations.size();
        _last_vi_stats.n_imu_factors   = n_imu_factors;
        _last_vi_stats.n_prior_factors = n_prior_factors;
        _last_vi_stats.n_other_factors = n_residual_blocks - n_imu_factors - n_prior_factors;
        _last_vi_stats.prior_cost0         = _prior_diag[0];
        _last_vi_stats.prior_lmk_offset    = _prior_diag[1];
        _last_vi_stats.prior_frame_rot_off = _prior_diag[2];
        _last_vi_stats.prior_frame_t_off   = _prior_diag[3];
        _last_vi_stats.prior_v_off         = _prior_diag[4];
        _last_vi_stats.prior_ba_off        = _prior_diag[5];
        _prior_diag.fill(0);
    }

    /*!
     * @brief Final cost of each factor type of a solved problem (diagnostics)
     */
    void recordCostsPerType(ceres::Problem &problem) {
        std::vector<ceres::ResidualBlockId> blocks;
        problem.GetResidualBlocks(&blocks);
        std::vector<ceres::ResidualBlockId> sets[4]; // visual, imu, bias, prior
        for (auto id : blocks) {
            const ceres::CostFunction *f = problem.GetCostFunctionForResidualBlock(id);
            if (dynamic_cast<const IMUFactor *>(f))
                sets[1].push_back(id);
            else if (dynamic_cast<const IMUBiasFactor *>(f))
                sets[2].push_back(id);
            else if (dynamic_cast<const MarginalizationFactor *>(f) || dynamic_cast<const IMUPriordx *>(f) ||
                     dynamic_cast<const PoseToLandmarkFactor *>(f) || dynamic_cast<const Landmark3DPrior *>(f))
                sets[3].push_back(id);
            else
                sets[0].push_back(id);
        }
        double *out[4] = {&_last_vi_stats.cost_visual, &_last_vi_stats.cost_imu, &_last_vi_stats.cost_bias,
                          &_last_vi_stats.cost_prior};
        for (int k = 0; k < 4; k++) {
            *out[k] = 0;
            if (sets[k].empty())
                continue;
            ceres::Problem::EvaluateOptions opt;
            opt.residual_blocks = sets[k];
            opt.apply_loss_function = true;
            problem.Evaluate(opt, out[k], nullptr, nullptr, nullptr);
        }
    }

    /*!
     * @brief Record the diagnostics of the dense prior inserted in the problem being built
     */
    void recordPriorDiag(MarginalizationFactor *factor, std::vector<double *> &blocks) {
        Eigen::VectorXd r(factor->num_residuals());
        factor->Evaluate(blocks.data(), r.data(), nullptr);
        std::array<double, 6> off = factor->linearizationOffsets();
        _prior_diag = {0.5 * r.squaredNorm(), off[0], off[1], off[2], off[3], off[4]};
    }
    std::array<double, 6> _prior_diag = {0, 0, 0, 0, 0, 0}; //!< Diagnostics of the dense prior of the problem

    VIOptimStats _last_vi_stats; //!< Diagnostics of the last visual-inertial window optimization

    bool _enable_sparsif = false;                           //!< enable sparsification of the marginalization
    bool _robust_visual_vo = false; //!< robust visual loss in the VO window and marginalization (see setRobustVisualVO)
    std::shared_ptr<Marginalization> _marginalization;      //!< marginalization object
    std::shared_ptr<Marginalization> _marginalization_last; //!< marginalization object of the last optimization
};

} // namespace isae

#endif // AOPTIMIZER_H
