#include <ceres/ceres.h>
#include <array>
#include <unordered_map>

#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/data/sensors/IMU.h"
#include "isaeslam/typedefs.h"
#include "utilities/geometry.h"

namespace isae {

/*!
 * @brief Marginalization block struct that stores a factor and the indices of the variables involved
 */
struct MarginalizationBlockInfo {

    MarginalizationBlockInfo(ceres::CostFunction *cost_function,
                             std::vector<int> parameter_idx,
                             std::vector<double *> parameter_blocks,
                             std::shared_ptr<ceres::LossFunction> loss = nullptr)
        : _cost_function(cost_function), _parameter_idx(parameter_idx), _parameter_blocks(parameter_blocks),
          _loss(loss) {}

    /*!
     * @brief Evaluate the factor; with a robust loss, the residual and Jacobians are corrected (Triggs et al.,
     * as Ceres and VINS-Mono do) so that the marginalized information is the robust one
     */
    void Evaluate();

    ceres::CostFunction *_cost_function;
    std::shared_ptr<ceres::LossFunction> _loss; //!< Robust loss of the factor (none if null)
    std::vector<int> _parameter_idx;
    std::vector<double *> _parameter_blocks;

    double **_raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _jacobians;
    Eigen::VectorXd _residuals;
};

/*!
 * @brief Marginalization class that handles marginalization (and sparsification) for fixed-lag smoothing.
 *
 * This class is responsible for selecting the variables to keep and to marginalize, computing the information matrix,
 * computing the Schur complement, and computing the Jacobians and residuals for the marginalization. It also provides
 * methods for sparsifying the dense prior in the case of VIO and VO.
 */
class Marginalization {
  public:
    /*!
     * @brief Select all the variables to keep and to marginalize in the Markov Blanket for fixed lag smoothing
     * @param frame0 frame to marginalize
     * @param frame1 frame linked to frame0
     * @param marginalization_last previous marginalization scheme
     */
    void preMarginalize(std::shared_ptr<Frame> &frame0,
                        std::shared_ptr<Frame> &frame1,
                        std::shared_ptr<Marginalization> &marginalization_last);

    /*!
     * @brief Select all the variables to keep and marginalize to derive the relative pose factor between two frames
     * @param frame0 The first frame
     * @param frame1 The second frame
     */
    void preMarginalizeRelative(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1);

    /*!
     * @brief Sparsify the dense prior factor in the VIO case
     */
    bool sparsifyVIO();

    /*!
     * @brief Jacobian of the PoseToLandmarkFactor residual T_f_w p - prior at the current state, w.r.t. the
     * pose increment (rotation, translation) and the landmark increment: [-R [p]x, R | R]
     */
    static Eigen::Matrix<double, 3, 9> poseToLandmarkJacobian(const Eigen::Affine3d &T_f_w, const Eigen::Vector3d &p);

    /*!
     * @brief Jacobian of the IMUPriordx residual at the prior, w.r.t. (rotation, translation, velocity, biases)
     * increments: pose block [R, 0; R [R^T t]x, R], identity for velocity and biases
     */
    static Eigen::Matrix<double, 15, 15> absolutePriorJacobian(const Eigen::Affine3d &T_f_w);

    /*!
     * @brief Sparsify the dense prior factor in the VO case
     */
    bool sparsifyVO();

    /*!
     * @brief Compute the information matrix and the gradient for a set of factors
     * @param blocks The vector of factors stored in Marginalization Blocks
     * @param A The information matrix
     * @param B The gradient
     */
    void computeInformationAndGradient(std::vector<std::shared_ptr<MarginalizationBlockInfo>> blocks,
                                       Eigen::MatrixXd &A,
                                       Eigen::VectorXd &b);

    /*!
     * @brief Compute the SVD of a given matrix to reveal its rank
     * @param A The input matrix
     * @param U The Eigen vectors of non null eigen values (up to a threshold)
     * @param d Non null Eigen values
     */
    void rankReveallingDecomposition(Eigen::MatrixXd A, Eigen::MatrixXd &U, Eigen::VectorXd &d);

    /*!
     * @brief Compute the dense prior with the Schur complement on _Ak
     */
    bool computeSchurComplement();

    /*!
     * @brief Compute the jacobian and the residual of the dense prior factor
     */
    bool computeJacobiansAndResiduals();

    /*!
     * @brief Compute the Entropy of a given landmark
     */
    double computeEntropy(std::shared_ptr<ALandmark> lmk);

    /*!
     * @brief Compute the Mutual Information between two landmarks
     */
    double computeMutualInformation(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j);

    /*!
     * @brief Approximate the Mutual Information between two landmarks using off diagonal blocks of _Ak
     */
    double computeOffDiag(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j);

    /*!
     * @brief Compute the KLD between the multivariate Gaussian with their Information Matrix assuming their mean is
     * equal
     */
    double computeKLD(Eigen::MatrixXd A_p, Eigen::MatrixXd A_q);

    int _m;                    //!< Parametric size of the variables to marginalize
    int _n;                    //!< Parametric size of the variables to keep
    int _n_full;               //!< Parametric size of the variables to keep after rank reveilling
    const double _eps = 1e-12; //!< Threshold to consider a null eigen value

    // Bookeeping of the variables to keep and to marginalize
    std::shared_ptr<Frame> _frame_to_marg;                            //!< Frame to marginalize
    std::shared_ptr<Frame> _frame_to_keep;                            //!< Frame to keep
    typed_vec_landmarks _lmk_to_keep;                                 //!< Set of landmarks to keep
    bool _has_prior = false; //!< A prior was computed and not discarded since (it may keep a frame and no landmark)
    typed_vec_landmarks _lmk_to_marg;                                 //!< Set of landmarks to marginalize
    std::unordered_map<std::shared_ptr<Frame>, int> _map_frame_idx;   //!< Map between frames and indices in _Ak
    std::unordered_map<std::shared_ptr<ALandmark>, int> _map_lmk_idx; //!< Map between landmarks and indices in _Ak
    std::unordered_map<std::shared_ptr<Frame>, Eigen::MatrixXd>
        _map_frame_inf; //!< Map between frame and their marginal information matrix
    std::vector<std::shared_ptr<MarginalizationBlockInfo>>
        _marginalization_blocks; //!< Vector of Marginalization blocks to derive _Ak

    // Sparsification info
    std::unordered_map<std::shared_ptr<ALandmark>, Eigen::Matrix3d>
        _map_lmk_inf; //!< Map between landmarks and info mat of sparse relative prior factors
    std::unordered_map<std::shared_ptr<ALandmark>, Eigen::Vector3d>
        _map_lmk_prior;                         //!< Map between landmarks and priors of sparse prior relative factors
    std::shared_ptr<ALandmark> _lmk_with_prior; //!< Landmark that has an absolute prior factor
    Eigen::Matrix3d _info_lmk;                  //!< Information matrix of the landmark absolute prior
    Eigen::Vector3d _prior_lmk;                 //!< Prior of the landmark absolute prior

    // Matrices and vectors of the dense prior
    Eigen::MatrixXd _Ak;                       //!< Information matrix of the subproblem
    Eigen::VectorXd _bk;                       //!< Gradient of the subproblem
    Eigen::MatrixXd _Sigma_k;                  //!< Covariance of the dense prior
    Eigen::MatrixXd _U;                        //!< Eigen vectors that have non null eigen values
    Eigen::VectorXd _Lambda;                   //!< Non null Eigen values
    Eigen::VectorXd _Sigma;                    //!< Inverse of _Lambda
    Eigen::MatrixXd _marginalization_jacobian; //!< Jacobian of the dense prior factor
    Eigen::VectorXd _marginalization_residual; //!< Residual of the dense prior factor

    // Linearization point of the kept variables. The prior is a linearization at these states: a later use
    // (next marginalization, next window) measures the kept variables from them, not from the current state
    Eigen::Affine3d _T_f_w_lin = Eigen::Affine3d::Identity(); //!< Pose of the frame to keep
    Eigen::Vector3d _v_lin     = Eigen::Vector3d::Zero();      //!< Velocity of the frame to keep
    Eigen::Vector3d _ba_lin    = Eigen::Vector3d::Zero();      //!< Accelerometer bias of the frame to keep
    Eigen::Vector3d _bg_lin    = Eigen::Vector3d::Zero();      //!< Gyroscope bias of the frame to keep
    std::unordered_map<std::shared_ptr<ALandmark>, Eigen::Affine3d> _map_lmk_lin; //!< Poses of the landmarks to keep

    /*!
     * @brief Record the current states of the kept variables as the linearization point of the prior
     */
    void storeLinearizationPoint();

    /*!
     * @brief The world frame moved by C (world_new <- world_old), with the kept states (a loop closure moving the
     * sliding window). The prior stays expressed in the world it was linearized in and records the change
     * (_W_prior): the dense factor maps the current states and their increments into that world before measuring
     * them from the linearization point (exactly, increments included), the sparse factors take their values in the
     * current world (linPoseCurrent() etc.). The next marginalization linearizes in the current world again.
     */
    void transformWorld(const Eigen::Affine3d &C) { _W_prior = _W_prior * C.inverse(); }

    //! World the prior is expressed in <- current world (identity until a loop closure moves the window)
    Eigen::Affine3d _W_prior = Eigen::Affine3d::Identity();

    //! Linearization pose of the kept frame (world to frame), in the current world
    Eigen::Affine3d linPoseCurrent() const { return _T_f_w_lin * _W_prior; }
    //! Linearization velocity of the kept frame, in the current world
    Eigen::Vector3d linVelocityCurrent() const { return _W_prior.rotation().transpose() * _v_lin; }

    /*!
     * @brief True if the landmark's position is well conditioned now: every observation's predicted bearing agrees
     * with the measured one (not behind a camera, not badly triangulated) and the observing rays span at least
     * min_ray_angle. Only such landmarks are linearized into the prior: a far or badly triangulated point slides
     * along its rays, where a linear prior breaks down (TUM-VI magistrale2: offsets of tens of metres, half of the
     * points behind the camera at linearization, prior cost exploding).
     */
    static bool wellConditioned(const std::shared_ptr<ALandmark> &lmk,
                                double max_bearing_err = 2.0 * M_PI / 180,
                                double min_ray_angle   = 1.0 * M_PI / 180);
};

/*!
 * @brief Ceres cost function of the dense prior factor
 */
class MarginalizationFactor : public ceres::CostFunction {
  public:
    MarginalizationFactor(std::shared_ptr<Marginalization> marginalization_info)
        : _marginalization_info(marginalization_info) {

        // Add frame block size
        if (_marginalization_info->_frame_to_keep) {
            this->mutable_parameter_block_sizes()->push_back(6);
            this->mutable_parameter_block_sizes()->push_back(3);
            this->mutable_parameter_block_sizes()->push_back(3);
            this->mutable_parameter_block_sizes()->push_back(3);
        }

        // Add landmark block size
        for (auto tlmk : _marginalization_info->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                (tlmk.first == "pointxd" ? this->mutable_parameter_block_sizes()->push_back(3)
                                         : this->mutable_parameter_block_sizes()->push_back(6));
            }
        }

        // Set the number of residuals
        this->set_num_residuals(_marginalization_info->_n_full);

        // The parameter blocks are increments on the states at the time the problem is built; the prior is
        // linearized at _marginalization_info's linearization point. Offsets between the two (right
        // perturbations, same conventions as the parameter blocks):
        // The current states are mapped into the world the prior is expressed in (W: identity unless a loop closure
        // moved the window since the prior was made)
        const Eigen::Affine3d &W = _marginalization_info->_W_prior;
        _R_W                     = W.rotation();
        _v_W                     = W.rotation().transpose() * W.translation();
        if (_marginalization_info->_frame_to_keep) {
            std::shared_ptr<Frame> f = _marginalization_info->_frame_to_keep;
            const Eigen::Affine3d T_lin = _marginalization_info->_T_f_w_lin;
            const Eigen::Affine3d T_now = f->getWorld2FrameTransform() * W.inverse();
            _A_rot = T_lin.rotation().transpose() * T_now.rotation();
            _dt0   = T_lin.rotation().transpose() * (T_now.translation() - T_lin.translation());
            if (f->getIMU()) {
                _dv0  = _R_W * f->getIMU()->getVelocity() - _marginalization_info->_v_lin;
                _dba0 = f->getIMU()->getBa() - _marginalization_info->_ba_lin;
                _dbg0 = f->getIMU()->getBg() - _marginalization_info->_bg_lin;
            }
        }
        for (auto tlmk : _marginalization_info->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                auto it = _marginalization_info->_map_lmk_lin.find(lmk);
                const Eigen::Affine3d T_lin = (it != _marginalization_info->_map_lmk_lin.end()) ? it->second
                                                                                                : lmk->getPose();
                const Eigen::Affine3d T_now = W * lmk->getPose();
                _lmk_offsets.emplace(lmk,
                                     std::make_pair(Eigen::Matrix3d(T_lin.rotation().transpose() * T_now.rotation()),
                                                    Eigen::Vector3d(T_lin.rotation().transpose() *
                                                                    (T_now.translation() - T_lin.translation()))));
            }
        }
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        int n = _marginalization_info->_n_full;
        Eigen::VectorXd dx(_marginalization_info->_n);
        dx.setZero();

        // Add frame dx
        int block_id = 0;
        // d(rotation dx) / d(rotation increment): d Log(A Exp(x)) / dx = Jr^-1(Log(A Exp(x))) Jr(x), the parameter
        // block being additive (Jr(x) = I only at x = 0)
        Eigen::Matrix3d J_rot = Eigen::Matrix3d::Identity(), J_trot = Eigen::Matrix3d::Zero();
        if (_marginalization_info->_frame_to_keep) {
            const int i0 = _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep);
            Eigen::Map<const Eigen::Matrix<double, 6, 1>> xp(parameters[block_id]);
            // Increments in the prior's world: y_r = R_W x_r, y_t = R_W x_t - R_W (Exp(x_r) - I) R_W^T t_W (T_f_w
            // right perturbations; the translation of a world-to-frame pose depends on the world origin)
            const Eigen::Vector3d xr   = xp.head<3>();
            const Eigen::Matrix3d Exr  = geometry::exp_so3(xr);
            const Eigen::Vector3d yr   = _R_W * xr;
            const Eigen::Vector3d yt   = _R_W * xp.tail<3>() - _R_W * (Exr * _v_W - _v_W);
            const Eigen::Vector3d drot = geometry::log_so3(_A_rot * geometry::exp_so3(yr));
            J_rot = geometry::so3_rightJacobian(drot).inverse() * geometry::so3_rightJacobian(yr) * _R_W;
            J_trot = _A_rot * _R_W * Exr * geometry::skewMatrix(_v_W) * geometry::so3_rightJacobian(xr);
            dx.segment<3>(i0)          = drot;
            dx.segment<3>(i0 + 3)      = _dt0 + _A_rot * yt;
            block_id++;
            dx.segment<3>(i0 + 6) = _dv0 + _R_W * Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
            block_id++;
            dx.segment<3>(i0 + 9) = _dba0 + Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
            block_id++;
            dx.segment<3>(i0 + 12) = _dbg0 + Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
            block_id++;
        }

        // Add landmarks dx
        for (auto tlmk : _marginalization_info->_lmk_to_keep) {
            for (auto lmk : tlmk.second) {
                if (_marginalization_info->_map_lmk_idx.at(lmk) == -1) {
                    block_id++; // the landmark still has its parameter block
                    continue;
                }

                const auto &off = _lmk_offsets.at(lmk);
                dx.segment<3>(_marginalization_info->_map_lmk_idx.at(lmk)) =
                    off.second + off.first * Eigen::Map<const Eigen::Vector3d>(parameters[block_id]);
                block_id++;
            }
        }

        // Compute the residual
        Eigen::Map<Eigen::VectorXd>(residuals, n) =
            _marginalization_info->_marginalization_residual + _marginalization_info->_marginalization_jacobian * dx;

        // Fill the jacobians
        if (jacobians) {

            block_id = 0;

            // Jacobians on frame
            if (_marginalization_info->_frame_to_keep) {
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 6);
                    jacobian.setZero();
                    const int i0 = _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep);
                    jacobian.leftCols(3) =
                        _marginalization_info->_marginalization_jacobian.middleCols(i0, 3) * J_rot +
                        _marginalization_info->_marginalization_jacobian.middleCols(i0 + 3, 3) * J_trot;
                    jacobian.middleCols(3, 3) =
                        _marginalization_info->_marginalization_jacobian.middleCols(i0 + 3, 3) * _A_rot * _R_W;
                }
                block_id++;
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 3);
                    jacobian.setZero();
                    jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                                               _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 6, 3) *
                                           _R_W;
                }
                block_id++;
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 3);
                    jacobian.setZero();
                    jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 9, 3);
                }
                block_id++;
                if (jacobians[block_id]) {

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                        jacobians[block_id], n, 3);
                    jacobian.setZero();
                    jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                        _marginalization_info->_map_frame_idx.at(_marginalization_info->_frame_to_keep) + 12, 3);
                }
                block_id++;
            }

            // Jacobians on landmarks
            for (auto tlmk : _marginalization_info->_lmk_to_keep) {
                for (auto lmk : tlmk.second) {
                    if (_marginalization_info->_map_lmk_idx.at(lmk) == -1) {
                        if (jacobians[block_id])
                            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
                                jacobians[block_id], n, 3)
                                .setZero();
                        block_id++;
                        continue;
                    }

                    if (jacobians[block_id]) {

                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(
                            jacobians[block_id], n, 3);
                        jacobian.setZero();
                        jacobian.leftCols(3) = _marginalization_info->_marginalization_jacobian.middleCols(
                                                   _marginalization_info->_map_lmk_idx.at(lmk), 3) *
                                               _lmk_offsets.at(lmk).first;
                    }
                    block_id++;
                }
            }
        }
        return true;
    }

    std::shared_ptr<Marginalization> _marginalization_info;

    /*!
     * @brief Diagnostics: distance of the current states from the linearization point of the prior
     * (largest landmark offset; frame rotation [rad], translation, velocity, accelerometer and gyroscope bias)
     */
    std::array<double, 6> linearizationOffsets() const {
        double lmk = 0;
        for (const auto &o : _lmk_offsets)
            lmk = std::max(lmk, o.second.second.norm());
        return {lmk, geometry::log_so3(_A_rot).norm(), _dt0.norm(), _dv0.norm(), _dba0.norm(), _dbg0.norm()};
    }

    // Offsets from the linearization point to the states the parameter blocks start from
    Eigen::Matrix3d _R_W   = Eigen::Matrix3d::Identity(); //!< Rotation of _W_prior (prior's world <- current world)
    Eigen::Vector3d _v_W   = Eigen::Vector3d::Zero();     //!< R_W^T t_W of _W_prior
    Eigen::Matrix3d _A_rot = Eigen::Matrix3d::Identity(); //!< R_lin^T R_now of the frame to keep
    Eigen::Vector3d _dt0   = Eigen::Vector3d::Zero();     //!< R_lin^T (t_now - t_lin) of the frame to keep
    Eigen::Vector3d _dv0 = Eigen::Vector3d::Zero(), _dba0 = Eigen::Vector3d::Zero(), _dbg0 = Eigen::Vector3d::Zero();
    std::unordered_map<std::shared_ptr<ALandmark>, std::pair<Eigen::Matrix3d, Eigen::Vector3d>>
        _lmk_offsets; //!< Per kept landmark: R_lin^T R_now and R_lin^T (p_now - p_lin)
};

} // namespace isae