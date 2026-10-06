#include <algorithm>
#include <unordered_set>
#include "isaeslam/optimizers/marginalization.hpp"

#include <mutex>
#include <thread>

namespace isae {

// Threshold below which an eigenvalue of an information matrix is treated as zero. The eigen solver's error is about
// 1e-16 of the largest eigenvalue, so an absolute threshold alone kept rounding noise: the unobservable directions of
// a VIO prior (position, yaw) came out at ~1e-10 next to 1e7, and their inverse amplified that noise
static double nullEigenvalueThreshold(const Eigen::VectorXd &eigenvalues, double eps) {
    return eigenvalues.size() > 0 ? std::max(eps, 1e-12 * eigenvalues.maxCoeff()) : eps;
}

void MarginalizationBlockInfo::Evaluate() {

    _residuals.resize(_cost_function->num_residuals());

    std::vector<int> block_sizes = _cost_function->parameter_block_sizes();
    _raw_jacobians               = new double *[block_sizes.size()];
    _jacobians.resize(block_sizes.size());

    for (size_t i = 0; i < block_sizes.size(); i++) {
        _jacobians[i].resize(_cost_function->num_residuals(), block_sizes[i]);
        _raw_jacobians[i] = _jacobians[i].data();
    }
    _cost_function->Evaluate(_parameter_blocks.data(), _residuals.data(), _raw_jacobians);

    if (!_loss)
        return;
    const double sq_norm = _residuals.squaredNorm();
    double rho[3];
    _loss->Evaluate(sq_norm, rho);
    const double sqrt_rho1 = std::sqrt(rho[1]);
    double residual_scaling, alpha_sq_norm;
    if (sq_norm == 0.0 || rho[2] <= 0.0) {
        residual_scaling = sqrt_rho1;
        alpha_sq_norm    = 0.0;
    } else {
        const double D     = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
        const double alpha = 1.0 - std::sqrt(D);
        residual_scaling   = sqrt_rho1 / (1 - alpha);
        alpha_sq_norm      = alpha / sq_norm;
    }
    for (size_t i = 0; i < block_sizes.size(); i++)
        _jacobians[i] = sqrt_rho1 * (_jacobians[i] - alpha_sq_norm * _residuals * (_residuals.transpose() * _jacobians[i]));
    _residuals *= residual_scaling;
}

void Marginalization::preMarginalize(std::shared_ptr<Frame> &frame0,
                                     std::shared_ptr<Frame> &frame1,
                                     std::shared_ptr<Marginalization> &marginalization_last) {
    // Reset variables
    _frame_to_marg = frame0;
    _frame_to_keep = nullptr;
    _has_prior     = false;
    _lmk_to_keep.clear();
    _lmk_to_marg.clear();
    _map_frame_idx.clear();
    _map_lmk_idx.clear();
    _map_lmk_inf.clear();
    _map_lmk_prior.clear();
    _map_frame_inf.clear(); // keyed by the kept frame: never cleared, it held every kept KF (and its images) alive
    _n = 0;
    _m = 0;

    // We add the frame to marginalize
    _map_frame_idx.emplace(_frame_to_marg, 0);
    _m           = 6;
    int last_idx = 6;

    // We add velocity and bias in the case of VIO
    if (frame0->getIMU()) {
        _m += 9;
        last_idx += 9;
    }

    // Distinguish lmk to marginalize and lmk to keep in landmarks linked to the frame to marginalize
    std::unordered_set<std::shared_ptr<ALandmark>> seen;
    for (auto tlmks : _frame_to_marg->getLandmarks()) {
        // For all type of landmark
        for (auto lmk : tlmks.second) {

            if (lmk->isOutlier() || !lmk->isInMap() || !lmk->isInitialized() || !seen.insert(lmk).second)
                continue;

            bool is_lonely = true;
            int num_cam    = 0;
            for (auto f : lmk->getFeatures()) {

                // If the landmark is linked to other frames, it is kept
                if (f.lock()->getSensor()->getFrame() != frame0) {
                    is_lonely = false;

                    // We only include landmarks that have stereo factors (for matrix invertibility)
                } else {
                    num_cam += 1;
                }
            }

            // If the landmark has no prior and doesn't have full 3D information from the frame, it is ignored.
            // A monocular frame never has it: its single-view landmarks are kept when other frames see them
            // (the Schur complement and the prior handle the rank-deficient information)
            const int full_3d_views = (_frame_to_marg->getSensors().size() >= 2) ? 2 : 1;
            if (num_cam < full_3d_views && !lmk->hasPrior()) {
                lmk->setMarg();
                continue;
            }

            // A landmark to keep that is not well conditioned now is not linearized into the prior: its frame-0
            // observations are dropped, and if it carries prior information it is marginalized out of the prior
            // here (the Schur complement removes it consistently; it stays a free variable of the window).
            // Lonely landmarks are eliminated right away and never become prior variables.
            if (!is_lonely && !wellConditioned(lmk)) {
                if (lmk->hasPrior()) {
                    _lmk_to_marg[tlmks.first].push_back(lmk);
                    (tlmks.first == "pointxd" ? _m += 3 : _m += 6);
                }
                continue;
            }

            // The lonely landmarks are marginalized, the other will have a prior
            if (!is_lonely) {
                lmk->setPrior();
                _lmk_to_keep[tlmks.first].push_back(lmk);
                // Index increment is 3 for pointxd, 6 for other
                (tlmks.first == "pointxd" ? _n += 3 : _n += 6);
            } else {
                _lmk_to_marg[tlmks.first].push_back(lmk);
                lmk->setMarg();
                (tlmks.first == "pointxd" ? _m += 3 : _m += 6);
            }
        }
    }

    // Fill map lmk idx for lmk to marg
    for (auto tlmks : _lmk_to_marg) {
        for (auto lmk : tlmks.second) {
            _map_lmk_idx.emplace(lmk, last_idx);
            (tlmks.first == "pointxd" ? last_idx += 3 : last_idx += 6);
        }
    }

    // We add the frame to keep in the case of VIO
    if (frame1->getIMU()) {
        _frame_to_keep = frame1;
        _map_frame_idx.emplace(_frame_to_keep, last_idx);
        _n += 15;
        last_idx += 15;
    }

    // Fill map lmk idx for lmk to keep
    for (auto tlmks : _lmk_to_keep) {
        for (auto lmk : tlmks.second) {
            _map_lmk_idx.emplace(lmk, last_idx);
            (tlmks.first == "pointxd" ? last_idx += 3 : last_idx += 6);
        }
    }

    // Add Landmarks from the last prior that are not linked to the current KF
    // (This is only for resurected landmarks)
    if (marginalization_last->_lmk_to_keep.empty())
        return;

    bool discard_prior = false;

    for (auto tlmks : marginalization_last->_lmk_to_keep) {
        // For all type of landmarks
        for (auto lmk : tlmks.second) {
            if (_map_lmk_idx.find(lmk) == _map_lmk_idx.end()) {

                // Outlier case: we remove the prior
                if (lmk->isOutlier()) {
                    discard_prior = true;
                    break;
                }

                _lmk_to_keep[tlmks.first].push_back(lmk);
                _map_lmk_idx.emplace(lmk, last_idx);
                (tlmks.first == "pointxd" ? _n += 3 : _n += 6);
                (tlmks.first == "pointxd" ? last_idx += 3 : last_idx += 6);
            }
        }
    }

    if (discard_prior) {
        marginalization_last->_lmk_to_keep.clear();
        marginalization_last->_has_prior = false;
    }
}

void Marginalization::computeInformationAndGradient(std::vector<std::shared_ptr<MarginalizationBlockInfo>> blocks,
                                                    Eigen::MatrixXd &A,
                                                    Eigen::VectorXd &b) {

    auto updateInfoMat = [&A, &b](std::vector<std::shared_ptr<MarginalizationBlockInfo>> block_vector) {
        for (auto block : block_vector) {
            block->Evaluate();

            for (size_t i = 0; i < block->_parameter_blocks.size(); i++) {
                int size_i                 = block->_cost_function->parameter_block_sizes().at(i);
                int idx_i                  = block->_parameter_idx.at(i);
                Eigen::MatrixXd jacobian_i = block->_jacobians.at(i);

                // Outlier case, how to handle this?
                if (idx_i == -1)
                    continue;

                for (size_t j = i; j < block->_parameter_blocks.size(); j++) {
                    int size_j                 = block->_cost_function->parameter_block_sizes().at(j);
                    int idx_j                  = block->_parameter_idx.at(j);
                    Eigen::MatrixXd jacobian_j = block->_jacobians.at(j);

                    // Outlier case, how to handle this?
                    if (idx_j == -1)
                        continue;

                    {
                        if (i == j) {
                            A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                        } else {
                            A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                            A.block(idx_j, idx_i, size_j, size_i) =
                                A.block(idx_i, idx_j, size_i, size_j).transpose().eval();
                        }
                    }
                }
                { b.segment(idx_i, size_i) += jacobian_i.transpose() * block->_residuals; }
            }
        }
    };

    // Split the blocks in chunks
    int n_thread = 1;
    std::vector<std::vector<std::shared_ptr<MarginalizationBlockInfo>>> thread_chunks;
    for (int i = 0; i < n_thread; i++) {
        std::vector<std::shared_ptr<MarginalizationBlockInfo>> block_vector;
        thread_chunks.push_back(block_vector);
    }
    for (uint i = 0; i < blocks.size(); i++) {
        thread_chunks.at(i % n_thread).push_back(blocks.at(i));
    }

    // Launch on different thread the local detections
    std::vector<std::thread> threads;
    for (auto block_vector : thread_chunks) {
        threads.push_back(std::thread(updateInfoMat, block_vector));
    }
    for (auto &th : threads) {
        th.join();
    }
}

bool Marginalization::computeSchurComplement() {

    if (_n < 4)
        return false;

    // Instanciate A and b
    Eigen::MatrixXd A(_m + _n, _m + _n);
    Eigen::VectorXd b(_m + _n);
    A.setZero();
    b.setZero();

    computeInformationAndGradient(_marginalization_blocks, A, b);

    // Free marginalization blocks for ceres
    for (size_t i = 0; i < _marginalization_blocks.size(); i++) {
        delete _marginalization_blocks[i]->_cost_function;
        delete _marginalization_blocks[i]->_raw_jacobians;
    }
    _marginalization_blocks.clear();

    // Schur Complement computation
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, _m, _m) + A.block(0, 0, _m, _m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    const double tol_mm = nullEigenvalueThreshold(saes.eigenvalues(), _eps);
    Eigen::MatrixXd Amm_inv =
        saes.eigenvectors() *
        Eigen::VectorXd((saes.eigenvalues().array() > tol_mm).select(saes.eigenvalues().array().inverse(), 0))
            .asDiagonal() *
        saes.eigenvectors().transpose();

    _Ak = A.block(_m, _m, _n, _n) - A.block(_m, 0, _n, _m) * Amm_inv * A.block(_m, 0, _n, _m).transpose();
    _bk = b.segment(_m, _n) - A.block(_m, 0, _n, _m) * Amm_inv * b.segment(0, _m);

    // Update the map index to apply the reduction
    for (auto lmk_idx : _map_lmk_idx) {
        _map_lmk_idx.at(lmk_idx.first) -= _m;
    }
    if (_frame_to_keep) {
        _map_frame_idx.at(_frame_to_keep) -= _m;
    }

    // Perform the rank revealling decomposition
    rankReveallingDecomposition(_Ak, _U, _Lambda);
    _Sigma   = _Lambda.array().inverse();
    _n_full  = _U.cols();
    _Sigma_k = _U * _Sigma.asDiagonal() * _U.transpose();

    return true;
}

double Marginalization::computeEntropy(std::shared_ptr<ALandmark> lmk) {

    int size;
    lmk->_label == "pointxd" ? size = 3 : size = 6;

    Eigen::MatrixXd Sigma = _Sigma_k.block(_map_lmk_idx.at(lmk), _map_lmk_idx.at(lmk), size, size);
    return std::log(std::pow(2 * M_PI * M_E, size / 2) * Sigma.determinant());
}

double Marginalization::computeMutualInformation(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j) {

    // Retrieve the size of the landmarks
    int size_i;
    lmk_i->_label == "pointxd" ? size_i = 3 : size_i = 6;

    int size_j;
    lmk_j->_label == "pointxd" ? size_j = 3 : size_j = 6;

    // Extract submatrices for MI calculation
    Eigen::MatrixXd Sigma_ii = _Sigma_k.block(_map_lmk_idx.at(lmk_i), _map_lmk_idx.at(lmk_i), size_i, size_i);
    Eigen::MatrixXd Sigma_ij = _Sigma_k.block(_map_lmk_idx.at(lmk_i), _map_lmk_idx.at(lmk_j), size_i, size_j);
    Eigen::MatrixXd Sigma_ji = Sigma_ij.transpose();
    Eigen::MatrixXd Sigma_jj = _Sigma_k.block(_map_lmk_idx.at(lmk_j), _map_lmk_idx.at(lmk_j), size_j, size_j);

    // Build the sigma matrix for the denominator
    Eigen::MatrixXd Sigma(size_i + size_j, size_i + size_j);
    Sigma.topLeftCorner(size_i, size_i)     = Sigma_ii;
    Sigma.topRightCorner(size_i, size_j)    = Sigma_ij;
    Sigma.bottomLeftCorner(size_j, size_i)  = Sigma_ji;
    Sigma.bottomRightCorner(size_j, size_j) = Sigma_jj;

    // Compute MI
    double num   = Sigma_ii.determinant() * Sigma_jj.determinant();
    double denom = Sigma.determinant();
    return std::log(num / denom);
}

double Marginalization::computeOffDiag(std::shared_ptr<ALandmark> lmk_i, std::shared_ptr<ALandmark> lmk_j) {

    // Retrieve the size of the landmarks
    int size_i;
    lmk_i->_label == "pointxd" ? size_i = 3 : size_i = 6;

    int size_j;
    lmk_j->_label == "pointxd" ? size_j = 3 : size_j = 6;

    return std::abs(_Ak.block(_map_lmk_idx.at(lmk_i), _map_lmk_idx.at(lmk_j), size_i, size_j).trace());
}

void Marginalization::rankReveallingDecomposition(Eigen::MatrixXd A, Eigen::MatrixXd &U, Eigen::VectorXd &d) {

    int n = A.rows();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(A);
    const double tol       = nullEigenvalueThreshold(saes.eigenvalues(), _eps);
    Eigen::VectorXd d_full = Eigen::VectorXd((saes.eigenvalues().array() > tol).select(saes.eigenvalues().array(), 0));

    int j = 0;
    for (int i = 0; i < n; i++) {
        if (d_full(i) > tol)
            j++;
    }

    U = Eigen::MatrixXd::Zero(n, j);
    d = Eigen::VectorXd::Zero(j);

    int k = 0;
    for (int i = 0; i < n; i++) {

        if (d_full(i) > tol) {
            U.col(k) = saes.eigenvectors().col(i);
            d(k)     = d_full(i);
            k++;
        }
    }
}

double Marginalization::computeKLD(Eigen::MatrixXd A_p, Eigen::MatrixXd A_q) {

    Eigen::MatrixXd U;
    Eigen::VectorXd d;
    rankReveallingDecomposition(A_p, U, d);

    Eigen::VectorXd Sigma_p = d.array().inverse();
    Eigen::MatrixXd delta   = U.transpose() * A_q * U * Sigma_p.asDiagonal();
    double delta_det        = delta.determinant();

    if (delta_det == 0) {
        delta = delta + 0.001 * Eigen::MatrixXd::Identity(delta.rows(), delta.rows());
        return 100;
    }

    return 0.5 * (delta.trace() - std::log(delta_det) - U.cols());
}

Eigen::Matrix<double, 3, 9> Marginalization::poseToLandmarkJacobian(const Eigen::Affine3d &T_f_w,
                                                                     const Eigen::Vector3d &p) {
    Eigen::Matrix<double, 3, 9> J;
    J.block(0, 0, 3, 3) = -T_f_w.rotation() * geometry::skewMatrix(p);
    J.block(0, 3, 3, 3) = T_f_w.rotation();
    J.block(0, 6, 3, 3) = T_f_w.rotation();
    return J;
}

Eigen::Matrix<double, 15, 15> Marginalization::absolutePriorJacobian(const Eigen::Affine3d &T_f_w) {
    const Eigen::Matrix3d R          = T_f_w.rotation();
    Eigen::Matrix<double, 15, 15> J = Eigen::Matrix<double, 15, 15>::Identity();
    J.block(0, 0, 3, 3)             = R;
    J.block(3, 0, 3, 3)             = R * geometry::skewMatrix(R.transpose() * T_f_w.translation());
    J.block(3, 3, 3, 3)             = R;
    return J;
}

// Square root of the information of a factor from its covariance block, zero in the directions it does not observe
static Eigen::MatrixXd sqrtInformation(const Eigen::MatrixXd &cov) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(cov);
    const Eigen::VectorXd l = saes.eigenvalues();
    const double tol        = 1e-12 * std::max(l.maxCoeff(), 0.0);
    Eigen::VectorXd s       = Eigen::VectorXd::Zero(l.size());
    for (int i = 0; i < l.size(); i++)
        if (l(i) > tol)
            s(i) = 1 / std::sqrt(l(i));
    return saes.eigenvectors() * s.asDiagonal() * saes.eigenvectors().transpose();
}

// Sparse topology: an absolute factor on the kept frame and a pose-to-landmark factor per kept landmark. Their stacked
// Jacobian J (factor coordinates y = J x) is square and block triangular with invertible diagonal blocks. The closed
// form (Mazuran et al.) gives factor i the information (Sigma_y,ii)^-1, the inverse of its block of the dense prior's
// covariance in y. The prior is rank-deficient (position, yaw), so that covariance is a pseudo-inverse, and a
// pseudo-inverse does not carry over through a non-orthogonal J: it must be taken in y, Sigma_y = pinv(J^-T A J^-1),
// not as J pinv(A) J^T. The latter also spanned ~16 orders of magnitude (1e10 next to 1e-7) and lost the precise
// directions of the absolute factor (rotation, velocity, biases). This form is exact when the dense prior has the
// sparse structure, and is the KLD minimum otherwise.
// With A = U Lambda U^T (rank r) and W = J^-T U (n x r, full column rank): Sigma_y = H H^T, H = W (W^T W)^-1
// Lambda^-1/2. Lambda stays apart (relative accuracy in the well-observed directions); W^T W only carries the
// conditioning of J; and J^-T U is a block back-substitution, so the cost is that of W^T W, O(n r^2)
bool Marginalization::sparsifyVIO() {

    if (_n == 0)
        return false;

    // Blocks of the factors' Jacobians (those of the factors actually used, at the current state). Every kept
    // variable other than the frame belongs to exactly one factor block: J has the frame block J_f (15 x 15) and, per
    // landmark, a diagonal block D (sz x sz) and a block on the frame's first 6 columns
    struct FactorBlock {
        std::shared_ptr<ALandmark> lmk;
        int row, col, sz;
        Eigen::MatrixXd D, F; // diagonal block, block on the frame (sz x 6; empty for a line)
    };
    Eigen::Affine3d T_f_w                 = _frame_to_keep->getWorld2FrameTransform();
    const int i_frame                     = _map_frame_idx.at(_frame_to_keep);
    const Eigen::Matrix<double, 15, 15> Jf = absolutePriorJacobian(T_f_w);
    std::vector<FactorBlock> blocks;
    int row = 15;
    for (auto tlmk : _lmk_to_keep) {
        for (auto lmk : tlmk.second) {
            FactorBlock b;
            b.lmk = lmk;
            b.row = row;
            b.col = _map_lmk_idx.at(lmk);
            if (tlmk.first != "pointxd") {
                // No sparse factor for a line (addSparsePriorResiduals uses points only): identity rows keep J
                // invertible and leave the other factors' blocks of Sigma_y unchanged, i.e. the line is marginalized
                b.sz = 6;
                b.D  = Eigen::MatrixXd::Identity(6, 6);
            } else {
                const Eigen::Matrix<double, 3, 9> J_lmk = poseToLandmarkJacobian(T_f_w, lmk->getPose().translation());
                b.sz                                    = 3;
                b.D                                     = J_lmk.block(0, 6, 3, 3);
                b.F                                     = J_lmk.block(0, 0, 3, 6);
            }
            row += b.sz;
            blocks.push_back(b);
        }
    }
    if (row != _n)
        return false; // a kept variable the sparse topology does not cover

    // W = J^-T U by block back-substitution: the landmark rows first, then the frame rows
    const int r = static_cast<int>(_U.cols());
    Eigen::MatrixXd W(_n, r);
    Eigen::MatrixXd rhs_f = _U.middleRows(i_frame, 15);
    for (const auto &b : blocks) {
        W.middleRows(b.row, b.sz) = b.D.transpose().partialPivLu().solve(_U.middleRows(b.col, b.sz));
        if (b.F.size() > 0)
            rhs_f.topRows(6) -= b.F.transpose() * W.middleRows(b.row, b.sz);
    }
    W.topRows(15) = Jf.transpose().partialPivLu().solve(rhs_f);

    // H^T = Lambda^-1/2 (W^T W)^-1 W^T
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(r, r);
    G.selfadjointView<Eigen::Lower>().rankUpdate(W.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(G.selfadjointView<Eigen::Lower>());
    if (llt.info() != Eigen::Success)
        return false;
    Eigen::MatrixXd Ht = llt.solve(W.transpose());
    Ht                 = _Lambda.cwiseSqrt().cwiseInverse().asDiagonal() * Ht;
    auto sigmaBlock    = [&](int i0, int sz) -> Eigen::MatrixXd {
        return Ht.middleCols(i0, sz).transpose() * Ht.middleCols(i0, sz);
    };

    for (const auto &b : blocks) {
        if (b.sz != 3)
            continue;
        _map_lmk_inf[b.lmk]   = sqrtInformation(sigmaBlock(b.row, 3));
        _map_lmk_prior[b.lmk] = T_f_w * b.lmk->getPose().translation();
    }
    _map_frame_inf[_frame_to_keep] = sqrtInformation(sigmaBlock(0, 15));

    return true;
}

bool Marginalization::sparsifyVO() {

    if (_n == 0)
        return false;

    /// CHOW LIU TREE REORDERING ///

    // Compute a matrix with all MI values
    Eigen::MatrixXd mi_matrix = Eigen::MatrixXd::Zero(_lmk_to_keep["pointxd"].size(), _lmk_to_keep["pointxd"].size());
    for (uint k = 0; k < _lmk_to_keep["pointxd"].size(); k++) {
        for (uint l = 0; l < _lmk_to_keep["pointxd"].size(); l++) {

            // Skip the diagonal elements and if the matrix is already filled
            if (k == l || mi_matrix(k, l) != 0)
                continue;

            std::shared_ptr<ALandmark> lmk_k = _lmk_to_keep["pointxd"].at(k);
            std::shared_ptr<ALandmark> lmk_l = _lmk_to_keep["pointxd"].at(l);
            mi_matrix(k, l)                  = computeOffDiag(lmk_k, lmk_l);
            mi_matrix(l, k)                  = mi_matrix(k, l);
        }
    }

    // Hungarian algorithm to find the best combination
    std::vector<std::shared_ptr<ALandmark>> lmk_to_keep_ordered;
    lmk_to_keep_ordered.reserve(_lmk_to_keep["pointxd"].size());
    int max_row, max_col;

    // First couple
    mi_matrix.maxCoeff(&max_row, &max_col);
    lmk_to_keep_ordered.push_back(_lmk_to_keep["pointxd"].at(max_row));
    lmk_to_keep_ordered.push_back(_lmk_to_keep["pointxd"].at(max_col));

    // Set zero lines and cols of the first lmk and cols of the second
    mi_matrix.col(max_row).setZero();
    mi_matrix.row(max_row).setZero();
    mi_matrix.col(max_col).setZero();

    int current_idx = max_col;

    // Loop until the maximum is zero
    while (mi_matrix.row(current_idx).maxCoeff(&max_row, &max_col) != 0) {
        lmk_to_keep_ordered.push_back(_lmk_to_keep["pointxd"].at(max_col));

        // Set zero line of the first lmk and col of the second
        mi_matrix.row(current_idx).setZero();
        mi_matrix.col(max_col).setZero();

        current_idx = max_col;
    }

    _lmk_to_keep["pointxd"].clear();
    _lmk_to_keep["pointxd"] = lmk_to_keep_ordered;

    // Compute a vector of entropy
    Eigen::VectorXd entropys(_lmk_to_keep["pointxd"].size());
    for (uint k = 0; k < _lmk_to_keep["pointxd"].size(); k++) {
        entropys(k) = computeEntropy(_lmk_to_keep["pointxd"].at(k));
    }
    entropys.minCoeff(&max_row);
    _lmk_with_prior = _lmk_to_keep["pointxd"].at(max_row);

    _lmk_to_keep["pointxd"].clear();
    _lmk_to_keep["pointxd"] = lmk_to_keep_ordered;

    /// LANDMARK CHAIN ///

    // A unary factor for the first landmark and relative pose factors between lmk two by two
    Eigen::MatrixXd J                                  = Eigen::MatrixXd::Zero(3, _n);
    J.block(0, _map_lmk_idx.at(_lmk_with_prior), 3, 3) = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd J_tilde                            = J * _U;

    Eigen::Matrix3d sig = (J_tilde * _Sigma.asDiagonal() * J_tilde.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes0(sig);
    Eigen::Vector3d S =
        Eigen::Vector3d((saes0.eigenvalues().array() > _eps).select(saes0.eigenvalues().array().inverse(), 0));
    Eigen::Vector3d S_sqrt   = S.cwiseSqrt();
    Eigen::Matrix3d inf_sqrt = saes0.eigenvectors() * S_sqrt.asDiagonal() * saes0.eigenvectors().transpose();

    _info_lmk  = inf_sqrt;
    _prior_lmk = _lmk_with_prior->getPose().translation();

    // relative pose factors between lmk two by two
    for (uint k = 0; k < _lmk_to_keep["pointxd"].size() - 1; k++) {
        std::shared_ptr<ALandmark> lmk_k   = _lmk_to_keep["pointxd"].at(k);
        std::shared_ptr<ALandmark> lmk_kp1 = _lmk_to_keep["pointxd"].at(k + 1);

        Eigen::MatrixXd J                          = Eigen::MatrixXd::Zero(3, _n);
        J.block(0, _map_lmk_idx.at(lmk_k), 3, 3)   = Eigen::Matrix3d::Identity();
        J.block(0, _map_lmk_idx.at(lmk_kp1), 3, 3) = -Eigen::Matrix3d::Identity();
        Eigen::MatrixXd J_tilde                    = J * _U;

        Eigen::Matrix3d sig = J_tilde * _Sigma.asDiagonal() * J_tilde.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes1(sig);
        Eigen::Vector3d S =
            Eigen::Vector3d((saes1.eigenvalues().array() > _eps).select(saes1.eigenvalues().array().inverse(), 0));
        Eigen::Vector3d S_sqrt   = S.cwiseSqrt();
        Eigen::Matrix3d inf_sqrt = saes1.eigenvectors() * S_sqrt.asDiagonal() * saes1.eigenvectors().transpose();

        _map_lmk_inf.emplace(lmk_kp1, inf_sqrt);
        _map_lmk_prior.emplace(lmk_kp1, lmk_k->getPose().translation() - lmk_kp1->getPose().translation());
    }

    return true;
}

bool Marginalization::computeJacobiansAndResiduals() {

    Eigen::VectorXd Lambda_sqrt = _Lambda.cwiseSqrt();
    Eigen::VectorXd Sigma_sqrt  = _Sigma.cwiseSqrt();

    // The marginalized factors contribute 1/2 dx^T Ak dx + bk^T dx, with Ak = U Lambda U^T and bk = (Schur
    // complement of) J^T r, as assembled in computeInformationAndGradient (b += J^T r). The prior
    // 1/2 |r_p + J_p dx|^2 reproduces it when J_p^T J_p = Ak and J_p^T r_p = bk:
    // => J_p = Lambda^{1/2} U^T and r_p = Lambda^{-1/2} U^T bk
    // (the sign was negative: the prior then pushed the kept states away from what the marginalized measurements
    // say, by their full disagreement, whenever it was built away from the optimum)

    _marginalization_jacobian = Lambda_sqrt.asDiagonal() * _U.transpose();
    _marginalization_residual = Sigma_sqrt.asDiagonal() * _U.transpose() * _bk;

    // The prior has just been linearized at the current states
    storeLinearizationPoint();

    return true;
}

bool Marginalization::wellConditioned(const std::shared_ptr<ALandmark> &lmk,
                                      double max_bearing_err,
                                      double min_ray_angle) {
    const Eigen::Vector3d p = lmk->getPose().translation();
    if (!p.allFinite())
        return false;
    std::vector<Eigen::Vector3d> rays;
    for (auto &wf : lmk->getFeatures()) {
        std::shared_ptr<AFeature> f = wf.lock();
        if (!f || !f->getSensor())
            continue;
        std::shared_ptr<ImageSensor> cam = f->getSensor();
        const Eigen::Vector3d p_s        = cam->getWorld2SensorTransform() * p;
        const Eigen::Vector3d b          = f->getBearingVectors().at(0).normalized();
        if (std::acos(std::clamp(p_s.normalized().dot(b), -1.0, 1.0)) > max_bearing_err)
            return false;
        rays.push_back((p - cam->getSensor2WorldTransform().translation()).normalized());
    }
    double max_angle = 0;
    for (size_t i = 0; i < rays.size(); i++)
        for (size_t j = i + 1; j < rays.size(); j++)
            max_angle = std::max(max_angle, std::acos(std::clamp(rays[i].dot(rays[j]), -1.0, 1.0)));
    return max_angle >= min_ray_angle;
}

void Marginalization::storeLinearizationPoint() {
    _map_lmk_lin.clear();
    if (_frame_to_keep) {
        _T_f_w_lin = _frame_to_keep->getWorld2FrameTransform();
        if (_frame_to_keep->getIMU()) {
            _v_lin  = _frame_to_keep->getIMU()->getVelocity();
            _ba_lin = _frame_to_keep->getIMU()->getBa();
            _bg_lin = _frame_to_keep->getIMU()->getBg();
        }
    }
    for (auto &tlmk : _lmk_to_keep)
        for (auto &lmk : tlmk.second)
            _map_lmk_lin[lmk] = lmk->getPose();
}

void Marginalization::preMarginalizeRelative(std::shared_ptr<Frame> &frame0, std::shared_ptr<Frame> &frame1) {

    // Reset variables
    _frame_to_marg = nullptr;
    _frame_to_keep = nullptr;
    _has_prior     = false;
    _lmk_to_keep.clear();
    _lmk_to_marg.clear();
    _map_frame_idx.clear();
    _map_lmk_idx.clear();
    _map_lmk_inf.clear();
    _map_lmk_prior.clear();
    _map_frame_inf.clear(); // keyed by the kept frame: never cleared, it held every kept KF (and its images) alive
    int last_idx = 0;
    _m           = 0;
    _n           = 0;

    // Set to marginalize all the common landamrks between the frames
    for (auto tlmks : frame0->getLandmarks()) {

        // For all type of landmark
        for (auto lmk : tlmks.second) {

            if (lmk->isOutlier() || !lmk->isInMap() || !lmk->isInitialized() || lmk->isResurected())
                continue;
            int num_cam          = 0;
            bool is_linked_to_f1 = false;
            for (auto f : lmk->getFeatures()) {

                if (f.lock()->isOutlier())
                    continue;

                if (f.lock()->getSensor()->getFrame() == frame1)
                    is_linked_to_f1 = true;
                

                if (f.lock()->getSensor()->getFrame() == frame0)
                    num_cam++;
            }

            // If the landmark is linked to frame1 and stereo triangulated it is marginalized
            if (is_linked_to_f1 && (num_cam == 2)) {
                _lmk_to_marg[tlmks.first].push_back(lmk);
                (tlmks.first == "pointxd" ? _m += 3 : _m += 6);
                _map_lmk_idx.emplace(lmk, last_idx);
                (tlmks.first == "pointxd" ? last_idx += 3 : last_idx += 6);
            }
        }
    }

    // Set the indices of the two poses we keep
    _map_frame_idx.emplace(frame0, last_idx);
    last_idx += 6;
    _n += 6;
    if (frame0->getIMU() && frame1->getIMU()) {
        _n += 9;
        last_idx += 9;
    }

    _map_frame_idx.emplace(frame1, last_idx);
    last_idx += 6;
    _n += 6;
    if (frame1->getIMU() && frame1->getIMU()) {
        _n += 9;
        last_idx += 9;
    }
}

} // namespace isae