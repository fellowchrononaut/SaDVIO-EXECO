#ifndef ISAE_TEST_JACOBIAN_CHECK_H
#define ISAE_TEST_JACOBIAN_CHECK_H

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <gtest/gtest.h>

#include <random>
#include <sstream>
#include <vector>

namespace isae_test {

/*!
 * @brief Compare the analytic Jacobians of a cost function with central finite differences.
 *
 * All SaDVIO parameter blocks are additive deltas, so the numerical derivative is taken directly on
 * the raw parameters (no manifold). A column fails when its max-abs error exceeds
 * tol * max(1, max-abs of the numerical column). The parameters are restored on return.
 */
inline ::testing::AssertionResult JacobiansMatch(const ceres::CostFunction &f,
                                                 const std::vector<double *> &params,
                                                 double tol = 1e-5,
                                                 double h   = 1e-6) {
    const std::vector<int32_t> &sizes = f.parameter_block_sizes();
    const int nr                      = f.num_residuals();

    using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    std::vector<RowMat> J(sizes.size());
    std::vector<double *> J_raw(sizes.size());
    for (size_t i = 0; i < sizes.size(); i++) {
        J[i].resize(nr, sizes[i]);
        J_raw[i] = J[i].data();
    }

    Eigen::VectorXd r(nr), r_plus(nr), r_minus(nr);
    if (!f.Evaluate(params.data(), r.data(), J_raw.data()))
        return ::testing::AssertionFailure() << "Evaluate() returned false";
    if (!r.allFinite())
        return ::testing::AssertionFailure() << "non-finite residual: " << r.transpose();

    std::ostringstream msg;
    bool ok = true;
    for (size_t i = 0; i < sizes.size(); i++) {
        if (!J[i].allFinite()) {
            ok = false;
            msg << "block " << i << ": non-finite analytic Jacobian\n";
            continue;
        }
        for (int k = 0; k < sizes[i]; k++) {
            const double x0 = params[i][k];
            params[i][k]    = x0 + h;
            f.Evaluate(params.data(), r_plus.data(), nullptr);
            params[i][k] = x0 - h;
            f.Evaluate(params.data(), r_minus.data(), nullptr);
            params[i][k] = x0;

            const Eigen::VectorXd num = (r_plus - r_minus) / (2 * h);
            const double err          = (J[i].col(k) - num).cwiseAbs().maxCoeff();
            const double scale        = std::max(1.0, num.cwiseAbs().maxCoeff());
            if (!(err <= tol * scale)) {
                ok = false;
                msg << "block " << i << " col " << k << ": max abs err " << err << " (column scale " << scale
                    << ")\n";
            }
        }
    }
    if (ok)
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << msg.str();
}

/*!
 * @brief Norm of the residual at the given parameters.
 */
inline double ResidualNorm(const ceres::CostFunction &f, const std::vector<double *> &params) {
    Eigen::VectorXd r(f.num_residuals());
    f.Evaluate(params.data(), r.data(), nullptr);
    return r.norm();
}

/*!
 * @brief Same check at a random point: every parameter is offset by U(-amplitude, amplitude).
 *
 * Bugs that only appear away from the zero increment (e.g. Exp(w) vs Exp(w)^T) are invisible to a
 * check at zero. The parameters are restored on return.
 */
inline ::testing::AssertionResult JacobiansMatchAtRandomPoint(const ceres::CostFunction &f,
                                                              const std::vector<double *> &params,
                                                              double amplitude,
                                                              unsigned seed = 42,
                                                              double tol    = 1e-5) {
    const std::vector<int32_t> &sizes = f.parameter_block_sizes();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-amplitude, amplitude);

    std::vector<std::vector<double>> saved(sizes.size());
    for (size_t i = 0; i < sizes.size(); i++) {
        saved[i].assign(params[i], params[i] + sizes[i]);
        for (int k = 0; k < sizes[i]; k++)
            params[i][k] += dist(gen);
    }
    ::testing::AssertionResult res = JacobiansMatch(f, params, tol);
    for (size_t i = 0; i < sizes.size(); i++)
        std::copy(saved[i].begin(), saved[i].end(), params[i]);
    return res;
}

} // namespace isae_test

#endif
