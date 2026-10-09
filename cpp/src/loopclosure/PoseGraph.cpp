#include "isaeslam/loopclosure/PoseGraph.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <algorithm>
#include <functional>
#include <map>

namespace isae {

namespace {

template <typename T> T normalizeAngle(const T &a) {
    return ceres::atan2(ceres::sin(a), ceres::cos(a));
}

// 4-DoF relative pose residual (VINS-Mono): node i has position t_i and yaw psi_i, with roll and pitch fixed from
// the odometry (R_i = Rz(psi_i) R_rp_i); the measurement is the relative translation in frame i and the yaw change
struct FourDofError {
    FourDofError(const Eigen::Vector3d &t_ij, double yaw_ij, const Eigen::Matrix3d &R_rp_i, double sigma_t,
                 double sigma_yaw)
        : _t_ij(t_ij), _yaw_ij(yaw_ij), _R_rp_i(R_rp_i), _wt(1 / sigma_t), _wy(1 / sigma_yaw) {}

    template <typename T>
    bool operator()(const T *yaw_i, const T *t_i, const T *yaw_j, const T *t_j, T *r) const {
        const T c = ceres::cos(yaw_i[0]), s = ceres::sin(yaw_i[0]);
        // R_i^T (t_j - t_i) = R_rp^T Rz(yaw)^T d
        const T d[3]  = {t_j[0] - t_i[0], t_j[1] - t_i[1], t_j[2] - t_i[2]};
        const T dz[3] = {c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]};
        for (int k = 0; k < 3; k++) {
            const T v = T(_R_rp_i(0, k)) * dz[0] + T(_R_rp_i(1, k)) * dz[1] + T(_R_rp_i(2, k)) * dz[2];
            r[k]      = (v - T(_t_ij(k))) * T(_wt);
        }
        r[3] = normalizeAngle(yaw_j[0] - yaw_i[0] - T(_yaw_ij)) * T(_wy);
        return true;
    }

    Eigen::Vector3d _t_ij;
    double _yaw_ij;
    Eigen::Matrix3d _R_rp_i;
    double _wt, _wy;
};

// 6-DoF relative pose residual: rotation error (angle-axis of R_ij_meas^T R_i^T R_j) and translation error in frame i
struct SixDofError {
    SixDofError(const Eigen::Affine3d &T_ij, double sigma_t, double sigma_rot)
        : _q_ij(T_ij.rotation()), _t_ij(T_ij.translation()), _wt(1 / sigma_t), _wr(1 / sigma_rot) {}

    template <typename T> bool operator()(const T *q_i, const T *t_i, const T *q_j, const T *t_j, T *r) const {
        Eigen::Map<const Eigen::Quaternion<T>> qi(q_i), qj(q_j);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ti(t_i), tj(t_j);
        const Eigen::Quaternion<T> q_err = _q_ij.cast<T>().conjugate() * qi.conjugate() * qj;
        const Eigen::Matrix<T, 3, 1> t_est = qi.conjugate() * (tj - ti);
        T aa[3];
        const T qe[4] = {q_err.w(), q_err.x(), q_err.y(), q_err.z()};
        ceres::QuaternionToAngleAxis(qe, aa);
        for (int k = 0; k < 3; k++) {
            r[k]     = (t_est(k) - T(_t_ij(k))) * T(_wt);
            r[3 + k] = aa[k] * T(_wr);
        }
        return true;
    }

    Eigen::Quaterniond _q_ij;
    Eigen::Vector3d _t_ij;
    double _wt, _wr;
};

Eigen::Matrix3d rotZ(double yaw) {
    return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

} // namespace

void PoseGraph::setEstimate(Node &n, const Eigen::Affine3d &T) const {
    const Eigen::Vector3d t = T.translation();
    std::copy(t.data(), t.data() + 3, n.t);
    if (_four_dof) {
        n.yaw = yawOf(T.rotation());
    } else {
        const Eigen::Quaterniond q(T.rotation());
        n.q[0] = q.x(), n.q[1] = q.y(), n.q[2] = q.z(), n.q[3] = q.w();
    }
}

int PoseGraph::addNode(const Eigen::Affine3d &T_w_f_odom, int segment) {
    Node n;
    n.T_odom  = T_w_f_odom;
    n.segment = segment;
    n.R_rp    = rotZ(-yawOf(T_w_f_odom.rotation())) * T_w_f_odom.rotation();
    setEstimate(n, correction(segment) * T_w_f_odom);
    _nodes.push_back(n);
    return static_cast<int>(_nodes.size()) - 1;
}

void PoseGraph::addLoop(int i, int j, const Eigen::Affine3d &T_fi_fj) {
    _loops.push_back({i, j, T_fi_fj, true});
}

size_t PoseGraph::nActiveLoops() const {
    return std::count_if(_loops.begin(), _loops.end(), [](const Loop &l) { return l.active; });
}

Eigen::Affine3d PoseGraph::pose(int i) const {
    const Node &n           = _nodes[i];
    Eigen::Affine3d T       = Eigen::Affine3d::Identity();
    T.translation()         = Eigen::Vector3d(n.t[0], n.t[1], n.t[2]);
    T.linear()              = _four_dof ? Eigen::Matrix3d(rotZ(n.yaw) * n.R_rp)
                                        : Eigen::Quaterniond(n.q[3], n.q[0], n.q[1], n.q[2]).toRotationMatrix();
    return T;
}

Eigen::Affine3d PoseGraph::correction(int segment) const {
    for (int i = static_cast<int>(_nodes.size()) - 1; i >= 0; i--)
        if (_nodes[i].segment == segment)
            return pose(i) * _nodes[i].T_odom.inverse();
    return Eigen::Affine3d::Identity();
}

void PoseGraph::reanchor(int segment, const Eigen::Affine3d &C) {
    for (auto &n : _nodes) {
        if (n.segment != segment)
            continue;
        n.T_odom = C * n.T_odom;
        n.R_rp   = rotZ(-yawOf(n.T_odom.rotation())) * n.T_odom.rotation();
    }
}

int PoseGraph::group(int segment) const {
    // Union-find over the segments joined by active loops, each group represented by its smallest segment
    std::map<int, int> parent;
    std::function<int(int)> find = [&](int x) {
        if (!parent.count(x))
            parent[x] = x;
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    };
    for (const auto &l : _loops) {
        if (!l.active)
            continue;
        const int a = find(_nodes[l.i].segment), b = find(_nodes[l.j].segment);
        if (a != b)
            parent[std::max(a, b)] = std::min(a, b);
    }
    return find(segment);
}

int PoseGraph::optimize(double reject_chi2) {
    if (_nodes.empty() || _loops.empty())
        return 0;

    // Loops are verified geometrically, so they enter without a robust loss: with drift of metres, correct loops
    // start with residuals far beyond any robust scale and would be ignored. A loop the solution disagrees with is
    // disabled instead, the worst one first, and the graph is solved again
    int disabled = 0;
    for (int pass = 0; pass < 5; pass++) {
        ceres::Problem problem;
        ceres::Manifold *quat_manifold = _four_dof ? nullptr : new ceres::EigenQuaternionManifold();
        auto addBlocks                 = [&](Node &n) {
            if (_four_dof) {
                problem.AddParameterBlock(&n.yaw, 1);
            } else {
                problem.AddParameterBlock(n.q, 4, quat_manifold);
            }
            problem.AddParameterBlock(n.t, 3);
        };
        auto addEdge = [&](int i, int j, const Eigen::Affine3d &T_ij, double st, double sr,
                           ceres::LossFunction *loss) -> ceres::ResidualBlockId {
            Node &a = _nodes[i], &b = _nodes[j];
            if (_four_dof) {
                const double yaw_ij = yawOf(a.T_odom.rotation() * T_ij.rotation()) - yawOf(a.T_odom.rotation());
                auto *c = new ceres::AutoDiffCostFunction<FourDofError, 4, 1, 3, 1, 3>(
                    new FourDofError(T_ij.translation(), yaw_ij, a.R_rp, st, sr));
                return problem.AddResidualBlock(c, loss, &a.yaw, a.t, &b.yaw, b.t);
            } else {
                auto *c = new ceres::AutoDiffCostFunction<SixDofError, 6, 4, 3, 4, 3>(new SixDofError(T_ij, st, sr));
                return problem.AddResidualBlock(c, loss, a.q, a.t, b.q, b.t);
            }
        };

        for (auto &n : _nodes)
            addBlocks(n);
        // Odometry edges: each node to its previous nodes of the same segment
        for (int j = 0; j < static_cast<int>(_nodes.size()); j++) {
            int linked = 0;
            for (int i = j - 1; i >= 0 && linked < _neighbors; i--) {
                if (_nodes[i].segment != _nodes[j].segment)
                    continue;
                addEdge(i, j, _nodes[i].T_odom.inverse() * _nodes[j].T_odom, _sigmas.odom_t, _sigmas.odom_rot,
                        nullptr);
                linked++;
            }
        }
        // Loop edges, robust
        std::vector<ceres::ResidualBlockId> loop_ids(_loops.size(), nullptr);
        for (size_t k = 0; k < _loops.size(); k++) {
            if (!_loops[k].active)
                continue;
            loop_ids[k] = addEdge(_loops[k].i, _loops[k].j, _loops[k].T_ij, _sigmas.loop_t, _sigmas.loop_rot,
                                  nullptr);
        }

        // Gauge: the earliest node of each group of segments connected by active loops
        std::map<int, int> parent;
        std::function<int(int)> find = [&](int s) {
            if (!parent.count(s))
                parent[s] = s;
            return parent[s] == s ? s : parent[s] = find(parent[s]);
        };
        for (auto &l : _loops)
            if (l.active)
                parent[find(_nodes[l.i].segment)] = find(_nodes[l.j].segment);
        std::map<int, int> first_of_group;
        for (int i = 0; i < static_cast<int>(_nodes.size()); i++) {
            const int g = find(_nodes[i].segment);
            if (!first_of_group.count(g))
                first_of_group[g] = i;
        }
        for (auto &fg : first_of_group) {
            Node &n = _nodes[fg.second];
            problem.SetParameterBlockConstant(_four_dof ? &n.yaw : n.q);
            problem.SetParameterBlockConstant(n.t);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        options.max_num_iterations                 = 50;
        options.num_threads                        = 2;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (!summary.IsSolutionUsable())
            return disabled;

        // The loop the solution disagrees with most (an outlier that passed verification), if any
        int worst = -1;
        double worst_r2 = reject_chi2;
        for (size_t k = 0; k < _loops.size(); k++) {
            if (!loop_ids[k])
                continue;
            double cost = 0; // |r|^2 / 2, whitened
            problem.EvaluateResidualBlock(loop_ids[k], false, &cost, nullptr, nullptr);
            if (2 * cost > worst_r2) {
                worst_r2 = 2 * cost;
                worst    = static_cast<int>(k);
            }
        }
        if (worst < 0)
            break;
        _loops[worst].active = false;
        disabled++;
    }
    return disabled;
}

} // namespace isae
