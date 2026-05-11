#include "isaeslam/stereo/GPMeshEstimator.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

namespace isae {
namespace {

struct CellKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator<(const CellKey& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct Region {
    Eigen::Vector3d min = Eigen::Vector3d::Zero();
    Eigen::Vector3d max = Eigen::Vector3d::Zero();
};

std::vector<double> evenLinSpaced(int n, double min_v, double max_v, bool full_cover) {
    std::vector<double> values(std::max(1, n), min_v);
    if (n <= 1)
        return values;

    if (full_cover) {
        const double interval = (max_v - min_v) / static_cast<double>(n - 1);
        for (int i = 0; i < n; ++i)
            values[i] = min_v + interval * static_cast<double>(i);
        values[n - 1] = max_v;
    } else {
        const double interval = (max_v - min_v) / static_cast<double>(n);
        for (int i = 0; i < n; ++i)
            values[i] = min_v + (static_cast<double>(i) + 0.5) * interval;
    }
    return values;
}

Eigen::Vector3d cellMin(const CellKey& key, double cell_size) {
    return Eigen::Vector3d(key.x * cell_size,
                           key.y * cell_size,
                           key.z * cell_size);
}

double coord(const Eigen::Vector3d& p, int axis) {
    return p(axis);
}

bool chooseDirections(const std::vector<Eigen::Vector3d>& points,
                      const GPMeshConfig& cfg,
                      std::array<bool, 3>& directions) {
    directions = {false, false, false};
    if (points.size() < 3)
        return false;

    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : points)
        mean += p;
    mean /= static_cast<double>(points.size());

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto& p : points) {
        const Eigen::Vector3d d = p - mean;
        cov += d * d.transpose();
    }
    cov /= static_cast<double>(points.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    if (eig.info() != Eigen::Success)
        return false;

    const Eigen::Vector3d evals = eig.eigenvalues();
    const Eigen::Matrix3d evecs = eig.eigenvectors();
    const double mid_eval = std::max(evals(1), 1e-12);

    // Same PCA direction logic as SLAMesh: the smallest eigenvector indicates
    // the surface normal when the cell is surface-like; ambiguous cases use
    // two or three prediction directions.
    if (evals(2) / mid_eval > cfg.eigen_1)
        return false;

    std::map<double, int> angle_min_axis;
    std::map<double, int> angle_max_axis;
    for (int axis = 0; axis < 3; ++axis) {
        double cos_min = std::max(-1.0, std::min(1.0, std::abs(evecs.col(0).dot(Eigen::Vector3d::Unit(axis)))));
        double cos_max = std::max(-1.0, std::min(1.0, std::abs(evecs.col(2).dot(Eigen::Vector3d::Unit(axis)))));
        angle_min_axis.emplace(std::acos(cos_min), axis);
        angle_max_axis.emplace(std::acos(cos_max), axis);
    }

    auto second_min = angle_min_axis.begin();
    ++second_min;
    auto second_max = angle_max_axis.begin();
    ++second_max;

    if (second_min->first - angle_min_axis.begin()->first > cfg.eigen_2) {
        directions[angle_min_axis.begin()->second] = true;
    } else if (angle_min_axis.rbegin()->first - second_min->first > cfg.eigen_3) {
        directions[angle_min_axis.begin()->second] = true;
        directions[second_min->second] = true;
    } else {
        directions = {true, true, true};
    }

    return directions[0] || directions[1] || directions[2];
}

void voxelFilter2D(const std::vector<Eigen::Vector3d>& points,
                   const Region& region,
                   int prediction_axis,
                   int loc_axis_x,
                   int loc_axis_y,
                   const GPMeshConfig& cfg,
                   std::vector<Eigen::Vector3d>& filtered) {
    const int n = std::max(1, cfg.num_test);
    const double interval_x = (region.max(loc_axis_x) - region.min(loc_axis_x)) / static_cast<double>(n);
    const double interval_y = (region.max(loc_axis_y) - region.min(loc_axis_y)) / static_cast<double>(n);

    struct Candidate {
        bool filled = false;
        double dist2 = std::numeric_limits<double>::max();
        Eigen::Vector3d point = Eigen::Vector3d::Zero();
    };

    std::vector<Candidate> bins(n * n);
    for (const auto& p : points) {
        int ix = static_cast<int>((coord(p, loc_axis_x) - region.min(loc_axis_x)) / interval_x);
        int iy = static_cast<int>((coord(p, loc_axis_y) - region.min(loc_axis_y)) / interval_y);
        ix = std::max(0, std::min(n - 1, ix));
        iy = std::max(0, std::min(n - 1, iy));

        const int idx = ix + n * iy;
        const double cx = region.min(loc_axis_x) + (static_cast<double>(ix) + 0.5) * interval_x;
        const double cy = region.min(loc_axis_y) + (static_cast<double>(iy) + 0.5) * interval_y;
        const double dx = coord(p, loc_axis_x) - cx;
        const double dy = coord(p, loc_axis_y) - cy;
        const double dist2 = dx * dx + dy * dy;

        if (!bins[idx].filled || dist2 < bins[idx].dist2) {
            bins[idx].filled = true;
            bins[idx].dist2 = dist2;
            bins[idx].point = p;
        }
    }

    filtered.clear();
    filtered.reserve(n * n);
    for (const auto& bin : bins) {
        if (bin.filled && std::isfinite(coord(bin.point, prediction_axis)))
            filtered.push_back(bin.point);
    }
}

bool gaussianProcessLayer(const std::vector<Eigen::Vector3d>& cell_points,
                          const Region& region,
                          int prediction_axis,
                          const GPMeshConfig& cfg,
                          std::vector<Eigen::Vector3d>& vertices,
                          std::vector<float>& variances) {
    const int n = std::max(1, cfg.num_test);
    const int n2 = n * n;
    const int loc_axis_x = (prediction_axis + 1) % 3;
    const int loc_axis_y = (prediction_axis + 2) % 3;

    std::vector<Eigen::Vector3d> train_points;
    voxelFilter2D(cell_points, region, prediction_axis, loc_axis_x, loc_axis_y, cfg, train_points);
    const int num_train = static_cast<int>(train_points.size());
    if (num_train == 0 || num_train > n2)
        return false;

    double train_x_min = coord(train_points[0], loc_axis_x);
    double train_x_max = train_x_min;
    double train_y_min = coord(train_points[0], loc_axis_y);
    double train_y_max = train_y_min;
    Eigen::VectorXd f(num_train);
    for (int i = 0; i < num_train; ++i) {
        f(i) = coord(train_points[i], prediction_axis);
        train_x_min = std::min(train_x_min, coord(train_points[i], loc_axis_x));
        train_x_max = std::max(train_x_max, coord(train_points[i], loc_axis_x));
        train_y_min = std::min(train_y_min, coord(train_points[i], loc_axis_y));
        train_y_max = std::max(train_y_max, coord(train_points[i], loc_axis_y));
    }

    const double mean = f.mean();
    f.array() -= mean;

    Eigen::MatrixXd K(num_train, num_train);
    for (int r = 0; r < num_train; ++r) {
        for (int c = 0; c < num_train; ++c) {
            const double dx = coord(train_points[c], loc_axis_x) - coord(train_points[r], loc_axis_x);
            const double dy = coord(train_points[c], loc_axis_y) - coord(train_points[r], loc_axis_y);
            K(r, c) = std::exp(-cfg.kernel_length * std::sqrt(dx * dx + dy * dy));
        }
    }
    K.diagonal().array() += cfg.variance_sensor * cfg.variance_sensor;

    Eigen::LLT<Eigen::MatrixXd> llt(K);
    if (llt.info() != Eigen::Success)
        return false;

    std::vector<double> test_x = evenLinSpaced(n, region.min(loc_axis_x), region.max(loc_axis_x), cfg.full_cover);
    std::vector<double> test_y = evenLinSpaced(n, region.min(loc_axis_y), region.max(loc_axis_y), cfg.full_cover);

    Eigen::MatrixXd K_star(n2, num_train);
    std::vector<Eigen::Vector3d> test_points(n2, Eigen::Vector3d::Zero());
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const int idx = row * n + col;
            test_points[idx](loc_axis_x) = test_x[row];
            test_points[idx](loc_axis_y) = test_y[col];
            for (int k = 0; k < num_train; ++k) {
                const double dx = coord(train_points[k], loc_axis_x) - coord(test_points[idx], loc_axis_x);
                const double dy = coord(train_points[k], loc_axis_y) - coord(test_points[idx], loc_axis_y);
                K_star(idx, k) = std::exp(-cfg.kernel_length * std::sqrt(dx * dx + dy * dy));
            }
        }
    }

    Eigen::MatrixXd K_inv_rhs = llt.solve(Eigen::MatrixXd::Identity(num_train, num_train));
    Eigen::MatrixXd KKY = K_star * K_inv_rhs;
    Eigen::VectorXd pred = KKY * f;
    pred.array() += mean;

    Eigen::VectorXd sigma2 = Eigen::VectorXd::Ones(n2) - (KKY * K_star.transpose()).diagonal();
    sigma2 = sigma2.cwiseMax(0.0);

    vertices.resize(n2);
    variances.resize(n2);
    for (int i = 0; i < n2; ++i) {
        Eigen::Vector3d p = test_points[i];
        p(prediction_axis) = std::max(region.min(prediction_axis),
                                      std::min(region.max(prediction_axis), pred(i)));
        vertices[i] = p;

        double v = sigma2(i);
        if (coord(p, loc_axis_x) > train_x_max || coord(p, loc_axis_x) < train_x_min ||
            coord(p, loc_axis_y) > train_y_max || coord(p, loc_axis_y) < train_y_min) {
            v *= 2.0;
        }
        variances[i] = static_cast<float>(v);
    }

    return true;
}

void appendGridFaces(DenseMesh& mesh, int base_vertex, const std::vector<float>& variances, const GPMeshConfig& cfg) {
    const int n = std::max(1, cfg.num_test);
    for (int vertex_i = 0; vertex_i < n * n; ++vertex_i) {
        const int ix = vertex_i % n;
        const int iy = vertex_i / n;
        if (ix + 1 < n && iy - 1 >= 0) {
            const double face_var = (variances[vertex_i] +
                                     variances[vertex_i + 1] +
                                     variances[vertex_i + 1 - n]) / 3.0;
            if (face_var < cfg.max_variance) {
                mesh.faces.emplace_back(base_vertex + vertex_i,
                                        base_vertex + vertex_i + 1,
                                        base_vertex + vertex_i + 1 - n);
            }
        }
        if (ix + 1 < n && iy + 1 < n) {
            const double face_var = (variances[vertex_i] +
                                     variances[vertex_i + 1] +
                                     variances[vertex_i + n]) / 3.0;
            if (face_var < cfg.max_variance) {
                mesh.faces.emplace_back(base_vertex + vertex_i,
                                        base_vertex + vertex_i + 1,
                                        base_vertex + vertex_i + n);
            }
        }
    }
}

} // namespace

DenseMesh GPMeshEstimator::estimate(const cv::Mat& disp_float,
                                     double f_rect,
                                     double baseline,
                                     double cx_rect,
                                     double cy_rect,
                                     const Eigen::Affine3d& T_w_rectcam) {
    DenseMesh mesh;
    if (disp_float.empty() || f_rect <= 0.0 || baseline <= 0.0 || _cfg.cell_size <= 0.0)
        return mesh;

    std::map<CellKey, std::vector<Eigen::Vector3d>> cell_map;
    for (int v = 0; v < disp_float.rows; ++v) {
        const float* row_ptr = disp_float.ptr<float>(v);
        for (int u = 0; u < disp_float.cols; ++u) {
            const float disp = row_ptr[u];
            if (disp <= 0.0f || !std::isfinite(disp))
                continue;

            const double Z = f_rect * baseline / static_cast<double>(disp);
            if (Z <= 0.0 || Z > _cfg.max_depth || !std::isfinite(Z))
                continue;

            const double X = (static_cast<double>(u) - cx_rect) * Z / f_rect;
            const double Y = (static_cast<double>(v) - cy_rect) * Z / f_rect;
            const Eigen::Vector3d p_world = T_w_rectcam * Eigen::Vector3d(X, Y, Z);

            CellKey key;
            key.x = static_cast<int>(std::floor(p_world.x() / _cfg.cell_size));
            key.y = static_cast<int>(std::floor(p_world.y() / _cfg.cell_size));
            key.z = static_cast<int>(std::floor(p_world.z() / _cfg.cell_size));
            cell_map[key].push_back(p_world);
        }
    }

    for (const auto& kv : cell_map) {
        const auto& points = kv.second;
        if (static_cast<int>(points.size()) < _cfg.min_pts_per_cell)
            continue;

        std::array<bool, 3> directions;
        if (!chooseDirections(points, _cfg, directions))
            continue;

        const Eigen::Vector3d min_corner = cellMin(kv.first, _cfg.cell_size);
        Region region;
        region.min = min_corner;
        region.max = min_corner + Eigen::Vector3d::Constant(_cfg.cell_size);

        for (int dir = 0; dir < 3; ++dir) {
            if (!directions[dir])
                continue;

            std::vector<Eigen::Vector3d> layer_vertices;
            std::vector<float> layer_variances;
            if (!gaussianProcessLayer(points, region, dir, _cfg, layer_vertices, layer_variances))
                continue;

            const int base_vertex = static_cast<int>(mesh.vertices.size());
            mesh.vertices.insert(mesh.vertices.end(), layer_vertices.begin(), layer_vertices.end());
            mesh.vertex_variance.insert(mesh.vertex_variance.end(), layer_variances.begin(), layer_variances.end());
            appendGridFaces(mesh, base_vertex, layer_variances, _cfg);
        }
    }

    mesh.computeFaceNormals();
    return mesh;
}

} // namespace isae
