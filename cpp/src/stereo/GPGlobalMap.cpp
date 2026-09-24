#include "isaeslam/stereo/GPGlobalMap.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>

// Port of SLAMesh (github.com/RuanJY/SLAMesh, src/map.cpp and src/cell.cpp) onto SaDVIO's dense
// stereo GP cells. Function names below refer to the SLAMesh originals. Behaviour that SLAMesh
// does not have is marked DEVIATION.

namespace isae {

using gp_detail::CellKey;
using gp_detail::Region;

namespace {

CellKey keyOf(const Eigen::Vector3d& p, double cell_size) {
    CellKey k;
    k.x = static_cast<int>(std::floor(p.x() / cell_size));
    k.y = static_cast<int>(std::floor(p.y() / cell_size));
    k.z = static_cast<int>(std::floor(p.z() / cell_size));
    return k;
}

CellKey shifted(CellKey k, int axis, int by) {
    if (axis == 0)
        k.x += by;
    else if (axis == 1)
        k.y += by;
    else
        k.z += by;
    return k;
}

// SLAMesh SurfNormAnalyticCostFunction: r = w * n . (R p + t - q). SLAMesh optimises a quaternion +
// translation with an analytic Jacobian; here the same residual is auto-differentiated over an
// angle-axis + translation increment, which needs no manifold (Ceres 2.2 dropped
// LocalParameterization).
struct PointToPlaneCost {
    PointToPlaneCost(const Eigen::Vector3d& p, const Eigen::Vector3d& q, const Eigen::Vector3d& n, double w)
        : p_(p), q_(q), n_(n), w_(w) {}

    template <typename T>
    bool operator()(const T* const pose, T* residual) const {
        const T p[3] = {T(p_.x()), T(p_.y()), T(p_.z())};
        T rp[3];
        ceres::AngleAxisRotatePoint(pose, p, rp);
        const T dx = rp[0] + pose[3] - T(q_.x());
        const T dy = rp[1] + pose[4] - T(q_.y());
        const T dz = rp[2] + pose[5] - T(q_.z());
        residual[0] = T(w_) * (T(n_.x()) * dx + T(n_.y()) * dy + T(n_.z()) * dz);
        return true;
    }

    static ceres::CostFunction* create(const Eigen::Vector3d& p, const Eigen::Vector3d& q,
                                       const Eigen::Vector3d& n, double w) {
        return new ceres::AutoDiffCostFunction<PointToPlaneCost, 1, 6>(new PointToPlaneCost(p, q, n, w));
    }

    Eigen::Vector3d p_, q_, n_;
    double w_;
};

double rotationAngleDeg(const Eigen::Matrix3d& R) {
    return Eigen::AngleAxisd(R).angle() * 180.0 / M_PI;
}

} // namespace

// ── cells ────────────────────────────────────────────────────────────────────

// SLAMesh Map::dividePointsIntoCell: bucket points on the world grid and GP-mesh every cell with
// enough points. Scan cells keep their raw points so updateMap can hand them to non-surface map
// cells.
GPGlobalMap::CellMap GPGlobalMap::buildCells(const std::vector<Eigen::Vector3d>& points_world) const {
    std::map<CellKey, std::vector<Eigen::Vector3d>> buckets;
    for (const auto& p : points_world)
        buckets[keyOf(p, _cfg.gp.cell_size)].push_back(p);

    CellMap cells;
    for (auto& kv : buckets) {
        if (static_cast<int>(kv.second.size()) < _cfg.gp.min_pts_per_cell)
            continue;
        Cell cell;
        cell.raw_points = std::move(kv.second);
        reconstructCell(kv.first, cell);
        cells.emplace(kv.first, std::move(cell));
    }
    return cells;
}

// SLAMesh Cell::reconstructSurfaces: PCA picks the prediction axes, then one GP layer per axis.
void GPGlobalMap::reconstructCell(const CellKey& key, Cell& cell) const {
    for (auto& layer : cell.layers) {
        layer.valid = false;
        layer.vertices.clear();
        layer.variances.clear();
    }

    std::array<bool, 3> directions;
    if (!gp_detail::chooseDirections(cell.raw_points, _cfg.gp, directions)) {
        cell.not_surface = true;
        return;
    }
    cell.not_surface = false;

    Region region;
    region.min = gp_detail::cellMin(key, _cfg.gp.cell_size);
    region.max = region.min + Eigen::Vector3d::Constant(_cfg.gp.cell_size);
    for (int dir = 0; dir < 3; ++dir) {
        if (!directions[dir])
            continue;
        Layer& layer = cell.layers[dir];
        layer.valid = gp_detail::gaussianProcessLayer(cell.raw_points, region, dir, _cfg.gp,
                                                      layer.vertices, layer.variances);
    }
}

// DEVIATION: SLAMesh keeps every raw point of a cell. Dense stereo gives thousands per cell per
// keyframe, so non-surface cells are thinned to a fixed budget by uniform striding.
void GPGlobalMap::capRawPoints(std::vector<Eigen::Vector3d>& points) const {
    const size_t cap = static_cast<size_t>(std::max(1, _cfg.max_raw_points_per_cell));
    if (points.size() <= cap)
        return;
    std::vector<Eigen::Vector3d> kept;
    kept.reserve(cap);
    const double stride = static_cast<double>(points.size()) / static_cast<double>(cap);
    for (size_t i = 0; i < cap; ++i)
        kept.push_back(points[static_cast<size_t>(i * stride)]);
    points.swap(kept);
}

// ── B. registration ──────────────────────────────────────────────────────────

// SLAMesh Map::findMatchPointToMesh (residual_combination = false). For each scan lattice vertex,
// the map vertex with the same lattice index is taken from the same cell or from cells shifted
// along the prediction axis by up to overlap_length; the closest along that axis wins. The normal
// comes from the matched map vertices' lattice neighbours.
GPGlobalMap::Matches GPGlobalMap::findMatchPointToMesh(const CellMap& scan, int overlap_length,
                                                       const Eigen::Affine3d& T_w_cam) const {
    const int n = std::max(1, _cfg.gp.num_test);
    const int n2 = n * n;
    const double thr = _cfg.variance_register;
    const double grid = _cfg.gp.cell_size;
    static const int off[2][4] = {{1, 0, -1, 0}, {0, -1, 0, 1}};
    const Eigen::Affine3d T_cam_w = T_w_cam.inverse();

    struct Best {
        bool found = false;
        Eigen::Vector3d p = Eigen::Vector3d::Zero();
        float variance = 0.f;
        double dist2 = 0.0;
    };

    Matches m;
    for (const auto& kv : scan) {
        for (int dir = 0; dir < 3; ++dir) {
            const Layer& sl = kv.second.layers[dir];
            if (!sl.valid)
                continue;

            std::vector<Best> best(n2);
            for (int k = -overlap_length; k <= overlap_length; ++k) {
                auto it = _cells.find(shifted(kv.first, dir, k));
                if (it == _cells.end())
                    continue;
                const Layer& gl = it->second.layers[dir];
                if (!gl.valid)
                    continue;

                for (int i = 0; i < n2; ++i) {
                    const double d = gl.vertices[i](dir) - sl.vertices[i](dir);
                    const double d2 = d * d;
                    const bool take_same_cell = (k == 0) && (!best[i].found || d2 < best[i].dist2);
                    const bool take_low_var = gl.variances[i] < thr &&
                                              (!best[i].found || d2 < best[i].dist2 || best[i].variance > thr);
                    if (take_same_cell || take_low_var) {
                        best[i].found = true;
                        best[i].p = gl.vertices[i];
                        best[i].variance = gl.variances[i];
                        best[i].dist2 = d2;
                    }
                }
            }

            for (int i = 0; i < n2; ++i) {
                if (!best[i].found)
                    continue;
                const int ix = i % n, iy = i / n;
                Eigen::Vector3d normal = Eigen::Vector3d::Zero();
                int valid_neighbours = 0;
                for (int o = 0; o < 4; ++o) {
                    const int x1 = ix + off[0][o], y1 = iy + off[1][o];
                    const int x2 = ix + off[0][(o + 1) % 4], y2 = iy + off[1][(o + 1) % 4];
                    if (x1 < 0 || x1 >= n || y1 < 0 || y1 >= n || x2 < 0 || x2 >= n || y2 < 0 || y2 >= n)
                        continue;
                    const int i1 = x1 + n * y1, i2 = x2 + n * y2;
                    if (!best[i1].found || !best[i2].found)
                        continue;
                    const Eigen::Vector3d a = best[i].p - best[i1].p;
                    const Eigen::Vector3d b = best[i].p - best[i2].p;
                    if (a.norm() > grid || b.norm() > grid)
                        continue;
                    const Eigen::Vector3d c = a.cross(b);
                    if (c.norm() > 1e-12) {
                        normal += c.normalized();
                        ++valid_neighbours;
                    }
                }
                if (valid_neighbours == 0)
                    normal = sl.vertices[i] - best[i].p;
                if (normal.norm() > 1e-12)
                    normal.normalize();

                if (best[i].dist2 > 0.0 && sl.variances[i] < thr && best[i].variance < thr &&
                    normal.norm() > 0.1) {
                    double w = 1.0;
                    if (_cfg.depth_weighting) {
                        // DEVIATION (opt-in): stereo depth error grows with z^2.
                        const double z = (T_cam_w * sl.vertices[i]).z();
                        if (z > _cfg.depth_weighting_ref)
                            w = (_cfg.depth_weighting_ref / z) * (_cfg.depth_weighting_ref / z);
                    }
                    m.scan.push_back(sl.vertices[i]);
                    m.map.push_back(best[i].p);
                    m.normal.push_back(normal);
                    m.weight.push_back(w);
                }
            }
        }
    }
    return m;
}

// SLAMesh Map::computeTPointToMesh: one Huber(0.1) point-to-plane solve, DENSE_QR, 10 iterations.
Eigen::Affine3d GPGlobalMap::computeTPointToMesh(const Matches& m) const {
    double pose[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(_cfg.huber_delta);
    for (size_t i = 0; i < m.scan.size(); ++i)
        problem.AddResidualBlock(PointToPlaneCost::create(m.scan[i], m.map[i], m.normal[i], m.weight[i]),
                                 loss, pose);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = _cfg.solver_iterations;
    options.num_threads = 1;
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    const Eigen::Vector3d aa(pose[0], pose[1], pose[2]);
    if (aa.norm() > 1e-15)
        T.linear() = Eigen::AngleAxisd(aa.norm(), aa.normalized()).toRotationMatrix();
    T.translation() = Eigen::Vector3d(pose[3], pose[4], pose[5]);
    return T;
}

// ── A. map update ────────────────────────────────────────────────────────────

// SLAMesh Cell::updateVertices: inverse-variance fusion of the prediction-axis coordinate. When
// either variance exceeds variance_map_update the lower-variance vertex is kept (ties keep the map).
void GPGlobalMap::fuseLayer(Layer& map_layer, const Layer& scan_layer, int axis) const {
    const double thr = _cfg.variance_map_update;
    for (size_t i = 0; i < map_layer.vertices.size() && i < scan_layer.vertices.size(); ++i) {
        const double u = map_layer.vertices[i](axis), o = scan_layer.vertices[i](axis);
        const double vu = map_layer.variances[i], vo = scan_layer.variances[i];
        if (vu <= thr && vo <= thr) {
            const double sum = vu + vo;
            if (sum > 0.0) {
                map_layer.vertices[i](axis) = (u * vo + o * vu) / sum;
                map_layer.variances[i] = static_cast<float>(vu * vo / sum);
            } else {
                map_layer.vertices[i](axis) = 0.5 * (u + o);
            }
        } else if (vo < vu) {
            map_layer.vertices[i](axis) = o;
            map_layer.variances[i] = static_cast<float>(vo);
        }
    }
}

// SLAMesh Map::updateMap. Existing surface cells fuse layer by layer (a layer the map lacks is not
// added, as in SLAMesh); non-surface map cells absorb the scan's raw points and are re-meshed;
// unseen cells are inserted.
void GPGlobalMap::updateMap(const CellMap& scan, GPIntegrationReport& report) {
    for (const auto& kv : scan) {
        const Cell& sc = kv.second;
        auto it = _cells.find(kv.first);
        if (it == _cells.end()) {
            Cell cell = sc;
            if (cell.not_surface)
                capRawPoints(cell.raw_points);
            else
                cell.raw_points.clear();
            _cells.emplace(kv.first, std::move(cell));
            ++report.cells_added;
            continue;
        }

        Cell& gc = it->second;
        if (gc.not_surface && !sc.not_surface) {
            gc.raw_points.insert(gc.raw_points.end(), sc.raw_points.begin(), sc.raw_points.end());
            capRawPoints(gc.raw_points);
            reconstructCell(kv.first, gc);
            if (!gc.not_surface)
                gc.raw_points.clear();
        } else {
            for (int dir = 0; dir < 3; ++dir) {
                if (gc.layers[dir].valid && sc.layers[dir].valid)
                    fuseLayer(gc.layers[dir], sc.layers[dir], dir);
            }
        }
        ++report.cells_fused;
    }
}

// ── integration ──────────────────────────────────────────────────────────────

GPIntegrationReport GPGlobalMap::integrate(const std::vector<Eigen::Vector3d>& points_cam,
                                           const Eigen::Affine3d& T_w_cam) {
    GPIntegrationReport report;
    report.T_w_cam_vo = T_w_cam;
    const Eigen::Affine3d T_guess = _cfg.carry_correction ? _carried * T_w_cam : T_w_cam;
    report.T_w_cam_guess = T_guess;

    std::vector<Eigen::Vector3d> points_world;
    points_world.reserve(points_cam.size());
    for (const auto& p : points_cam)
        points_world.push_back(T_guess * p);
    CellMap scan = buildCells(points_world);

    Eigen::Affine3d C = Eigen::Affine3d::Identity(); // world-frame correction: T_used = C * T_guess
    if (!_cfg.register_enabled) {
        report.status = "disabled";
    } else if (_cells.empty()) {
        report.status = "empty_map";
    } else {
        report.register_attempted = true;
        bool ok = true;
        // SLAMesh Map::registerToMap
        for (int it = 0; it < _cfg.register_times; ++it) {
            const int overlap = (it < _cfg.register_times - 1) ? _cfg.cross_cell_overlap_length : 0;
            const Matches m = findMatchPointToMesh(scan, overlap, C * T_guess);
            report.iterations = it + 1;
            report.matches = static_cast<int>(m.scan.size());
            if (report.matches < _cfg.min_matches) { // DEVIATION: SLAMesh solves regardless
                ok = false;
                report.status = "too_few_matches";
                break;
            }
            const Eigen::Affine3d D = computeTPointToMesh(m);
            C = D * C;
            for (auto& p : points_world)
                p = D * p;
            scan = buildCells(points_world); // SLAMesh re-meshes the scan after every step
            const double delta = 5.0 * (D.linear() - Eigen::Matrix3d::Identity()).norm() + D.translation().norm();
            if (delta < _cfg.converge_thr && it > 0)
                break;
        }
        if (ok && (C.translation().norm() > _cfg.max_correction_translation ||
                   rotationAngleDeg(C.linear()) > _cfg.max_correction_rotation_deg)) {
            ok = false; // DEVIATION: SLAMesh has no plausibility gate
            report.status = "correction_too_large";
        }
        if (ok) {
            report.register_accepted = true;
            report.status = "ok";
        } else {
            C = Eigen::Affine3d::Identity();
            points_world.clear();
            for (const auto& p : points_cam)
                points_world.push_back(T_guess * p);
            scan = buildCells(points_world);
        }
    }

    report.T_w_cam_used = C * T_guess;
    if (report.register_accepted && _cfg.carry_correction)
        _carried = C * _carried;

    report.scan_cells = static_cast<int>(scan.size());
    updateMap(scan, report);
    report.map_cells = static_cast<int>(_cells.size());
    return report;
}

// ── output ───────────────────────────────────────────────────────────────────

DenseMesh GPGlobalMap::mesh() const {
    DenseMesh mesh;
    std::vector<gp_detail::PatchInfo> patches;
    for (const auto& kv : _cells) {
        for (int dir = 0; dir < 3; ++dir) {
            const Layer& layer = kv.second.layers[dir];
            if (!layer.valid)
                continue;
            const int base = static_cast<int>(mesh.vertices.size());
            mesh.vertices.insert(mesh.vertices.end(), layer.vertices.begin(), layer.vertices.end());
            mesh.vertex_variance.insert(mesh.vertex_variance.end(), layer.variances.begin(), layer.variances.end());
            gp_detail::appendGridFaces(mesh, base, layer.variances, _cfg.gp);

            gp_detail::PatchInfo patch;
            patch.cell = kv.first;
            patch.prediction_axis = dir;
            patch.base_vertex = base;
            patch.variances = layer.variances;
            patches.push_back(std::move(patch));
        }
    }
    gp_detail::appendSeamFaces(mesh, patches, _cfg.gp);
    mesh.computeFaceNormals();
    return mesh;
}

bool GPGlobalMap::savePly(const std::string& path) const {
    return writeDenseMeshPly(mesh(), path, "SaDVIO dense GP global map (SLAMesh-style fusion)");
}

} // namespace isae
