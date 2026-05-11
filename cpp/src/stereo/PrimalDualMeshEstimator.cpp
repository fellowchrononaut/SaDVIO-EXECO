#include "isaeslam/stereo/PrimalDualMeshEstimator.h"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <set>
#include <vector>

namespace isae {
namespace {

struct Vertex2D {
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    bool has_tracking_depth = false;
    double tracking_inv_depth = 0.0;
};

struct EdgeOp {
    int i = 0;
    int j = 0;
    double alpha = 1.0;
    double beta = 1.0;
    Eigen::Vector2d du = Eigen::Vector2d::Zero(); // u_i - u_j
};

struct DepthRow {
    int i0 = 0;
    int i1 = 0;
    int i2 = 0;
    double a0 = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double inv_depth = 0.0;
};

double clampScalar(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

double shrinkToReference(double value, double reference, double amount) {
    if (value - reference > amount)
        return value - amount;
    if (value - reference < -amount)
        return value + amount;
    return reference;
}

bool barycentric(const Eigen::Vector2d& p,
                 const Eigen::Vector2d& a,
                 const Eigen::Vector2d& b,
                 const Eigen::Vector2d& c,
                 double& wa,
                 double& wb,
                 double& wc) {
    const double den = (b.y() - c.y()) * (a.x() - c.x()) +
                       (c.x() - b.x()) * (a.y() - c.y());
    if (std::abs(den) < 1e-12)
        return false;

    wa = ((b.y() - c.y()) * (p.x() - c.x()) +
          (c.x() - b.x()) * (p.y() - c.y())) / den;
    wb = ((c.y() - a.y()) * (p.x() - c.x()) +
          (a.x() - c.x()) * (p.y() - c.y())) / den;
    wc = 1.0 - wa - wb;
    return true;
}

void addOrUpdateVertex(std::vector<Vertex2D>& vertices,
                       std::map<std::pair<int, int>, int>& rounded_lookup,
                       double u,
                       double v,
                       bool has_tracking_depth,
                       double tracking_inv_depth) {
    const int ru = static_cast<int>(std::round(u));
    const int rv = static_cast<int>(std::round(v));
    const auto key = std::make_pair(ru, rv);
    auto it = rounded_lookup.find(key);
    if (it != rounded_lookup.end()) {
        Vertex2D& existing = vertices[it->second];
        if (has_tracking_depth) {
            if (existing.has_tracking_depth)
                existing.tracking_inv_depth = 0.5 * (existing.tracking_inv_depth + tracking_inv_depth);
            else
                existing.tracking_inv_depth = tracking_inv_depth;
            existing.has_tracking_depth = true;
        }
        return;
    }

    Vertex2D vertex;
    vertex.uv = Eigen::Vector2d(u, v);
    vertex.has_tracking_depth = has_tracking_depth;
    vertex.tracking_inv_depth = tracking_inv_depth;
    rounded_lookup.emplace(key, static_cast<int>(vertices.size()));
    vertices.push_back(vertex);
}

int findVertexIndex(float x,
                    float y,
                    const std::vector<Vertex2D>& vertices,
                    const std::map<std::pair<int, int>, int>& rounded_lookup) {
    const int rx = static_cast<int>(std::round(x));
    const int ry = static_cast<int>(std::round(y));
    auto it = rounded_lookup.find({rx, ry});
    if (it != rounded_lookup.end())
        return it->second;

    int best = -1;
    double best_dist2 = std::numeric_limits<double>::max();
    for (int i = 0; i < static_cast<int>(vertices.size()); ++i) {
        const double dx = vertices[i].uv.x() - static_cast<double>(x);
        const double dy = vertices[i].uv.y() - static_cast<double>(y);
        const double dist2 = dx * dx + dy * dy;
        if (dist2 < best_dist2) {
            best_dist2 = dist2;
            best = i;
        }
    }
    return best_dist2 <= 4.0 ? best : -1;
}

std::vector<Eigen::Vector3i> buildDelaunayFaces(const std::vector<Vertex2D>& vertices,
                                                const std::map<std::pair<int, int>, int>& rounded_lookup,
                                                int img_w,
                                                int img_h) {
    std::vector<Eigen::Vector3i> faces;
    if (vertices.size() < 3)
        return faces;

    cv::Subdiv2D subdiv(cv::Rect(0, 0, img_w, img_h));
    for (const auto& vertex : vertices) {
        const float u = static_cast<float>(clampScalar(vertex.uv.x(), 0.0, static_cast<double>(img_w - 1)));
        const float v = static_cast<float>(clampScalar(vertex.uv.y(), 0.0, static_cast<double>(img_h - 1)));
        subdiv.insert(cv::Point2f(u, v));
    }

    std::vector<cv::Vec6f> triangles;
    subdiv.getTriangleList(triangles);

    std::set<std::array<int, 3>> unique_faces;
    for (const auto& tri : triangles) {
        const Eigen::Vector2d p0(tri[0], tri[1]);
        const Eigen::Vector2d p1(tri[2], tri[3]);
        const Eigen::Vector2d p2(tri[4], tri[5]);
        auto in_image = [&](const Eigen::Vector2d& p) {
            return p.x() >= -1e-4 && p.x() < img_w + 1e-4 &&
                   p.y() >= -1e-4 && p.y() < img_h + 1e-4;
        };
        if (!in_image(p0) || !in_image(p1) || !in_image(p2))
            continue;

        const int i0 = findVertexIndex(tri[0], tri[1], vertices, rounded_lookup);
        const int i1 = findVertexIndex(tri[2], tri[3], vertices, rounded_lookup);
        const int i2 = findVertexIndex(tri[4], tri[5], vertices, rounded_lookup);
        if (i0 < 0 || i1 < 0 || i2 < 0 || i0 == i1 || i1 == i2 || i0 == i2)
            continue;

        std::array<int, 3> key = {i0, i1, i2};
        std::array<int, 3> sorted_key = key;
        std::sort(sorted_key.begin(), sorted_key.end());
        if (unique_faces.insert(sorted_key).second)
            faces.emplace_back(i0, i1, i2);
    }

    return faces;
}

std::vector<EdgeOp> buildEdgeOperators(const std::vector<Eigen::Vector3i>& faces,
                                       const std::vector<Vertex2D>& vertices) {
    std::vector<EdgeOp> edges;
    std::set<std::pair<int, int>> seen_edges;

    for (const auto& tri : faces) {
        for (int e = 0; e < 3; ++e) {
            const int a = tri[e];
            const int b = tri[(e + 1) % 3];
            const auto key = std::make_pair(std::min(a, b), std::max(a, b));
            if (!seen_edges.insert(key).second)
                continue;

            EdgeOp op;
            op.i = a;
            op.j = b;
            op.du = vertices[a].uv - vertices[b].uv;
            const double edge_len = std::max(1e-6, op.du.norm());
            op.alpha = 1.0 / edge_len;
            op.beta = 1.0;
            edges.push_back(op);
        }
    }

    return edges;
}

std::vector<DepthRow> rasterizeDenseDepthRows(const cv::Mat& disp_float,
                                              const std::vector<Vertex2D>& vertices,
                                              const std::vector<Eigen::Vector3i>& faces,
                                              double f_rect,
                                              double baseline,
                                              double min_depth,
                                              double max_depth) {
    std::vector<DepthRow> rows;
    if (disp_float.empty())
        return rows;

    cv::Mat assigned = cv::Mat::zeros(disp_float.size(), CV_8U);
    rows.reserve(static_cast<size_t>(disp_float.rows) * static_cast<size_t>(disp_float.cols) / 2);

    for (const auto& face : faces) {
        const Eigen::Vector2d& p0 = vertices[face[0]].uv;
        const Eigen::Vector2d& p1 = vertices[face[1]].uv;
        const Eigen::Vector2d& p2 = vertices[face[2]].uv;

        const int min_u = std::max(0, static_cast<int>(std::floor(std::min({p0.x(), p1.x(), p2.x()}))));
        const int max_u = std::min(disp_float.cols - 1, static_cast<int>(std::ceil(std::max({p0.x(), p1.x(), p2.x()}))));
        const int min_v = std::max(0, static_cast<int>(std::floor(std::min({p0.y(), p1.y(), p2.y()}))));
        const int max_v = std::min(disp_float.rows - 1, static_cast<int>(std::ceil(std::max({p0.y(), p1.y(), p2.y()}))));

        for (int v = min_v; v <= max_v; ++v) {
            const float* disp_row = disp_float.ptr<float>(v);
            uint8_t* assigned_row = assigned.ptr<uint8_t>(v);
            for (int u = min_u; u <= max_u; ++u) {
                if (assigned_row[u])
                    continue;

                double a0, a1, a2;
                if (!barycentric(Eigen::Vector2d(u, v), p0, p1, p2, a0, a1, a2))
                    continue;
                constexpr double eps = -1e-6;
                if (a0 < eps || a1 < eps || a2 < eps)
                    continue;

                const float disp = disp_row[u];
                if (disp <= 0.0f || !std::isfinite(disp))
                    continue;

                const double depth = f_rect * baseline / static_cast<double>(disp);
                if (depth < min_depth || depth > max_depth || !std::isfinite(depth))
                    continue;

                DepthRow row;
                row.i0 = face[0];
                row.i1 = face[1];
                row.i2 = face[2];
                row.a0 = a0;
                row.a1 = a1;
                row.a2 = a2;
                row.inv_depth = 1.0 / depth;
                rows.push_back(row);
                assigned_row[u] = 1;
            }
        }
    }

    return rows;
}

} // namespace

DenseMesh PrimalDualMeshEstimator::estimate(const cv::Mat& disp_float,
                                             double f_rect,
                                             double baseline,
                                             double cx_rect,
                                             double cy_rect,
                                             const Eigen::Affine3d& T_w_rectcam,
                                             const cv::Mat& R1_inv,
                                             const std::shared_ptr<Frame>& frame) {
    (void)R1_inv;

    DenseMesh mesh;
    if (disp_float.empty() || f_rect <= 0.0 || baseline <= 0.0)
        return mesh;

    const int img_h = disp_float.rows;
    const int img_w = disp_float.cols;
    const double inv_min_depth = 1.0 / _cfg.max_depth;
    const double inv_max_depth = 1.0 / _cfg.min_depth;

    std::vector<Vertex2D> vertices_2d;
    std::map<std::pair<int, int>, int> rounded_lookup;

    auto add_vertex = [&](double u, double v, bool has_track, double inv_track) {
        u = clampScalar(u, 0.0, static_cast<double>(img_w - 1));
        v = clampScalar(v, 0.0, static_cast<double>(img_h - 1));
        addOrUpdateVertex(vertices_2d, rounded_lookup, u, v, has_track, inv_track);
    };

    if (frame && !frame->getSensors().empty()) {
        const Eigen::Affine3d T_cam_world = T_w_rectcam.inverse();
        const auto landmarks = frame->getLandmarks();
        for (const auto& type_lmks : landmarks) {
            for (const auto& lmk : type_lmks.second) {
                if (!lmk || lmk->isOutlier() || lmk->isMarg())
                    continue;

                const Eigen::Vector3d p_cam = T_cam_world * lmk->getPose().translation();
                if (p_cam.z() < _cfg.min_depth || p_cam.z() > _cfg.max_depth || !std::isfinite(p_cam.z()))
                    continue;

                const double u = f_rect * p_cam.x() / p_cam.z() + cx_rect;
                const double v = f_rect * p_cam.y() / p_cam.z() + cy_rect;
                if (u >= 0.0 && u < img_w && v >= 0.0 && v < img_h)
                    add_vertex(u, v, true, 1.0 / p_cam.z());
            }
        }
    }

    const int spacing = std::max(2, _cfg.steiner_spacing);
    for (int v = spacing / 2; v < img_h; v += spacing) {
        for (int u = spacing / 2; u < img_w; u += spacing)
            add_vertex(u, v, false, 0.0);
    }
    for (int u = 0; u < img_w; u += spacing) {
        add_vertex(u, 0, false, 0.0);
        add_vertex(u, img_h - 1, false, 0.0);
    }
    for (int v = 0; v < img_h; v += spacing) {
        add_vertex(0, v, false, 0.0);
        add_vertex(img_w - 1, v, false, 0.0);
    }
    add_vertex(0, 0, false, 0.0);
    add_vertex(img_w - 1, 0, false, 0.0);
    add_vertex(0, img_h - 1, false, 0.0);
    add_vertex(img_w - 1, img_h - 1, false, 0.0);

    if (vertices_2d.size() < 3)
        return mesh;

    std::vector<Eigen::Vector3i> faces = buildDelaunayFaces(vertices_2d, rounded_lookup, img_w, img_h);
    if (faces.empty())
        return mesh;

    std::vector<DepthRow> depth_rows = rasterizeDenseDepthRows(disp_float,
                                                               vertices_2d,
                                                               faces,
                                                               f_rect,
                                                               baseline,
                                                               _cfg.min_depth,
                                                               _cfg.max_depth);

    const int nv = static_cast<int>(vertices_2d.size());
    Eigen::VectorXd xi(nv);
    Eigen::VectorXd w1 = Eigen::VectorXd::Zero(nv);
    Eigen::VectorXd w2 = Eigen::VectorXd::Zero(nv);

    double mean_inv_depth = 0.5 * (inv_min_depth + inv_max_depth);
    if (!depth_rows.empty()) {
        mean_inv_depth = 0.0;
        for (const auto& row : depth_rows)
            mean_inv_depth += row.inv_depth;
        mean_inv_depth /= static_cast<double>(depth_rows.size());
    }

    for (int i = 0; i < nv; ++i) {
        const int u = static_cast<int>(std::round(vertices_2d[i].uv.x()));
        const int v = static_cast<int>(std::round(vertices_2d[i].uv.y()));
        const float disp = disp_float.at<float>(std::max(0, std::min(img_h - 1, v)),
                                                std::max(0, std::min(img_w - 1, u)));
        if (disp > 0.0f && std::isfinite(disp)) {
            const double depth = f_rect * baseline / static_cast<double>(disp);
            if (depth >= _cfg.min_depth && depth <= _cfg.max_depth)
                xi(i) = 1.0 / depth;
            else
                xi(i) = vertices_2d[i].has_tracking_depth ? vertices_2d[i].tracking_inv_depth : mean_inv_depth;
        } else {
            xi(i) = vertices_2d[i].has_tracking_depth ? vertices_2d[i].tracking_inv_depth : mean_inv_depth;
        }
        xi(i) = clampScalar(xi(i), inv_min_depth, inv_max_depth);
    }

    std::vector<EdgeOp> edges = buildEdgeOperators(faces, vertices_2d);
    std::vector<Eigen::Vector3d> q(edges.size(), Eigen::Vector3d::Zero());
    std::vector<double> p(depth_rows.size(), 0.0);

    Eigen::VectorXd xi_bar = xi;
    Eigen::VectorXd w1_bar = w1;
    Eigen::VectorXd w2_bar = w2;

    for (int iter = 0; iter < _cfg.num_iterations; ++iter) {
        for (size_t e = 0; e < edges.size(); ++e) {
            const EdgeOp& op = edges[e];
            Eigen::Vector3d d;
            d(0) = op.alpha * (xi_bar(op.i) - xi_bar(op.j) -
                               w1_bar(op.i) * op.du.x() -
                               w2_bar(op.i) * op.du.y());
            d(1) = op.beta * (w1_bar(op.i) - w1_bar(op.j));
            d(2) = op.beta * (w2_bar(op.i) - w2_bar(op.j));
            q[e] += _cfg.sigma * d;
            q[e](0) = clampScalar(q[e](0), -1.0, 1.0);
            q[e](1) = clampScalar(q[e](1), -1.0, 1.0);
            q[e](2) = clampScalar(q[e](2), -1.0, 1.0);
        }

        for (size_t d = 0; d < depth_rows.size(); ++d) {
            const DepthRow& row = depth_rows[d];
            const double pred = row.a0 * xi_bar(row.i0) +
                                row.a1 * xi_bar(row.i1) +
                                row.a2 * xi_bar(row.i2);
            p[d] = clampScalar(p[d] + _cfg.sigma * _cfg.lambda * (pred - row.inv_depth), -1.0, 1.0);
        }

        Eigen::VectorXd grad_xi = Eigen::VectorXd::Zero(nv);
        Eigen::VectorXd grad_w1 = Eigen::VectorXd::Zero(nv);
        Eigen::VectorXd grad_w2 = Eigen::VectorXd::Zero(nv);

        for (size_t e = 0; e < edges.size(); ++e) {
            const EdgeOp& op = edges[e];
            const Eigen::Vector3d& qe = q[e];
            grad_xi(op.i) += op.alpha * qe(0);
            grad_w1(op.i) += -op.alpha * op.du.x() * qe(0) + op.beta * qe(1);
            grad_w2(op.i) += -op.alpha * op.du.y() * qe(0) + op.beta * qe(2);
            grad_xi(op.j) += -op.alpha * qe(0);
            grad_w1(op.j) += -op.beta * qe(1);
            grad_w2(op.j) += -op.beta * qe(2);
        }

        for (size_t d = 0; d < depth_rows.size(); ++d) {
            const DepthRow& row = depth_rows[d];
            const double weighted_p = _cfg.lambda * p[d];
            grad_xi(row.i0) += row.a0 * weighted_p;
            grad_xi(row.i1) += row.a1 * weighted_p;
            grad_xi(row.i2) += row.a2 * weighted_p;
        }

        Eigen::VectorXd xi_next = xi - _cfg.tau * grad_xi;
        Eigen::VectorXd w1_next = w1 - _cfg.tau * grad_w1;
        Eigen::VectorXd w2_next = w2 - _cfg.tau * grad_w2;

        for (int i = 0; i < nv; ++i) {
            if (vertices_2d[i].has_tracking_depth) {
                xi_next(i) = shrinkToReference(xi_next(i),
                                               clampScalar(vertices_2d[i].tracking_inv_depth, inv_min_depth, inv_max_depth),
                                               _cfg.tau * _cfg.lambda);
            }
            xi_next(i) = clampScalar(xi_next(i), inv_min_depth, inv_max_depth);
        }

        xi_bar = xi_next + _cfg.theta * (xi_next - xi);
        w1_bar = w1_next + _cfg.theta * (w1_next - w1);
        w2_bar = w2_next + _cfg.theta * (w2_next - w2);

        xi = std::move(xi_next);
        w1 = std::move(w1_next);
        w2 = std::move(w2_next);
    }

    mesh.vertices.resize(nv);
    for (int i = 0; i < nv; ++i) {
        const double depth = 1.0 / xi(i);
        const double X = (vertices_2d[i].uv.x() - cx_rect) * depth / f_rect;
        const double Y = (vertices_2d[i].uv.y() - cy_rect) * depth / f_rect;
        mesh.vertices[i] = T_w_rectcam * Eigen::Vector3d(X, Y, depth);
    }
    mesh.faces = std::move(faces);
    mesh.computeFaceNormals();
    return mesh;
}

} // namespace isae
