#include "isaeslam/stereo/SGBMZNCCMeshEstimator.h"

#include "utilities/imgProcessing.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <unordered_map>

namespace isae {

namespace {

int pointKey(const cv::Point2f& p, int width) {
    int u = static_cast<int>(std::lround(p.x));
    int v = static_cast<int>(std::lround(p.y));
    return v * width + u;
}

bool inPatchBounds(const cv::Mat& img, int u, int v, int half_patch) {
    return u - half_patch >= 0 && v - half_patch >= 0 &&
           u + half_patch < img.cols && v + half_patch < img.rows;
}

} // namespace

bool SGBMZNCCMeshEstimator::passGeometry(const VertexSample& a,
                                         const VertexSample& b,
                                         const VertexSample& c) const {
    const double e01 = (b.p_world - a.p_world).norm();
    const double e02 = (c.p_world - a.p_world).norm();
    const double e12 = (c.p_world - b.p_world).norm();
    if (e01 > _cfg.max_edge_length || e02 > _cfg.max_edge_length || e12 > _cfg.max_edge_length)
        return false;

    const double z_min = std::min({a.depth, b.depth, c.depth});
    const double z_max = std::max({a.depth, b.depth, c.depth});
    if (z_max - z_min > _cfg.max_depth_jump)
        return false;

    const Eigen::Vector3d n = (b.p_world - a.p_world).cross(c.p_world - a.p_world);
    return n.norm() > 1e-8;
}

bool SGBMZNCCMeshEstimator::passZNCC(const cv::Mat& left_img,
                                     const cv::Mat& right_img,
                                     const VertexSample& a,
                                     const VertexSample& b,
                                     const VertexSample& c) const {
    const int patch_size = (_cfg.zncc_patch_size % 2 == 0) ? _cfg.zncc_patch_size + 1 : _cfg.zncc_patch_size;
    const int half_patch = patch_size / 2;

    const double u_l = (a.uv.x + b.uv.x + c.uv.x) / 3.0;
    const double v_l = (a.uv.y + b.uv.y + c.uv.y) / 3.0;
    const double d   = (a.disparity + b.disparity + c.disparity) / 3.0;
    const double u_r = u_l - d;
    const double v_r = v_l;

    const int ul = static_cast<int>(std::lround(u_l));
    const int vl = static_cast<int>(std::lround(v_l));
    const int ur = static_cast<int>(std::lround(u_r));
    const int vr = static_cast<int>(std::lround(v_r));

    if (!inPatchBounds(left_img, ul, vl, half_patch) || !inPatchBounds(right_img, ur, vr, half_patch))
        return false;

    cv::Mat patch_l(left_img, cv::Rect(ul - half_patch, vl - half_patch, patch_size, patch_size));
    cv::Mat patch_r(right_img, cv::Rect(ur - half_patch, vr - half_patch, patch_size, patch_size));
    const double zncc = imgproc::ZNCC(patch_l, patch_r);
    return std::isfinite(zncc) && zncc >= _cfg.zncc_threshold;
}

DenseMesh SGBMZNCCMeshEstimator::estimate(const cv::Mat& disp,
                                          const cv::Mat& left_img,
                                          const cv::Mat& right_img,
                                          double f_rect,
                                          double baseline,
                                          double cx_rect,
                                          double cy_rect,
                                          const Eigen::Affine3d& T_w_rectcam) const {
    DenseMesh mesh;
    if (disp.empty() || left_img.empty() || right_img.empty())
        return mesh;

    const int stride = std::max(2, _cfg.vertex_stride);
    std::vector<VertexSample> samples;
    std::vector<cv::Point2f> points;
    std::unordered_map<int, int> point_to_sample;

    for (int v = 0; v < disp.rows; v += stride) {
        const float* d_row = disp.ptr<float>(v);
        for (int u = 0; u < disp.cols; u += stride) {
            const float disparity = d_row[u];
            if (disparity <= 0.f || !std::isfinite(disparity))
                continue;

            const double depth = f_rect * baseline / static_cast<double>(disparity);
            if (depth <= 0.0 || depth > _cfg.max_depth || !std::isfinite(depth))
                continue;

            VertexSample sample;
            sample.uv = cv::Point2f(static_cast<float>(u), static_cast<float>(v));
            sample.disparity = disparity;
            sample.depth = depth;
            const double x = (static_cast<double>(u) - cx_rect) * depth / f_rect;
            const double y = (static_cast<double>(v) - cy_rect) * depth / f_rect;
            sample.p_world = T_w_rectcam * Eigen::Vector3d(x, y, depth);

            const int idx = static_cast<int>(samples.size());
            point_to_sample.emplace(pointKey(sample.uv, disp.cols), idx);
            points.push_back(sample.uv);
            samples.push_back(sample);
        }
    }

    if (points.size() < 3)
        return mesh;

    cv::Subdiv2D subdiv(cv::Rect2f(0.0F, 0.0F, static_cast<float>(disp.cols), static_cast<float>(disp.rows)));
    subdiv.insert(points);

    std::vector<cv::Vec6f> triangles;
    subdiv.getTriangleList(triangles);

    std::unordered_map<int, int> sample_to_vertex;
    int accepted = 0;
    for (const auto& tri : triangles) {
        const cv::Point2f p0(tri[0], tri[1]);
        const cv::Point2f p1(tri[2], tri[3]);
        const cv::Point2f p2(tri[4], tri[5]);

        const auto it0 = point_to_sample.find(pointKey(p0, disp.cols));
        const auto it1 = point_to_sample.find(pointKey(p1, disp.cols));
        const auto it2 = point_to_sample.find(pointKey(p2, disp.cols));
        if (it0 == point_to_sample.end() || it1 == point_to_sample.end() || it2 == point_to_sample.end())
            continue;

        const int s0 = it0->second;
        const int s1 = it1->second;
        const int s2 = it2->second;
        if (s0 == s1 || s1 == s2 || s2 == s0)
            continue;

        const VertexSample& a = samples[s0];
        const VertexSample& b = samples[s1];
        const VertexSample& c = samples[s2];
        if (!passGeometry(a, b, c) || !passZNCC(left_img, right_img, a, b, c))
            continue;

        const int sample_ids[3] = {s0, s1, s2};
        Eigen::Vector3i face;
        for (int k = 0; k < 3; ++k) {
            auto it = sample_to_vertex.find(sample_ids[k]);
            if (it == sample_to_vertex.end()) {
                const int vertex_idx = static_cast<int>(mesh.vertices.size());
                sample_to_vertex.emplace(sample_ids[k], vertex_idx);
                mesh.vertices.push_back(samples[sample_ids[k]].p_world);
                face[k] = vertex_idx;
            } else {
                face[k] = it->second;
            }
        }

        const Eigen::Vector3d& v0 = mesh.vertices[face[0]];
        const Eigen::Vector3d& v1 = mesh.vertices[face[1]];
        const Eigen::Vector3d& v2 = mesh.vertices[face[2]];
        Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
        Eigen::Vector3d view_dir = ((v0 + v1 + v2) / 3.0) - T_w_rectcam.translation();
        if (n.dot(view_dir) > 0.0)
            std::swap(face[1], face[2]);

        mesh.faces.push_back(face);
        accepted++;
    }

    mesh.computeFaceNormals();
    mesh.computeCurvature();
    std::cout << "[SGBMZNCC] samples=" << samples.size()
              << "  tri2d=" << triangles.size()
              << "  accepted=" << accepted
              << "  vertices=" << mesh.vertices.size()
              << "  faces=" << mesh.faces.size() << std::endl;
    return mesh;
}

} // namespace isae
