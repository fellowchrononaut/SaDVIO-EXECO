#include "isaeslam/stereo/PrimalDualMeshEstimator.h"
#include "isaeslam/data/sensors/ASensor.h"
#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace isae {

DenseMesh PrimalDualMeshEstimator::estimate(const cv::Mat& disp_float,
                                             double f_rect, double baseline,
                                             double cx_rect, double cy_rect,
                                             const Eigen::Affine3d& T_w_rectcam,
                                             const cv::Mat& R1_inv,
                                             const std::shared_ptr<Frame>& frame) {
    DenseMesh mesh;

    const int img_h = disp_float.rows;
    const int img_w = disp_float.cols;

    // -------------------------------------------------------------------------
    // Step 1: Build vertex set
    // - VIO landmarks projected to 2D rectified image
    // - Steiner grid every steiner_spacing pixels
    // -------------------------------------------------------------------------
    std::vector<Eigen::Vector2d> pts_2d;

    // Project VIO landmarks into rectified image coordinates
    // R1_inv maps from rectified to original cam; R1 = R1_inv^T maps original to rectified
    // So p_rect = R1 * K_orig^{-1} * (p_orig - c_orig) -- but here we directly use
    // the frame's world transform and rectification matrix to get rectified coords.
    // Approach: for each landmark, get its 3D world position, project to left camera,
    // then rotate by R1 (original -> rectified) and project with rectified intrinsics.
    Eigen::Matrix3d R1_eig;
    cv::cv2eigen(R1_inv.t(), R1_eig); // R1 = R1_inv.transpose()

    if (frame && !frame->getSensors().empty()) {
        // Get T_cam_world for the left (rectified) camera
        Eigen::Affine3d T_cam_world = T_w_rectcam.inverse();

        auto landmarks = frame->getLandmarks();
        for (auto& type_lmks : landmarks) {
            for (auto& lmk : type_lmks.second) {
                if (!lmk || lmk->isOutlier() || lmk->isMarg())
                    continue;

                Eigen::Vector3d p_world = lmk->getPose().translation();
                // Transform to rectified camera frame
                Eigen::Vector3d p_cam = T_cam_world * p_world;

                if (p_cam.z() <= 0.0)
                    continue;

                // Project with rectified intrinsics
                double u = f_rect * p_cam.x() / p_cam.z() + cx_rect;
                double v = f_rect * p_cam.y() / p_cam.z() + cy_rect;

                if (u >= 0 && u < img_w && v >= 0 && v < img_h)
                    pts_2d.push_back(Eigen::Vector2d(u, v));
            }
        }
    }

    // Add Steiner grid points
    for (int v = _cfg.steiner_spacing / 2; v < img_h; v += _cfg.steiner_spacing) {
        for (int u = _cfg.steiner_spacing / 2; u < img_w; u += _cfg.steiner_spacing) {
            pts_2d.push_back(Eigen::Vector2d(static_cast<double>(u), static_cast<double>(v)));
        }
    }

    if (pts_2d.size() < 3) {
        return mesh; // not enough vertices
    }

    int NV = static_cast<int>(pts_2d.size());

    // -------------------------------------------------------------------------
    // Step 2: Delaunay triangulation of 2D vertex positions using cv::Subdiv2D
    // -------------------------------------------------------------------------
    cv::Rect rect(0, 0, img_w, img_h);
    cv::Subdiv2D subdiv(rect);

    for (auto& pt : pts_2d) {
        float fx = static_cast<float>(pt.x());
        float fy = static_cast<float>(pt.y());
        // Clamp to avoid assertion errors at boundary
        fx = std::max(0.0f, std::min(fx, static_cast<float>(img_w - 1)));
        fy = std::max(0.0f, std::min(fy, static_cast<float>(img_h - 1)));
        subdiv.insert(cv::Point2f(fx, fy));
    }

    std::vector<cv::Vec6f> triangles_cv;
    subdiv.getTriangleList(triangles_cv);

    // Build face list: map cv triangle vertices back to our vertex indices
    // Use nearest-point lookup via a simple map on rounded coordinates
    std::map<std::pair<int,int>, int> pt_to_idx;
    for (int i = 0; i < NV; ++i) {
        int ru = static_cast<int>(std::round(pts_2d[i].x()));
        int rv = static_cast<int>(std::round(pts_2d[i].y()));
        pt_to_idx[{ru, rv}] = i;
    }

    std::vector<Eigen::Vector3i> tri_indices;
    for (auto& tri : triangles_cv) {
        // tri = [x0 y0 x1 y1 x2 y2]
        auto find_idx = [&](float x, float y) -> int {
            int rx = static_cast<int>(std::round(x));
            int ry = static_cast<int>(std::round(y));
            auto it = pt_to_idx.find({rx, ry});
            if (it == pt_to_idx.end()) return -1;
            return it->second;
        };

        int i0 = find_idx(tri[0], tri[1]);
        int i1 = find_idx(tri[2], tri[3]);
        int i2 = find_idx(tri[4], tri[5]);

        if (i0 < 0 || i1 < 0 || i2 < 0) continue;

        // Check vertices are within image
        auto in_img = [&](int idx) -> bool {
            double u = pts_2d[idx].x();
            double v = pts_2d[idx].y();
            return u >= 0 && u < img_w && v >= 0 && v < img_h;
        };
        if (!in_img(i0) || !in_img(i1) || !in_img(i2)) continue;

        tri_indices.push_back(Eigen::Vector3i(i0, i1, i2));
    }

    // -------------------------------------------------------------------------
    // Step 3: Initialize inverse depths from SGBM at vertex positions
    // -------------------------------------------------------------------------
    const double inv_min_depth = 1.0 / _cfg.max_depth;
    const double inv_max_depth = 1.0 / _cfg.min_depth;

    Eigen::VectorXd eta(NV);
    std::vector<bool> has_disp(NV, false);

    for (int i = 0; i < NV; ++i) {
        int u = static_cast<int>(std::round(pts_2d[i].x()));
        int v = static_cast<int>(std::round(pts_2d[i].y()));
        u = std::max(0, std::min(u, img_w - 1));
        v = std::max(0, std::min(v, img_h - 1));

        float disp = disp_float.at<float>(v, u);
        if (disp > 0.0f && std::isfinite(disp)) {
            double depth = f_rect * baseline / disp;
            double inv_d = 1.0 / depth;
            // clamp
            inv_d = std::max(inv_min_depth, std::min(inv_max_depth, inv_d));
            eta(i) = inv_d;
            has_disp[i] = true;
        } else {
            // Initialize to mid-range if no disparity
            eta(i) = 0.5 * (inv_min_depth + inv_max_depth);
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: Precompute H and q matrices
    // H = A^T M A  (data term)  where A projects inverse depth to depth observation
    // For each vertex i with a valid disparity:
    //   data term: 0.5 * (eta_i - b_i)^2  where b_i = 1/depth_i
    // So H = diag(mask_i) and q = mask_i * b_i
    // -------------------------------------------------------------------------
    Eigen::VectorXd H_diag(NV);
    Eigen::VectorXd q_vec(NV);
    for (int i = 0; i < NV; ++i) {
        if (has_disp[i]) {
            H_diag(i) = 1.0;
            q_vec(i) = eta(i); // b_i = the observed inverse depth
        } else {
            H_diag(i) = 0.0;
            q_vec(i) = 0.0;
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: Chambolle-Pock iterations
    // Build directed edge list from triangulation
    // -------------------------------------------------------------------------

    // Build directed edges from triangle adjacency
    // For each undirected edge (i,j), we create directed edge i->j and j->i
    // Weight w_e = 1.0 (uniform for now)
    struct DirEdge { int from, to; };
    std::vector<DirEdge> edges;
    std::set<std::pair<int,int>> seen_edges;

    for (auto& tri : tri_indices) {
        for (int e = 0; e < 3; ++e) {
            int a = tri[e];
            int b = tri[(e+1) % 3];
            auto key = std::make_pair(std::min(a,b), std::max(a,b));
            if (seen_edges.find(key) == seen_edges.end()) {
                seen_edges.insert(key);
                edges.push_back({a, b});
                edges.push_back({b, a});
            }
        }
    }

    int NE = static_cast<int>(edges.size());
    Eigen::VectorXd p(NE);
    p.setZero();

    Eigen::VectorXd eta_bar = eta;
    double w_e = 1.0; // uniform edge weight

    for (int iter = 0; iter < _cfg.num_iterations; ++iter) {
        // 1. Dual update: for each directed edge e=(i,j)
        //    p_e = clip(-lambda, lambda, p_e + sigma * w_e * (eta_bar_i - eta_bar_j))
        for (int e = 0; e < NE; ++e) {
            int i = edges[e].from;
            int j = edges[e].to;
            double new_p = p(e) + _cfg.sigma * w_e * (eta_bar(i) - eta_bar(j));
            p(e) = std::max(-_cfg.lambda, std::min(_cfg.lambda, new_p));
        }

        // 2. Compute divergence: for each vertex i,
        //    div_i = sum_{edges leaving i} w_e * p_e - sum_{edges entering i} w_e * p_e
        //    With directed edges, "leaving" means from==i, "entering" means to==i
        Eigen::VectorXd div(NV);
        div.setZero();
        for (int e = 0; e < NE; ++e) {
            int i = edges[e].from;
            int j = edges[e].to;
            div(i) += w_e * p(e);
            div(j) -= w_e * p(e);
        }

        // 3. Data gradient: data_grad = H * eta - q
        Eigen::VectorXd data_grad = H_diag.cwiseProduct(eta) - q_vec;

        // 4. Primal update
        Eigen::VectorXd eta_new = eta - _cfg.tau * (data_grad + div);
        // Clip to [inv_min_depth, inv_max_depth]
        for (int i = 0; i < NV; ++i) {
            eta_new(i) = std::max(inv_min_depth, std::min(inv_max_depth, eta_new(i)));
        }

        // 5. Extrapolated primal
        eta_bar = 2.0 * eta_new - eta;
        eta = eta_new;
    }

    // -------------------------------------------------------------------------
    // Step 6: Backproject final inverse depths to 3D world points
    // -------------------------------------------------------------------------
    mesh.vertices.resize(NV);
    for (int i = 0; i < NV; ++i) {
        double depth = 1.0 / eta(i);
        double Z = depth;
        double X = (pts_2d[i].x() - cx_rect) * Z / f_rect;
        double Y = (pts_2d[i].y() - cy_rect) * Z / f_rect;
        Eigen::Vector3d p_cam(X, Y, Z);
        mesh.vertices[i] = T_w_rectcam * p_cam;
    }

    mesh.faces = tri_indices;
    mesh.computeFaceNormals();

    return mesh;
}

} // namespace isae
