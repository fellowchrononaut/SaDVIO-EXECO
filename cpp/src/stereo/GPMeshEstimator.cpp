#include "isaeslam/stereo/GPMeshEstimator.h"
#include <Eigen/Cholesky>
#include <map>
#include <algorithm>
#include <random>
#include <cmath>

namespace isae {

DenseMesh GPMeshEstimator::estimate(const cv::Mat& disp_float,
                                     double f_rect, double baseline,
                                     double cx_rect, double cy_rect,
                                     const Eigen::Affine3d& T_w_rectcam) {
    DenseMesh mesh;

    const int rows = disp_float.rows;
    const int cols = disp_float.cols;

    // Step 1: Backproject all valid SGBM pixels -> world 3D points
    // Organize by BEV cell
    struct WorldPt {
        double x, y, z;
    };

    // Key: (ix, iy) = floor(x/cell_size), floor(y/cell_size)
    std::map<std::pair<int,int>, std::vector<WorldPt>> cell_map;

    for (int v = 0; v < rows; ++v) {
        const float* row_ptr = disp_float.ptr<float>(v);
        for (int u = 0; u < cols; ++u) {
            float disp = row_ptr[u];
            if (disp <= 0.0f || !std::isfinite(disp))
                continue;

            double Z = f_rect * baseline / disp;
            double X = (u - cx_rect) * Z / f_rect;
            double Y = (v - cy_rect) * Z / f_rect;

            Eigen::Vector3d p_cam(X, Y, Z);
            Eigen::Vector3d p_world = T_w_rectcam * p_cam;

            int ix = static_cast<int>(std::floor(p_world.x() / _cfg.cell_size));
            int iy = static_cast<int>(std::floor(p_world.y() / _cfg.cell_size));

            cell_map[{ix, iy}].push_back({p_world.x(), p_world.y(), p_world.z()});
        }
    }

    // Vertex and face vectors to assemble
    std::vector<Eigen::Vector3d>& out_verts = mesh.vertices;
    std::vector<Eigen::Vector3i>& out_faces = mesh.faces;
    std::vector<float>& out_var = mesh.vertex_variance;

    std::mt19937 rng(42);

    // Step 3: For each cell with enough points, fit GP and predict
    for (auto& kv : cell_map) {
        auto& pts = kv.second;

        if (static_cast<int>(pts.size()) < _cfg.min_pts_per_cell)
            continue;

        // Subsample if too many points
        if (static_cast<int>(pts.size()) > _cfg.max_pts_per_cell) {
            std::shuffle(pts.begin(), pts.end(), rng);
            pts.resize(_cfg.max_pts_per_cell);
        }

        // Check z-range for depth discontinuities
        double z_min = pts[0].z, z_max = pts[0].z;
        for (auto& p : pts) {
            z_min = std::min(z_min, p.z);
            z_max = std::max(z_max, p.z);
        }
        if ((z_max - z_min) > _cfg.cell_size)
            continue;

        int N = static_cast<int>(pts.size());

        // Build Laplacian kernel matrix K_ii: k(i,j) = exp(-kappa * norm([xi-xj, yi-yj]))
        Eigen::MatrixXd K_ii(N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double dx = pts[i].x - pts[j].x;
                double dy = pts[i].y - pts[j].y;
                double dist = std::sqrt(dx*dx + dy*dy);
                K_ii(i, j) = std::exp(-_cfg.kappa * dist);
            }
        }

        // Regularize
        Eigen::MatrixXd A = K_ii + _cfg.sigma2_noise * Eigen::MatrixXd::Identity(N, N);

        // Cholesky decomposition
        Eigen::LLT<Eigen::MatrixXd> llt(A);
        if (llt.info() != Eigen::Success)
            continue;

        // z observations
        Eigen::VectorXd z_obs(N);
        for (int i = 0; i < N; ++i)
            z_obs(i) = pts[i].z;

        // alpha = A^{-1} z
        Eigen::VectorXd alpha = llt.solve(z_obs);

        // Build prediction grid: n_out x n_out uniform grid over cell
        int n = _cfg.n_out;
        double ix_world = kv.first.first  * _cfg.cell_size;
        double iy_world = kv.first.second * _cfg.cell_size;

        // Grid x/y positions (world frame)
        std::vector<double> xs(n), ys(n);
        for (int r = 0; r < n; ++r) {
            xs[r] = ix_world + (r + 0.5) * _cfg.cell_size / n;
            ys[r] = iy_world + (r + 0.5) * _cfg.cell_size / n;
        }

        int n2 = n * n;

        // Build K_star: n2 x N  (prediction points x training points)
        Eigen::MatrixXd K_star(n2, N);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                int idx = r * n + c;
                for (int i = 0; i < N; ++i) {
                    double dx = xs[r] - pts[i].x;
                    double dy = ys[c] - pts[i].y;
                    double dist = std::sqrt(dx*dx + dy*dy);
                    K_star(idx, i) = std::exp(-_cfg.kappa * dist);
                }
            }
        }

        // Predicted mean: z_pred = K_star * alpha
        Eigen::VectorXd z_pred = K_star * alpha;

        // Predicted variance: sigma2_j = 1 - ||L^{-1} K_star[j,:]^T ||^2
        // Solve L * X = K_star^T  (X is N x n2)
        // then sigma2_j = 1 - sum(X[:,j]^2)
        const Eigen::MatrixXd& L = llt.matrixL();
        Eigen::MatrixXd X = L.triangularView<Eigen::Lower>().solve(K_star.transpose());
        // sigma2 for each prediction point
        Eigen::VectorXd sigma2 = Eigen::VectorXd::Ones(n2) - X.colwise().squaredNorm().transpose();
        // Clamp to [0, inf)
        sigma2 = sigma2.cwiseMax(0.0);

        // Record base vertex index for this cell
        int base_vertex = static_cast<int>(out_verts.size());

        // Build vertex validity array
        std::vector<bool> valid(n2, true);
        for (int j = 0; j < n2; ++j) {
            if (sigma2(j) > _cfg.max_variance)
                valid[j] = false;
        }

        // Add vertices
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                int idx = r * n + c;
                out_verts.push_back(Eigen::Vector3d(xs[r], ys[c], z_pred(idx)));
                out_var.push_back(static_cast<float>(sigma2(idx)));
            }
        }

        // Triangulate n x n grid into 2*(n-1)^2 triangles
        for (int r = 0; r < n - 1; ++r) {
            for (int c = 0; c < n - 1; ++c) {
                int v00 = base_vertex + r * n + c;
                int v01 = base_vertex + r * n + c + 1;
                int v10 = base_vertex + (r+1) * n + c;
                int v11 = base_vertex + (r+1) * n + c + 1;

                // Local indices for validity check
                int l00 = r * n + c;
                int l01 = r * n + c + 1;
                int l10 = (r+1) * n + c;
                int l11 = (r+1) * n + c + 1;

                // Triangle 1: (r*n+c, r*n+c+1, (r+1)*n+c)
                if (valid[l00] && valid[l01] && valid[l10])
                    out_faces.push_back(Eigen::Vector3i(v00, v01, v10));

                // Triangle 2: (r*n+c+1, (r+1)*n+c+1, (r+1)*n+c)
                if (valid[l01] && valid[l11] && valid[l10])
                    out_faces.push_back(Eigen::Vector3i(v01, v11, v10));
            }
        }
    }

    // Compute face normals
    mesh.computeFaceNormals();

    return mesh;
}

} // namespace isae
