#ifndef GP_GLOBAL_MAP_H
#define GP_GLOBAL_MAP_H

// Fused world-frame GP mesh map built from per-keyframe dense stereo points, following SLAMesh
// (Ruan et al., ICRA 2023):
//
//   A. Map update  - SLAMesh Map::updateMap / Cell::updateVertices. Every cell of the world grid
//                    keeps, per prediction axis, a num_test x num_test lattice of GP vertices and
//                    variances. A new keyframe's lattice is fused into the stored one vertex by
//                    vertex with inverse-variance weighting.
//   B. Registration (optional) - SLAMesh Map::registerToMap. Before fusing, the keyframe's lattice
//                    is registered to the map by point-to-mesh ICP, pairing vertices by lattice
//                    index (no kd-tree), starting from the VO pose.
//
// Deviations from SLAMesh are marked "DEVIATION" in GPGlobalMap.cpp and listed in
// doc/dense_gp_global_map.md.

#include "isaeslam/stereo/DenseMesh.h"
#include "isaeslam/stereo/GPMeshDetail.h"
#include "isaeslam/stereo/GPMeshEstimator.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <map>
#include <string>
#include <vector>

namespace isae {

struct GPGlobalMapConfig {
    GPMeshConfig gp;                          // per-cell GP settings, shared with GPMeshEstimator

    // A. map update (SLAMesh variance_map_update)
    double variance_map_update     = 0.5;    // fuse two vertices only if both variances are below this
    int    max_raw_points_per_cell = 2000;   // DEVIATION: cap on raw points kept for non-surface cells

    // B. frame-to-model registration (SLAMesh registerToMap), off by default
    bool   register_enabled          = false;
    int    register_times            = 5;     // SLAMesh register_times
    double variance_register         = 0.1;   // SLAMesh variance_register (its code default; online.yaml uses 0.6)
    int    cross_cell_overlap_length = 1;     // SLAMesh cross_cell_overlap_length
    double converge_thr              = 1e-5;  // SLAMesh converge_thr
    double huber_delta               = 0.1;   // SLAMesh HuberLoss(0.1)
    int    solver_iterations         = 10;    // SLAMesh ceres max_num_iterations
    int    min_matches               = 30;    // DEVIATION: keep the VO pose below this many pairs
    double max_correction_translation   = 0.5;  // DEVIATION: reject larger corrections [m]
    double max_correction_rotation_deg  = 10.0; // DEVIATION: reject larger corrections [deg]
    bool   carry_correction          = true;  // start each keyframe from the last accepted correction
    bool   depth_weighting           = false; // DEVIATION (opt-in): residual weight min(1, (ref/z)^2)
    double depth_weighting_ref       = 2.0;   // metres
};

struct GPIntegrationReport {
    Eigen::Affine3d T_w_cam_vo    = Eigen::Affine3d::Identity(); // pose handed in by VO
    Eigen::Affine3d T_w_cam_guess = Eigen::Affine3d::Identity(); // VO pose with the carried correction
    Eigen::Affine3d T_w_cam_used  = Eigen::Affine3d::Identity(); // pose the keyframe was fused with
    bool        register_attempted = false;
    bool        register_accepted  = false;
    int         iterations         = 0;
    int         matches            = 0;   // pairs in the last solve
    std::string status;                    // why registration was skipped / rejected, or "ok"
    int         scan_cells         = 0;
    int         cells_fused        = 0;   // existing cells updated
    int         cells_added        = 0;
    int         map_cells          = 0;
};

class GPGlobalMap {
  public:
    explicit GPGlobalMap(const GPGlobalMapConfig& cfg) : _cfg(cfg) {}

    // points_cam: dense points in the keyframe's (rectified left) camera frame.
    // T_w_cam: that camera's pose from VO.
    GPIntegrationReport integrate(const std::vector<Eigen::Vector3d>& points_cam,
                                  const Eigen::Affine3d& T_w_cam);

    // Global mesh: every stored lattice, grid faces and seams filtered by gp.max_variance.
    DenseMesh mesh() const;

    // Binary little-endian PLY with a per-vertex "variance" property. Written to path + ".tmp"
    // and renamed, so a process killed mid-write never leaves a truncated file.
    bool savePly(const std::string& path) const;

    size_t numCells() const { return _cells.size(); }

  private:
    struct Layer {
        bool valid = false;
        std::vector<Eigen::Vector3d> vertices; // num_test^2, lattice order
        std::vector<float> variances;
    };
    struct Cell {
        bool not_surface = false;
        std::vector<Eigen::Vector3d> raw_points; // kept only while not_surface
        std::array<Layer, 3> layers;
    };
    using CellMap = std::map<gp_detail::CellKey, Cell>;

    // Pairs found for one registration iteration (SLAMesh ary_overlap_vertices / ary_normal).
    struct Matches {
        std::vector<Eigen::Vector3d> scan, map, normal;
        std::vector<double> weight;
    };

    CellMap buildCells(const std::vector<Eigen::Vector3d>& points_world) const;
    void    reconstructCell(const gp_detail::CellKey& key, Cell& cell) const;
    Matches findMatchPointToMesh(const CellMap& scan, int overlap_length,
                                 const Eigen::Affine3d& T_w_cam) const;
    Eigen::Affine3d computeTPointToMesh(const Matches& m) const;
    void    updateMap(const CellMap& scan, GPIntegrationReport& report);
    void    fuseLayer(Layer& map_layer, const Layer& scan_layer, int axis) const;
    void    capRawPoints(std::vector<Eigen::Vector3d>& points) const;

    GPGlobalMapConfig _cfg;
    CellMap _cells;
    Eigen::Affine3d _carried = Eigen::Affine3d::Identity();
};

} // namespace isae
#endif
