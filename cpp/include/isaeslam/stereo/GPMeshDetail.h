#ifndef GP_MESH_DETAIL_H
#define GP_MESH_DETAIL_H

// Per-cell building blocks of the SLAMesh-style GP mesher, shared by GPMeshEstimator
// (one keyframe at a time) and GPGlobalMap (fused world map). Implemented in
// GPMeshEstimator.cpp.

#include "isaeslam/stereo/DenseMesh.h"
#include "isaeslam/stereo/GPMeshEstimator.h"
#include <Eigen/Core>
#include <array>
#include <vector>

namespace isae {
namespace gp_detail {

// World-aligned voxel index: cell (x, y, z) spans [x, x+1) * cell_size along each axis.
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

// One GP surface layer of a cell, as laid out in DenseMesh::vertices.
struct PatchInfo {
    CellKey cell;
    int prediction_axis = 0;
    int base_vertex = 0;
    std::vector<float> variances;
};

Eigen::Vector3d cellMin(const CellKey& key, double cell_size);

// SLAMesh PCA direction choice. Returns false for cells that are not surface-like.
bool chooseDirections(const std::vector<Eigen::Vector3d>& points,
                      const GPMeshConfig& cfg,
                      std::array<bool, 3>& directions);

// GP regression of the prediction-axis coordinate over the cell's num_test x num_test lattice.
// Vertex i of the lattice is at (row, col) = (i / num_test, i % num_test) for every call with the
// same cell and axis, which is what lets SLAMesh pair vertices by index.
bool gaussianProcessLayer(const std::vector<Eigen::Vector3d>& cell_points,
                          const Region& region,
                          int prediction_axis,
                          const GPMeshConfig& cfg,
                          std::vector<Eigen::Vector3d>& vertices,
                          std::vector<float>& variances);

void appendGridFaces(DenseMesh& mesh, int base_vertex, const std::vector<float>& variances,
                     const GPMeshConfig& cfg);

void appendSeamFaces(DenseMesh& mesh, const std::vector<PatchInfo>& patches, const GPMeshConfig& cfg);

} // namespace gp_detail
} // namespace isae
#endif
