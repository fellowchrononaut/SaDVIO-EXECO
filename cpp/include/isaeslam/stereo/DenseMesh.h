#ifndef DENSE_MESH_H
#define DENSE_MESH_H

#include <Eigen/Core>
#include <vector>
#include <cmath>

namespace isae {

struct DihedralEdge {
    int v0, v1;    // edge vertex indices
    double angle;  // dihedral angle in radians
};

struct DenseMesh {
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> faces;           // CCW winding
    std::vector<Eigen::Vector3d> face_normals;    // unit normals
    std::vector<float>           vertex_variance; // GP uncertainty (GP only)
    std::vector<double>          gaussian_curvature;
    std::vector<double>          mean_curvature;

    void computeFaceNormals();
    void computeCurvature();
    std::vector<DihedralEdge> computeDihedralAngles() const;
};

} // namespace isae

#endif
