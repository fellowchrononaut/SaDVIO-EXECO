#ifndef DENSE_MESH_H
#define DENSE_MESH_H

#include <Eigen/Core>
#include <string>
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

// Binary little-endian PLY (x y z variance per vertex, 0 where vertex_variance is missing). Written to
// <path>.tmp and renamed, so a reader never sees a partial file.
bool writeDenseMeshPly(const DenseMesh& mesh, const std::string& path, const std::string& comment);

} // namespace isae

#endif
