#include "isaeslam/stereo/DenseMesh.h"
#include <Eigen/Geometry>
#include <map>
#include <set>
#include <unordered_map>
#include <cmath>

namespace isae {

void DenseMesh::computeFaceNormals() {
    face_normals.resize(faces.size());
    for (size_t f = 0; f < faces.size(); ++f) {
        const Eigen::Vector3d& v0 = vertices[faces[f][0]];
        const Eigen::Vector3d& v1 = vertices[faces[f][1]];
        const Eigen::Vector3d& v2 = vertices[faces[f][2]];
        Eigen::Vector3d n = (v1 - v0).cross(v2 - v0);
        double len = n.norm();
        if (len > 1e-12)
            face_normals[f] = n / len;
        else
            face_normals[f] = Eigen::Vector3d::Zero();
    }
}

void DenseMesh::computeCurvature() {
    size_t NV = vertices.size();
    size_t NF = faces.size();

    gaussian_curvature.assign(NV, 0.0);
    mean_curvature.assign(NV, 0.0);

    // For each vertex: accumulate angle sum and mixed area
    std::vector<double> angle_sum(NV, 0.0);
    std::vector<double> mixed_area(NV, 0.0);

    // For mean curvature: accumulate cotangent weighted sum
    // H_i = || sum_{j~i} (cot a_ij + cot b_ij)(v_i - v_j) || / (4 * A_mixed_i)
    // (factor of 2 in denominator relative to formula, following standard derivation)
    std::vector<Eigen::Vector3d> H_vec(NV, Eigen::Vector3d::Zero());

    // Build edge -> face adjacency for cotangent formula
    // key: pair(min_v, max_v), value: list of face indices
    std::map<std::pair<int,int>, std::vector<int>> edge_faces;
    for (size_t f = 0; f < NF; ++f) {
        for (int e = 0; e < 3; ++e) {
            int a = faces[f][e];
            int b = faces[f][(e+1) % 3];
            auto key = std::make_pair(std::min(a,b), std::max(a,b));
            edge_faces[key].push_back(static_cast<int>(f));
        }
    }

    // Helper: angle at vertex 'tip' in triangle (tip, a, b)
    auto angle_at = [&](int tip, int a, int b) -> double {
        Eigen::Vector3d va = vertices[a] - vertices[tip];
        Eigen::Vector3d vb = vertices[b] - vertices[tip];
        double dot = va.dot(vb);
        double cross = va.cross(vb).norm();
        return std::atan2(cross, dot);
    };

    // Helper: cotangent of angle at vertex 'tip'
    auto cot_at = [&](int tip, int a, int b) -> double {
        Eigen::Vector3d va = vertices[a] - vertices[tip];
        Eigen::Vector3d vb = vertices[b] - vertices[tip];
        double dot = va.dot(vb);
        double cross = va.cross(vb).norm();
        if (cross < 1e-12) return 0.0;
        return dot / cross;
    };

    for (size_t f = 0; f < NF; ++f) {
        int i0 = faces[f][0];
        int i1 = faces[f][1];
        int i2 = faces[f][2];

        // Triangle area
        Eigen::Vector3d e01 = vertices[i1] - vertices[i0];
        Eigen::Vector3d e02 = vertices[i2] - vertices[i0];
        double area = 0.5 * e01.cross(e02).norm();
        if (area < 1e-15) continue;

        // Angles at each vertex
        double a0 = angle_at(i0, i1, i2);
        double a1 = angle_at(i1, i0, i2);
        double a2 = angle_at(i2, i0, i1);

        angle_sum[i0] += a0;
        angle_sum[i1] += a1;
        angle_sum[i2] += a2;

        // Mixed Voronoi area contribution
        // Obtuse check: if obtuse at vertex k, use area/2 for k and area/4 for others
        bool obtuse0 = (a0 > M_PI / 2.0);
        bool obtuse1 = (a1 > M_PI / 2.0);
        bool obtuse2 = (a2 > M_PI / 2.0);

        if (obtuse0 || obtuse1 || obtuse2) {
            if (obtuse0) {
                mixed_area[i0] += area / 2.0;
                mixed_area[i1] += area / 4.0;
                mixed_area[i2] += area / 4.0;
            } else if (obtuse1) {
                mixed_area[i0] += area / 4.0;
                mixed_area[i1] += area / 2.0;
                mixed_area[i2] += area / 4.0;
            } else {
                mixed_area[i0] += area / 4.0;
                mixed_area[i1] += area / 4.0;
                mixed_area[i2] += area / 2.0;
            }
        } else {
            mixed_area[i0] += area / 3.0;
            mixed_area[i1] += area / 3.0;
            mixed_area[i2] += area / 3.0;
        }

        // Cotangent formula for mean curvature
        // For each edge (i,j), the opposite vertex in this face provides one cotangent
        // edge (i0,i1): opposite angle is a2 at i2
        // edge (i1,i2): opposite angle is a0 at i0
        // edge (i0,i2): opposite angle is a1 at i1
        double cot2 = cot_at(i2, i0, i1); // opposite to edge (i0,i1)
        double cot0 = cot_at(i0, i1, i2); // opposite to edge (i1,i2)
        double cot1 = cot_at(i1, i0, i2); // opposite to edge (i0,i2)

        H_vec[i0] += cot2 * (vertices[i0] - vertices[i1]);
        H_vec[i1] += cot2 * (vertices[i1] - vertices[i0]);

        H_vec[i1] += cot0 * (vertices[i1] - vertices[i2]);
        H_vec[i2] += cot0 * (vertices[i2] - vertices[i1]);

        H_vec[i0] += cot1 * (vertices[i0] - vertices[i2]);
        H_vec[i2] += cot1 * (vertices[i2] - vertices[i0]);
    }

    // Finalize curvatures
    for (size_t i = 0; i < NV; ++i) {
        double A = mixed_area[i];
        if (A < 1e-15) {
            gaussian_curvature[i] = 0.0;
            mean_curvature[i] = 0.0;
            continue;
        }
        // Gaussian curvature: angle deficit
        gaussian_curvature[i] = (2.0 * M_PI - angle_sum[i]) / A;
        // Mean curvature: cotangent formula magnitude
        mean_curvature[i] = H_vec[i].norm() / (2.0 * A);
    }
}

std::vector<DihedralEdge> DenseMesh::computeDihedralAngles() const {
    // Build edge -> face adjacency
    std::map<std::pair<int,int>, std::vector<int>> edge_faces;
    for (size_t f = 0; f < faces.size(); ++f) {
        for (int e = 0; e < 3; ++e) {
            int a = faces[f][e];
            int b = faces[f][(e+1) % 3];
            auto key = std::make_pair(std::min(a,b), std::max(a,b));
            edge_faces[key].push_back(static_cast<int>(f));
        }
    }

    std::vector<DihedralEdge> result;
    for (auto& kv : edge_faces) {
        if (kv.second.size() != 2)
            continue; // boundary edge or non-manifold, skip

        int f0 = kv.second[0];
        int f1 = kv.second[1];

        if (f0 >= static_cast<int>(face_normals.size()) ||
            f1 >= static_cast<int>(face_normals.size()))
            continue;

        const Eigen::Vector3d& n0 = face_normals[f0];
        const Eigen::Vector3d& n1 = face_normals[f1];

        double cos_angle = n0.dot(n1);
        // Clamp for numerical safety
        cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
        double angle = std::acos(cos_angle);

        DihedralEdge de;
        de.v0    = kv.first.first;
        de.v1    = kv.first.second;
        de.angle = angle;
        result.push_back(de);
    }

    return result;
}

} // namespace isae
