#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#include <limits>

#include <Eigen/Eigenvalues>
#include <opencv2/core.hpp>

#include "isaeslam/typedefs.h"

namespace isae {

//< 3D geometry utilities
namespace geometry {

/*!
* @brief Computes the skew matrix of a given 3D vector
*/
template <typename Derived> inline Eigen::Matrix3d skewMatrix(const Eigen::MatrixBase<Derived> &w) {
    Eigen::Matrix3d skew;
    skew << 0, -w[2], w[1], //
        w[2], 0, -w[0],     //
        -w[1], w[0], 0;     //
    return skew;
}

/*!
* @brief Computes the 3D vector from a given skew matrix
*/
inline Eigen::Vector3d FromskewMatrix(const Eigen::Matrix3d &skew) {
    Eigen::Vector3d w(skew(2, 1), skew(0, 2), skew(1, 0));
    return w;
}

/*!
* @brief Compute the right jacobian on SO(3) manifold
*
* From "A micro Lie theory for state estimation in robotics" Solà et al 
* Source https://arxiv.org/abs/1812.01537
*/
inline Eigen::Matrix3d so3_rightJacobian(const Eigen::Vector3d &w) {
    double w_norm = w.norm();
    Eigen::Matrix3d w_skew = skewMatrix(w);
    if (w_norm < 1e-5)
        return Eigen::Matrix3d::Identity();
    return Eigen::Matrix3d::Identity() - ((1 - cos(w_norm)) / (w_norm * w_norm)) * w_skew +
           ((w_norm - sin(w_norm)) / (w_norm * w_norm * w_norm)) * w_skew * w_skew;
}

/*!
* @brief Compute the left jacobian on SO(3) manifold
*
* From "A micro Lie theory for state estimation in robotics" Solà et al 
* Source https://arxiv.org/abs/1812.01537
*/
inline Eigen::Matrix3d so3_leftJacobian(const Eigen::Vector3d &w) {
    double w_norm          = w.norm();
    Eigen::Matrix3d w_skew = skewMatrix(w);
    // Small angle: first-order expansion (the closed form divides by |w|)
    if (w_norm < 1e-5)
        return Eigen::Matrix3d::Identity() + 0.5 * w_skew;
    return Eigen::Matrix3d::Identity() + ((1 - cos(w_norm)) / (w_norm * w_norm)) * w_skew +
           ((w_norm - sin(w_norm)) / (w_norm * w_norm * w_norm)) * w_skew * w_skew;
}

/*!
* @brief Compute the Euler angle representation of a rotation matrix
*/
inline Eigen::Vector3d rotationMatrixToEulerAnglesEigen(Eigen::Matrix3d R) {

    float sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular) {
        x = atan2(R(2, 1), R(2, 2));
        y = atan2(-R(2, 0), sy);
        z = atan2(R(1, 0), R(0, 0));
    } else {
        x = atan2(-R(1, 2), R(1, 1));
        y = atan2(-R(2, 0), sy);
        z = 0;
    }
    return Eigen::Vector3d(x * 180 / 3.1416, y * 180 / 3.1416, z * 180 / 3.1416);
}

/*!
* @brief Rotation matrix that rotates a right handed frame with X front to a frame with Z front
*/
inline Eigen::Matrix3d xFront2zFront() {
    Eigen::Matrix3d xF2zF;
    xF2zF << 0., -1., 0., //
        0., 0., -1.,      //
        1., 0., 0.;       //
    return xF2zF;
}

/*!
* @brief Rotation matrix that rotates a right handed frame with Z front to a frame with X front
*/
inline Eigen::Matrix3d zFront2xFront() {
    Eigen::Matrix3d zF2xF;
    zF2xF << 0, 0, 1, //
        -1, 0, 0,     //
        0, -1, 0;     //
    return zF2xF;
}

/*!
* @brief Rotation matrix that rotates a right handed frame with X right to a frame with X front
*/
inline Eigen::Matrix3d xRight2xFront() {
    Eigen::Matrix3d xR2xF;
    xR2xF << 0., 1., 0., //
        -1., 0., 0.,     //
        0., 0., 1.;      //
    return xR2xF;
}


/*!
* @brief Gives the rotation matrix from a direction vector
*/
inline Eigen::Matrix3d directionVector2Rotation(Eigen::Vector3d v,
                                                const Eigen::Vector3d &reference_vector = Eigen::Vector3d(1, 0, 0)) {

    // v = R*reference
    // Rodrigues's rotation formula gives the result of a rotation of a unit vector a about an axis of
    // rotation k through the angle θ. We can make use of this by realizing that, in order to bring
    // a normalized vector a into coincidence with another normalized vector b, we simply need to
    // rotate a about k=(a+b)/2 by the angle pi.
    v                   = v.normalized(); // assure v is normalized
    Eigen::Vector3d sum = v + reference_vector;
    Eigen::Matrix3d R   = 2.0 * (sum * sum.transpose()) / (sum.transpose() * sum) - Eigen::Matrix3d::Identity();
    return R;
}

/*!
* @brief Gives the direction vector from a rotation matrix
*/
inline Eigen::Vector3d Rotation2directionVector(const Eigen::Matrix3d &R,
                                                const Eigen::Vector3d &reference_vector = Eigen::Vector3d(1, 0, 0)) {
    Eigen::Vector3d v = R * reference_vector;
    return v.normalized(); // assure v is normalized (should already be)
}

/*!
* @brief Compute the exponential of the SO(3) manifold
*
* From "A micro Lie theory for state estimation in robotics" Solà et al 
* Source https://arxiv.org/abs/1812.01537
*/
inline Eigen::Matrix3d exp_so3(const Eigen::Vector3d &v) {

    double tolerance = 1e-9;
    double angle     = v.norm();

    Eigen::Matrix3d Rot;
    if (angle < tolerance) {
        // Near |phi|==0, use first order Taylor expansion
        Rot = Eigen::Matrix3d::Identity() + geometry::skewMatrix(v);
    } else {
        Eigen::Vector3d axis = v / angle;
        Eigen::Matrix3d skew = geometry::skewMatrix(axis);
        Rot                  = Eigen::Matrix3d::Identity() + (1. - cos(angle)) * skew * skew +
              sin(angle) * skew;
    }
    return Rot;
}

/*!
* @brief Compute the logarithm of the SO(3) manifold
*
* From "A micro Lie theory for state estimation in robotics" Solà et al 
* Source https://arxiv.org/abs/1812.01537
*/
inline Eigen::Vector3d log_so3(const Eigen::Matrix3d &M) {
    double tolerance = 1e-9;
    double cos_angle = 0.5 * M.trace() - 0.5;

    // Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
    cos_angle    = std::min(std::max(cos_angle, -1.), 1.);
    double angle = acos(cos_angle);

    Eigen::Vector3d phi;
    // If angle is close to zero, use first-order Taylor expansion
    if (std::fabs(sin(angle)) < tolerance || angle < tolerance)
        phi = 0.5 * geometry::FromskewMatrix(M - M.transpose());
    else {
        // Otherwise take the matrix logarithm and return the rotation vector
        phi = (0.5 * angle / sin(angle)) * geometry::FromskewMatrix(M - M.transpose());
    }
    return phi;
}

/*!
* @brief Compute the logarithm of the SO(3)xT(3) composite manifold
*
* From "A micro Lie theory for state estimation in robotics" Solà et al 
* Source https://arxiv.org/abs/1812.01537
*/
inline Vector6d se3_RTtoVec6d(Eigen::Affine3d RT) {
    Vector6d pose;
    Eigen::Vector3d w = log_so3(RT.linear());
    pose(0)           = w(0);
    pose(1)           = w(1);
    pose(2)           = w(2);
    Eigen::Vector3d t = RT.translation();
    pose(3)           = t(0);
    pose(4)           = t(1);
    pose(5)           = t(2);
    return pose;
}

/*!
* @brief Compute the Exponential of the SO(3)xT(3) composite manifold
*
* From "A micro Lie theory for state estimation in robotics" Solà et al 
* Source https://arxiv.org/abs/1812.01537
*/
inline Eigen::Affine3d se3_Vec6dtoRT(Vector6d pose) {
    Eigen::Affine3d RT;
    Eigen::Vector3d w, t;
    w << pose(0), pose(1), pose(2);
    t << pose(3), pose(4), pose(5);
    RT.linear()      = exp_so3(w);
    RT.translation() = t;
    return RT;
}

/*!
* @brief Compute the Exponential of the SO(3)xT(3) composite manifold from a double*
*
* From "A micro Lie theory for state estimation in robotics" Solà et al 
* Source https://arxiv.org/abs/1812.01537
*/
inline Eigen::Affine3d se3_doubleVec6dtoRT(const double *pose) {
    Eigen::Affine3d RT;
    RT.linear() = exp_so3(Eigen::Vector3d(pose[0], pose[1], pose[2]));
    RT.translation() << pose[3], pose[4], pose[5];
    return RT;
}


/*!
* @brief Turns a point 3d in double* into an Affine3d with Id rotation
*/
inline Eigen::Affine3d se3_doubleVec3dtoRT(const double *p3d) {
    Eigen::Affine3d RT;
    RT.linear() = Eigen::Matrix3d::Identity();
    RT.translation() << p3d[0], p3d[1], p3d[2];
    return RT;
}

/*!
* @brief Compute the logarithm of the SO(3)xT(3) composite manifold from a double*
*
* From "A micro Lie theory for state estimation in robotics" Solà et al 
* Source https://arxiv.org/abs/1812.01537
*/
template <typename T> inline Eigen::Transform<T, 3, 4> se3_doubleVec6dtoRT(const T *pose) {
    Eigen::Transform<T, 3, 4> RT;
    // Eigen::Affine3d RT;
    Eigen::Matrix<T, 3, 1> w, t;
    // Eigen::Vector3d w, t;
    w << pose[0], pose[1], pose[2];
    t << pose[3], pose[4], pose[5];
    RT.linear()      = exp_so3(w);
    RT.translation() = t;
    return RT;
}

/*!
* @brief Get the angle formed by the segments p-p1 and p-p2
*/
inline double getAngle(Eigen::Vector3d p, Eigen::Vector3d p1, Eigen::Vector3d p2) {
    Eigen::Vector3d u1 = p - p1;
    Eigen::Vector3d u2 = p - p2;

    return std::acos(u1.dot(u2) / (u1.norm() * u2.norm()));
}

/*!
* @brief Check if a point is in a triangle with the barycentric technique
*
* Source https://blackpawn.com/texts/pointinpoly/
*/
template <typename Derived = Eigen::VectorXd> inline bool pointInTriangle(Derived pt, std::vector<Derived> triangle) {
    // Compute vectors
    Derived v0 = triangle.at(2) - triangle.at(0);
    Derived v1 = triangle.at(1) - triangle.at(0);
    Derived v2 = pt - triangle.at(0);

    // Compute dot products
    double dot00 = v0.dot(v0);
    double dot01 = v0.dot(v1);
    double dot02 = v0.dot(v2);
    double dot11 = v1.dot(v1);
    double dot12 = v1.dot(v2);

    // Compute barycentric coordinates
    double inv_denom = 1 / (dot00 * dot11 - dot01 * dot01);
    double u         = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    double v         = (dot00 * dot12 - dot01 * dot02) * inv_denom;

    // Check if point is in triangle
    return (u >= 0) && (v >= 0) && (u + v < 1);
}

/*!
* @brief Computes the covariance of a 2D triangle according to CGAL library 
*
* Source "Principal Component Analysis in CGAL" Gupta et al
*/
inline Eigen::Matrix2d cov2dTriangle(std::vector<Eigen::Vector2d> triangle) {

    // Compute transformation
    Eigen::Vector2d x0 = triangle.at(0);
    Eigen::Matrix2d At;
    At << triangle.at(1).x() - x0.x(), triangle.at(2).x() - x0.x(), triangle.at(1).y() - x0.y(),
        triangle.at(2).y() - x0.y();
    Eigen::Vector2d barycenter = (1.0 / 3.0) * (x0 + triangle.at(1) + triangle.at(2));

    // Primitive covariance
    Eigen::Matrix2d Mot;
    Mot << 1.0 / 12, 1.0 / 24, 1.0 / 24, 1.0 / 12;
    Eigen::Matrix2d M = 2 * At * Mot * At.transpose() +
                        (barycenter * x0.transpose() + x0 * barycenter.transpose() - x0 * x0.transpose()) -
                        barycenter * barycenter.transpose();

    return M;
}

/*!
* @brief Compute the area of a 2D triangle
*/
inline double areaTriangle(std::vector<Eigen::Vector2d> triangle) {

    Eigen::Vector2d x0 = triangle.at(0);
    Eigen::Matrix2d At;
    At << triangle.at(1).x() - x0.x(), triangle.at(2).x() - x0.x(), triangle.at(1).y() - x0.y(),
        triangle.at(2).y() - x0.y();

    return At.determinant() / 2;

}

inline Eigen::MatrixXd J_norm(Eigen::Vector3d X){
    return X.transpose()/X.norm();
}

inline Eigen::Matrix3d J_normalization(Eigen::Vector3d X){
    return (Eigen::Matrix3d::Identity() - X*X.transpose())/X.norm();
}

inline Eigen::Matrix3d J_XcrossA(Eigen::Vector3d A){
    return geometry::skewMatrix(A);
}

inline Eigen::Matrix3d J_AcrossX(Eigen::Vector3d A){
    return -geometry::skewMatrix(A);
}

inline Eigen::Matrix3d J_Rexpwt(Eigen::Matrix3d R, Eigen::Matrix3d expw, Eigen::Vector3d t){    
    return -R*expw*geometry::skewMatrix(t)*Eigen::Matrix3d::Identity()*geometry::so3_rightJacobian(geometry::log_so3(expw));
}


} // namespace geometry
} // namespace isae

#endif // GEOMETRY_H
