#ifndef POSEGRAPH_H
#define POSEGRAPH_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <tuple>
#include <vector>

namespace isae {

/*!
 * @brief Pose graph over the keyframes that left the window, for loop closure.
 *
 * Nodes are keyframe poses (frame to world). Odometry edges link each node to the previous nodes of its segment
 * (the relative poses of the window's final estimates); loop edges come from verified loop detections. With an IMU,
 * roll and pitch are observable, so the graph is 4-DoF (position and yaw per node, roll and pitch kept from the
 * odometry, as in VINS-Mono); without, it is 6-DoF.
 *
 * Segments (re-initializations) have unrelated world frames: no odometry edge links two segments, a loop edge may.
 * The gauge is fixed by the earliest node of each group of segments connected by loops.
 */
class PoseGraph {
  public:
    struct Sigmas {
        double odom_t   = 0.1;    //!< Odometry edges: translation (m)
        double odom_rot = 0.0175; //!< Odometry edges: rotation (rad)
        double loop_t   = 0.1;    //!< Loop edges: translation (m)
        double loop_rot = 0.0175; //!< Loop edges: rotation (rad)
    };

    PoseGraph(bool four_dof, int odom_neighbors = 4) : _four_dof(four_dof), _neighbors(odom_neighbors) {}
    void setSigmas(const Sigmas &s) { _sigmas = s; }

    /*!
     * @brief Add a keyframe with its odometry pose; its estimate starts from its segment's current correction
     * @return the node index
     */
    int addNode(const Eigen::Affine3d &T_w_f_odom, int segment);

    /*!
     * @brief Add a loop edge: relative pose T_fi_fj between nodes i and j (from a verified detection)
     */
    void addLoop(int i, int j, const Eigen::Affine3d &T_fi_fj);

    /*!
     * @brief Optimize all nodes; while the loop edge with the largest residual (whitened, squared) is above
     * reject_chi2, it is disabled and the graph is solved again (at most 5 solves)
     * @return the number of loop edges disabled
     */
    int optimize(double reject_chi2 = 20.0);

    size_t size() const { return _nodes.size(); }

    /*!
     * @brief Loop edges as (node i, node j, active)
     */
    std::vector<std::tuple<int, int, bool>> loops() const {
        std::vector<std::tuple<int, int, bool>> out;
        for (const auto &l : _loops)
            out.emplace_back(l.i, l.j, l.active);
        return out;
    }
    size_t nLoops() const { return _loops.size(); }
    size_t nActiveLoops() const;
    Eigen::Affine3d pose(int i) const;
    const Eigen::Affine3d &odomPose(int i) const { return _nodes[i].T_odom; }
    int segment(int i) const { return _nodes[i].segment; }

    /*!
     * @brief Correction world_corrected <- world_odometry of a segment, from its latest node (identity if none)
     */
    Eigen::Affine3d correction(int segment) const;

    /*!
     * @brief Group of a segment: the smallest segment joined to it by active loops (the segments of a group share the
     * pose graph's frame)
     */
    int group(int segment) const;

    /*!
     * @brief The odometry of a segment was moved by C (the sliding window was corrected): its odometry poses become
     * C T_odom, so the odometry edges are unchanged and the segment's correction becomes the identity
     */
    void reanchor(int segment, const Eigen::Affine3d &C);

    static double yawOf(const Eigen::Matrix3d &R) { return std::atan2(R(1, 0), R(0, 0)); }

  private:
    struct Node {
        Eigen::Affine3d T_odom;
        int segment;
        double t[3];
        double yaw;          // 4-DoF
        Eigen::Matrix3d R_rp; // 4-DoF: roll and pitch part of the odometry rotation (R = Rz(yaw) R_rp)
        double q[4];         // 6-DoF: x, y, z, w
    };
    struct Loop {
        int i, j;
        Eigen::Affine3d T_ij;
        bool active = true;
    };

    void setEstimate(Node &n, const Eigen::Affine3d &T) const;

    bool _four_dof;
    int _neighbors;
    Sigmas _sigmas;
    std::vector<Node> _nodes;
    std::vector<Loop> _loops;
};

} // namespace isae

#endif // POSEGRAPH_H
