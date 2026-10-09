#ifndef LOOPCLOSURE_H
#define LOOPCLOSURE_H

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "DBoW3.h"
#include "isaeslam/loopclosure/GlobalDescriptor.h"
#include "isaeslam/loopclosure/PoseGraph.h"

namespace isae {

class Frame;
class ImageSensor;

/*!
 * @brief Loop closure over the keyframes that left the sliding window (option 1: separate pose graph).
 *
 * Each keyframe is handed over once, with its final window estimate, when it leaves the window. The loop closure
 * keeps a compact record per keyframe (no image): 1000 pyramid ORB of camera 0 as rays with a bag-of-words vector,
 * and camera 0's landmarks (positions in camera 0, oriented ORB computed at their pixels: the front end's
 * descriptors have no orientation and do not match oriented ORB, Hamming ~121 of 256, chance level).
 *
 * For a new keyframe: candidates from the detector (bag-of-words, proximity gate, or both), then verification: the
 * new keyframe's landmarks are matched to a candidate's ORB and its pose is found by PnP RANSAC, and the same the
 * other way round; the two relative poses must agree. A verified loop is added to the pose graph, which is solved
 * at once. The sliding window is left untouched: the pose graph's correction is applied to the output only
 * (log_slam/results_loop*.csv).
 *
 * The phase-0 benchmark (doc/loop_closure) chose 1000 pyramid ORB: the ORB-SLAM vocabulary matches them much
 * better than the front end's single-scale FAST corners (R@1 0.74 against 0.47).
 */
class LoopClosure {
  public:
    struct Options {
        std::string detector   = "bow"; //!< bow | proximity | proximity_bow | learned
        std::string vocabulary;          //!< DBoW3 vocabulary (.dbow3, or ORB-SLAM's ORBvoc.txt)
        std::string model;               //!< learned: global descriptor model (.onnx for CPU, TensorRT engine for GPU)
        std::string model_device = "CPU"; //!< learned: CPU (OpenVINO) or GPU (TensorRT)
        int model_threads      = 8;     //!< learned, CPU: inference threads (0: OpenVINO's choice)
        double min_gap         = 20.0;  //!< Candidates are at least this much older than the query (s)
        int min_inliers        = 12;    //!< PnP inliers to accept a direction of the verification
        double gate_radius     = 2.0;   //!< Proximity gate around the query's corrected position (m)
        double gate_angle      = 45.0;  //!< Proximity gate: viewing directions within this angle (deg)
        int n_candidates       = 3;     //!< Candidates verified per keyframe (at least min_gap apart in time)
        int max_loops_per_kf   = 2;     //!< Loops accepted per keyframe (at least min_gap apart in time)
        int n_orb              = 1000;  //!< Pyramid ORB extracted per keyframe
        bool four_dof          = false; //!< 4-DoF pose graph (roll and pitch kept from the odometry), else 6-DoF
        bool async             = false; //!< Process the keyframes in a thread of their own
        bool correct_window    = false; //!< After a loop, the SLAM moves its sliding window by the correction
    };

    struct Stats {
        int n_keyframes = 0, n_candidates = 0, n_verified = 0, n_loops = 0, n_rejected_by_graph = 0;
        double avg_ms_keyframe = 0, avg_ms_optimize = 0, avg_ms_describe = 0;
    };

    explicit LoopClosure(const Options &opt);
    ~LoopClosure();

    /*!
     * @brief Hand over a keyframe leaving the window (its final estimate). Copies what the loop closure keeps; in
     * async mode the image is processed later in the loop thread
     */
    void addKeyframe(const std::shared_ptr<Frame> &f, int segment);

    /*!
     * @brief Pose (frame to world) corrected by the current pose graph, for a pose of the given segment
     */
    Eigen::Affine3d correct(const Eigen::Affine3d &T_w_f, int segment) const;

    Stats stats() const;

    /*!
     * @brief With correct_window: if a loop was closed since the last call, the current correction of the segment
     * (corrected world <- odometry world), to apply to the sliding window. The pose graph then takes the segment's
     * odometry as moved by it (and so do keyframes still queued)
     * @return false if there is nothing to apply
     */
    bool takeCorrection(int segment, Eigen::Affine3d &C);

    //! What a viewer shows: the loop-closed trajectory (consecutive KFs of a segment as line segments) and the loops
    struct Display {
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> traj;      //!< corrected trajectory, as segments
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> loops;     //!< accepted loops (corrected positions)
        std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> new_loops; //!< loops of the latest loop KF
        unsigned long version = 0;                                          //!< increases with every KF processed
    };

    /*!
     * @brief Fill a display snapshot if anything changed since version known
     * @return false if nothing changed
     */
    bool display(Display &out, unsigned long known) const;

  private:
    friend struct LoopClosureTestAccess; // unit tests of the verification (cpp/tests/loopclosure_test.cpp)

    struct Input {
        unsigned long long ts;
        int segment;
        Eigen::Affine3d T_w_f;              // odometry (window) estimate, frame to world
        Eigen::Affine3d T_s_f;              // frame to camera 0
        std::shared_ptr<ImageSensor> cam;   // held until processed, for the image and the camera model
        std::vector<Eigen::Vector3d> lmk_c; // landmarks seen by camera 0, in camera 0
        std::vector<cv::Point2f> lmk_px;    // their pixels in camera 0's image
    };
    struct Record {
        unsigned long long ts;
        int segment;
        int node;                            // pose graph node
        Eigen::Affine3d T_s_f;
        std::vector<cv::Point2f> kp_n;       // ORB as normalized coordinates (ray x/z, y/z), camera 0
        cv::Mat desc;                        // ORB descriptors (rows aligned with kp_n)
        DBoW3::BowVector bow;
        Eigen::VectorXf gdesc;               // learned global descriptor (detector learned)
        std::vector<Eigen::Vector3d> lmk_c;  // landmarks in camera 0
        cv::Mat lmk_desc;                    // oriented ORB at their pixels (rows aligned with lmk_c)
        double focal;
    };

    void process(Input &in);
    std::vector<int> candidates(const Record &q);
    bool verify(const Record &q, const Record &c, Eigen::Affine3d &T_fc_fq, int &inl_q, int &inl_c);
    bool pnp(const std::vector<Eigen::Vector3d> &pts, const cv::Mat &pts_desc, const Record &target,
             Eigen::Affine3d &T_t_src, int &inliers, int &matches) const;
    void writeTrajectory();
    void loop();

    Options _opt;
    DBoW3::Vocabulary _voc;
    DBoW3::Database _db;
    cv::Ptr<cv::ORB> _orb;
    std::unique_ptr<GlobalDescriptor> _vpr;
    PoseGraph _graph;
    std::vector<Record> _records;
    std::map<int, std::pair<int, int>> _online_label;
    bool _pending_correction = false; // a loop was closed: the window should move (correct_window)
    unsigned long _version   = 0;     // KFs processed (display)
    std::vector<std::pair<int, int>> _last_loops; // node pairs of the latest loop KF (display) // segment -> (group seen last, frame changes): online labels

    mutable std::mutex _graph_mutex; // _graph, read by correct() from the SLAM threads
    Stats _stats;

    // Async mode
    std::mutex _queue_mutex;
    std::condition_variable _queue_cv;
    std::deque<Input> _queue;
    std::atomic<bool> _stop{false};
    std::thread _thread;
};

} // namespace isae

#endif // LOOPCLOSURE_H
