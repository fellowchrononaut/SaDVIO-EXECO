#ifndef SLAMCORE_H
#define SLAMCORE_H

#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/maps/globalmap.h"
#include "isaeslam/data/maps/localmap.h"
#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/mesh/mesher.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/data/sensors/DoubleSphere.h"
#include "isaeslam/dataproviders/adataprovider.h"
#include "isaeslam/estimator/ESKFEstimator.h"
#include "isaeslam/estimator/EpipolarPoseEstimator.h"
#include "isaeslam/estimator/PnPPoseEstimator.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvORBFeatureDetector.h"
#include "isaeslam/featurematchers/Point2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Point2DFeatureTracker.h"
#include "isaeslam/landmarkinitializer/Point3DlandmarkInitializer.h"
#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/slamParameters.h"
#include "isaeslam/stereo/MarginalDepthInjector.h"
#include "isaeslam/typedefs.h"
#include "utilities/timer.h"

namespace isae {

/*!
 * @brief The core abstract class of the SLAM system. It handles front-end, back-end and initialization.
 *
 * This is the skeleton of the SLAM system that will call all the subsystems (sensor processing, optimizer, map
 * processing...) It loads all the parameters of the SLAM and implements frontEnd and backEnds that must be called in a
 * separated thread. A few important methods relative to feature processing and pose estimation are implemented as well
 * to make the frontEndStep() and backEndStep() methods readable. Also, a few profiling variables and methods are
 * implemented to monitor the performances.
 */
class SLAMCore {
  public:
    SLAMCore() {};
    SLAMCore(std::shared_ptr<isae::SLAMParameters> slam_param);

    /*!
     * @brief Initialization step : create the first 3D landmarks and keyframe(s)
     */
    virtual bool init() = 0;

    /*!
     * @brief Front End: detection, tracking, pose estimation and landmark triangulation
     */
    virtual bool frontEndStep() = 0;

    /*!
     * @brief brief description Back End: marginalization, local map optimization
     */
    virtual bool backEndStep() = 0;

    /*!
     * @brief Thread for the backend
     */
    void runBackEnd();

    /*!
     * @brief Thread for the frontend
     */
    void runFrontEnd();

    /*!
     * @brief Thread for the backend
     */
    void runFullOdom();

    bool _is_init         = false; //!< Flag for initialization
    int _successive_fails = 0;     //!< Number of successive failure to trigger reinitialization

    // Public variables for display
    std::shared_ptr<isae::SLAMParameters> _slam_param;
    std::shared_ptr<Frame> _frame_to_display;
    std::shared_ptr<isae::LocalMap> _local_map_to_display;
    std::shared_ptr<isae::GlobalMap> _global_map_to_display;
    std::shared_ptr<Mesh3D> _mesh_to_display;

    // Dense stereo injector — null for non-stereo modes, initialised in BiMono/BiMonoVIO::init()
    std::shared_ptr<MarginalDepthInjector> _depth_injector;

    /*!
     * @brief Detect all types of features for a given sensor with bucketting
     */
    typed_vec_features detectFeatures(std::shared_ptr<ImageSensor> &sensor);

    /*!
     * @brief Clean all the features that are outliers or are linked to outlier landmark
     */
    void cleanFeatures(std::shared_ptr<Frame> &f);

    /*!
     * @brief Computes the velocity of the features in the frame, it is used to estimate the time delay between IMU and
     * cameras
     */
    void computeFeatureVelocity(typed_vec_match &matches);

    /*!
     * @brief Predicts the position of the features and matches all the features between sensors
     *
     * @param sensor0 First sensor
     * @param sensor1 Second sensor
     * @param matches Typed vector of matches without landmarks passed as reference
     * @param matches_lmk Typed vector of matches with landmarks passed as reference
     * @param features_to_track Typed vector of the features on sensor0 that will be matched for prediction
     *
     * @return The number of tracks
     */
    uint matchFeatures(std::shared_ptr<ImageSensor> &sensor0,
                       std::shared_ptr<ImageSensor> &sensor1,
                       typed_vec_match &matches,
                       typed_vec_match &matches_lmk,
                       typed_vec_features features_to_track);

    /*!
     * @brief Predicts the position of the features and tracks all the features
     *
     * @param sensor0 First sensor
     * @param sensor1 Second sensor
     * @param matches Typed vector of matches without landmarks passed as reference
     * @param matches_lmk Typed vector of matches with landmarks passed as reference
     * @param features_to_track Typed vector of the features on sensor0 that will be tracked for prediction
     *
     * @return The number of tracks
     */
    uint trackFeatures(std::shared_ptr<ImageSensor> &sensor0,
                       std::shared_ptr<ImageSensor> &sensor1,
                       typed_vec_match &matches,
                       typed_vec_match &matches_lmk,
                       typed_vec_features features_to_track);

    /*!
     * @brief Predicts the position of a set of features on a given sensor
     *
     * @param features Set of features to be predicted
     * @param sensor Sensor on which the features has to be predicted
     * @param features_init The set of features predicted passed as a reference
     * @param previous_matches A prior on features_init that can eventually be used default is an empty set
     */
    void predictFeature(std::vector<std::shared_ptr<AFeature>> features,
                        std::shared_ptr<ImageSensor> sensor,
                        std::vector<std::shared_ptr<AFeature>> &features_init,
                        vec_match previous_matches);

    /*!
     * @brief Filters the matches between two sensors (with consistent pose estimates) using epipolar plane check
     *
     * @param cam0 First sensor
     * @param cam1 Second sensor
     * @param matches Typed vector of matches to be filtered
     *
     * @return A Typed vector of the valid matches
     */
    typed_vec_match
    epipolarFiltering(std::shared_ptr<ImageSensor> &cam0, std::shared_ptr<ImageSensor> &cam1, typed_vec_match matches);

    /*!
     * @brief Remove all the outlier features of _frame
     */
    void outlierRemoval();

    /*!
     * @brief Estimates the pose of a given frame
     *
     * @param f The frame whose pose is estimated passed as a reference
     *
     * @return True if the prediction is a success, False otherwise
     */
    bool predict(std::shared_ptr<Frame> &f);

    /*!
     * @brief Initialize all the landmarks of the current frame using _matches_in_time and _matches_in_frame
     */
    void initLandmarks(std::shared_ptr<Frame> &f);
    void updateLandmarks(typed_vec_match matches_lmk);

    /*!
     * @brief Performs landmark resurection
     *
     * @param sensor Sensor on which the landmarks are resurected
     *
     * @return The number of resurected landmarks
     */
    uint recoverFeatureFromMapLandmarks(std::shared_ptr<ImageSensor> &sensor);

    /*!
     * @brief Determines if the frame in argument is a KF
     */
    bool shouldInsertKeyframe(std::shared_ptr<Frame> &f);
    std::shared_ptr<Frame> getLastKF() { return _local_map->getLastFrame(); }

    /*!
     * @brief A function to monitor the SLAM behaviour
     */
    void profiling();

    /*!
     * @brief Make kf's preintegration start at kf_prev (integrated again over the longer interval), before the
     * KF in between is removed from the window; without kf_prev the IMU link is cut.
     */
    void bridgePreintegration(const std::shared_ptr<Frame> &kf, const std::shared_ptr<Frame> &kf_prev);

    /*!
     * @brief True if a low-parallax KF can be dropped: the bridged preintegration (kf_prev -> kf) must still be
     * usable as a factor. Otherwise (e.g. a long standstill, where every new KF has low parallax) the bridged
     * interval grew past the factor limit and the newest KFs lost their IMU link; the oldest KF is marginalized
     * instead.
     */
    bool canBridge(const std::shared_ptr<Frame> &kf, const std::shared_ptr<Frame> &kf_prev) const;

    /*!
     * @brief True if the relative pose T_new (last KF -> frame) is a plausible update of T_ref: their difference
     * (rotation [rad] and translation [m] stacked) is at most 10 times the motion of T_ref, with a floor of 0.1 on
     * that motion (i.e. at least 1.0 is always allowed)
     */
    static bool plausibleUpdate(const Eigen::Affine3d &T_ref, const Eigen::Affine3d &T_new);

    /*!
     * @brief Consecutive frames without visual pose estimate after which the SLAM re-initializes: config
     * max_lost_frames, or the mode default. VO: 5. Stereo VIO: 20 (~1 s at 20 Hz; the IMU carries the pose through
     * short visual dropouts such as motion blur, where a reset threw the whole state away). Mono VIO: 10 (its velocity
     * and scale are less certain; a longer IMU-only stretch gave no measured benefit).
     */
    int maxLostFrames() const {
        if (_slam_param->_config.max_lost_frames > 0)
            return _slam_param->_config.max_lost_frames;
        const std::string &mode = _slam_param->_config.slam_mode;
        return mode == "bimonovio" ? 20 : mode == "monovio" ? 10 : 5;
    }

    /*!
     * @brief Release the IMU measurements older than a KF leaving the window: every IMU sample holds the previous one,
     * so the whole history of the run stayed in memory. Nothing walks the chain past the oldest window KF.
     */
    static void releaseImuHistory(const std::shared_ptr<Frame> &kf) {
        if (kf && kf->getIMU())
            kf->getIMU()->setLastIMU(nullptr);
    }

    /*!
     * @brief Before a re-initialization after a visual dropout (VIO): keep the state of imu's frame (pose, velocity,
     * biases) for the next initialization, if reinit_carry_state is set and the state is usable (see
     * CarriedImuState); otherwise the next initialization starts from scratch
     */
    void captureCarriedState(const std::shared_ptr<IMU> &imu);

    /*!
     * @brief Write the KFs of the window not written yet to log_slam/results.csv, before a re-initialization
     * discards them (they belong to the segment that ends)
     */
    void logWindowBeforeReset();

    /*!
     * @brief Check the result of the inertial initialization: usable solution (scale > 0), finite states and
     * plausible biases (|ba| < 2 m/s^2, |bg| < 0.5 rad/s, generous bounds for MEMS IMUs).
     */
    bool inertialInitAccepted(double scale);

    /*!
     * @brief Append one row of visual-inertial diagnostics for a keyframe to log_slam/vio_diag.csv
     */
    void logVIODiag(const std::shared_ptr<Frame> &kf, const VIOptimStats &stats);

  protected:
    std::shared_ptr<Frame> _frame; //!< Current frame

    // Typed vector for matches
    typed_vec_match _matches_in_time;     //!< Typed vector of the matches between the last KF and _frame
    typed_vec_match _matches_in_time_lmk; //!< Typed vector of the matches with landmarks between the last KF and _frame
    typed_vec_match _matches_in_frame;    //!< Typed vector of the matches between the sensors of _frame
    typed_vec_match _matches_in_frame_lmk; //!< Typed vector of the matches with landmarks between the sensors of _frame

    // Local Map, Mesh and Keyframe voting policy
    std::shared_ptr<isae::LocalMap> _local_map;   //!< Current local map
    std::shared_ptr<isae::GlobalMap> _global_map; //!< Current global map
    std::shared_ptr<Mesher> _mesher;              //!< Mesher of the SLAM
    double _max_movement_parallax;                //!< Max parallax until a KF is voted
    double _min_movement_parallax;                //!< Under this parallax, no motion is considered
    double _min_lmk_number;                       //!< Under this number of landmark in _frame a KF is voted
    double _parallax;                             //!< Parallax between the last KF and _frame
    double _parallax_to_optim = 0;                //!< Parallax of _frame_to_optim (handed to the back end with it)
    Vector6d _6d_velocity;                        //!< Current Velocity as a Twist vector

    // To ensure safe communication between threads
    std::mutex _map_mutex;
    std::shared_ptr<Frame> _frame_to_optim; //!< For communication between front-end and back-end

    // Multithreading: the front end holds _step_mutex during the processing of a frame and the back end during the
    // processing of a KF, so they never modify the maps at the same time. The front end releases it while it waits
    // for data or for the back end (see nextFrame() and waitBackEnd())
    std::mutex _step_mutex;
    std::condition_variable _step_cv;
    std::unique_lock<std::mutex> *_frontend_lock = nullptr; //!< Lock of the front end thread, null in single thread

    /*!
     * @brief Next frame of the data provider; in multithreading the back end can run while the front end waits,
     * and the frame is returned once the back end has taken the pending KF into the local map
     */
    std::shared_ptr<Frame> nextFrame();

    /*!
     * @brief Wait until the back end has processed _frame_to_optim (no-op in single thread, where it already did)
     */
    void waitBackEnd();

    // VIO diagnostics
    uint _n_rejected_imu   = 0;     //!< IMU samples rejected by processIMU() (e.g. out of order)
    bool _vio_diag_started = false; //!< Header of log_slam/vio_diag.csv written
    bool _kf_votes_started = false; //!< Header of log_slam/kf_votes.csv written
    bool _results_started  = false; //!< log_slam/results.csv and cov_mat.csv created (once per run)
    int _segment           = 0;     //!< Re-initializations so far: the segment column of results.csv (each segment has
                                    //!< its own world frame after a re-initialization)
    bool _segment_has_rows = false; //!< Rows of the current segment written (the init path calls profiling() twice)
    unsigned long long _last_logged_ts = 0; //!< Last KF written to results.csv: each KF is written once
    std::shared_ptr<Frame> _results_front;  //!< Oldest KF of the window, written when it leaves (final estimate)

    /*!
     * @brief Append the pose of KF f to log_slam/results.csv, unless it is not newer than the last KF written
     */
    void writeResultRow(const std::shared_ptr<Frame> &f);

    // Re-initialization from the last state
    CarriedImuState _carried;              //!< State kept for the next initialization (VIO), dead-reckoned meanwhile
    unsigned long long _last_visual_ts = 0; //!< Last frame whose pose came from the camera(s) (VIO)

    /*!
     * @brief Append the KF vote of a tracked frame to log_slam/kf_votes.csv (diagnostics)
     */
    void logKfVote(const std::shared_ptr<Frame> &f, double n_matches, double n_matches_lmk, const char *reason);

    /*!
     * @brief If EXECO_KF_FEATURES_LOG is set, append the point features of a KF handed to the back end to
     * log_slam/kf_features.csv (pixel position in camera 0 and landmark status, as in the ROS image_kps view)
     */
    void logKfFeatures(const std::shared_ptr<Frame> &kf);
    bool _kf_features_started = false; //!< Header of log_slam/kf_features.csv written

    // Profiling variables
    uint _nframes;
    uint _nkeyframes;
    float _avg_detect_t;
    float _avg_processing_t;
    float _avg_match_frame_t;
    float _avg_match_time_t;
    float _avg_filter_t;
    float _avg_lmk_init_t;
    float _avg_lmk_resur_t;
    float _avg_predict_t;
    float _avg_frame_opt_t;
    float _avg_clean_t;
    float _avg_marg_t;
    float _avg_wdw_opt_t;
    float _removed_lmk;
    float _removed_feat;
    float _lmk_inmap;
    float _avg_matches_time;
    float _avg_matches_frame;
    float _avg_resur_lmk;

    // For timing statistics
    std::vector<std::vector<float>> _timings_frate;
    std::vector<std::vector<float>> _timings_kfrate_fe;
    std::vector<std::vector<float>> _timings_kfrate_be;
};

/*!
 * @brief A SLAM class for bi-monocular setups
 */
class SLAMBiMono : public SLAMCore {

  public:
    SLAMBiMono(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

  private:
    // Re-initialization from the last pose (reinit_carry_state): without an IMU, the pose of the last tracked frame
    // is extrapolated with its velocity (the constant-velocity model of the tracking) to the first frame of the new map
    Eigen::Affine3d _last_tracked_T_w_f = Eigen::Affine3d::Identity(); //!< Pose of the last tracked frame
    Vector6d _last_tracked_velocity     = Vector6d::Zero();            //!< Its velocity (twist per second)
    unsigned long long _last_tracked_ts = 0;                           //!< Its time (0: none)
    bool _carry_pose                    = false; //!< The next initialization starts from the extrapolated pose
};

/*!
 * @brief A SLAM class for bi-monocular + IMU setups
 */
class SLAMBiMonoVIO : public SLAMCore {

  public:
    SLAMBiMonoVIO(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

    /*!
     * @brief Initialization steps with bi-monocular only for IMU initialization
     */
    bool step_init();

    /*!
     * @brief For profiling at IMU rate
     */
    void IMUprofiling();

  private:
    std::shared_ptr<IMU> _last_IMU; //!< Last IMU processed by the SLAM
};

/*!
 * @brief A SLAM class for monocular + IMU setups
 */
class SLAMMonoVIO : public SLAMCore {

  public:
    SLAMMonoVIO(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

    /*!
     * @brief Initialization steps with bi-monocular only for IMU initialization
     */
    bool step_init();

    /*!
     * @brief For profiling at IMU rate
     */
    void IMUprofiling();

  private:
    std::shared_ptr<IMU> _last_IMU; //!< Last IMU processed by the SLAM
};

/*!
 * @brief A SLAM class for monocular setups
 */
class SLAMMono : public SLAMCore {

  public:
    SLAMMono(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

};

/*!
 * @brief A SLAM class for non overlapping FoV setups
 */
class SLAMNonOverlappingFov : public SLAMCore {

  public:
    SLAMNonOverlappingFov() {};
    SLAMNonOverlappingFov(std::shared_ptr<SLAMParameters> slam_param) : SLAMCore(slam_param) {}

    bool init() override;
    bool frontEndStep() override;
    bool backEndStep() override;

    /*!
     * @brief To remove outliers on both cameras
     */
    void outlierRemoval();

    /*!
     * @brief To init landmarks on both cameras
     */
    void initLandmarks(std::shared_ptr<Frame> &f);

    /*!
     * @brief To compute the scale using a single point in a RANSAC fashion
     *
     * @param T_cam0_cam1 Extrinsic between cameras
     * @param T_cam0_cam0p Up to scale estimation of motion of cam0
     * @param matches_cam1 Matches in time of cam1
     * @param lambda Scale estimate passed as reference
     *
     * @return Number of inliers of the RANSAC
     */
    int scaleEstimationRANSAC(const Eigen::Affine3d T_cam0_cam1,
                              const Eigen::Affine3d T_cam0_cam0p,
                              typed_vec_match matches_cam1,
                              double &lambda);

    /*!
     * @brief To check if the scale can't be recovered
     */
    bool isDegenerativeMotion(Eigen::Affine3d T_cam0_cam0p, Eigen::Affine3d T_cam0_cam1, typed_vec_match matches);

    /*!
     * @brief Prediction that takes into account both camera motions
     */
    bool predict(std::shared_ptr<Frame> &f);

  private:
    typed_vec_match _matches_in_time_cam1; //!< Typed vector of the matches between the last KF and _frame on cam1
    typed_vec_match
        _matches_in_time_cam1_lmk; //!< Typed vector of the matches with lmks between the last KF and _frame on cam1
};

} // namespace isae

#endif // SLAMCORE_H