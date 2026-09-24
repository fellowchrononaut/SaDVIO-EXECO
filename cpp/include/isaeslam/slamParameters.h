#ifndef SLAMPARAMETERS_H
#define SLAMPARAMETERS_H

#include "isaeslam/optimizers/AOptimizer.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

namespace isae {

class AFeatureDetector;
class AFeatureMatcher;
class AFeatureTracker;
class ADataProvider;
class APoseEstimator;
class ALandmarkInitializer;
class BundleAdjustmentCERES;
class LocalMap;

/*!
 * @brief A struct that contains a feature matcher and its parameters
 */
struct FeatureMatcherStruct {
    int matcher_width;
    int matcher_height;
    std::shared_ptr<AFeatureMatcher> feature_matcher;
};

/*!
 * @brief A struct that contains a feature tracker and its parameters
 */
struct FeatureTrackerStruct {
    int tracker_width;
    int tracker_height;
    int tracker_nlvls_pyramids;
    double tracker_max_err;
    std::shared_ptr<AFeatureTracker> feature_tracker;
};

/*!
 * @brief A struct that gathers all the parameters for a feature
 */
struct FeatureStruct {
    std::string label_feature;    //!< Label of the feature
    std::string detector_label;   //!< label of the feature detector
    int number_detected_features; //!< number of features to be detected by the detector
    int n_features_per_cell;      //!< number of features per cell for bucketting
    std::string tracker_label;    //!< class name of the tracker we will use in our SLAM
    int tracker_height;           //!< searchAreaHeight of tracker
    int tracker_width;            //!< searchAreaWidth of tracker
    int tracker_nlvls_pyramids;   //!< nlevels of pyramids for klt tracking
    double tracker_max_err;       //!< error threshold for klt tracking
    std::string matcher_label;    //!< class name of the matcher we will use in our SLAM
    double max_matching_dist;     //!< distance for matching
    int matcher_height;           //!< searchAreaHeight of tracker
    int matcher_width;            //!< searchAreaWidth of tracker
    std::string
        lmk_triangulator; //!< landmarkTriangulation class we will use to triangulate landmark of label_feature type
};

/*!
 * @brief This structure contains the configuration parameters located in the config file.
 */
struct Config {
    std::string dataset_path;    //!< Path to the dataset
    std::string dataset_id;      //!< Id of the dataset
    std::string slam_mode;       //!< SLAM mode (mono, bimono, monovio...)
    bool multithreading;         //!< Allow to run front-end and back-end on different threads (unstable...)
    bool enable_visu;            //!< Allow visualization
    bool estimate_td;            //!< Estimate time delay between IMU and cameras
    std::string optimizer;       //!< Optimizer type (ReprojectionError, AngularError...)
    int contrast_enhancer;       //!< integer to choose the contrast enhancement algorithm
    float clahe_clip;            //!< Clip of CLAHE (useful only if it is chosen for contrast enhancement)
    float downsampling;          //!< Float to reduce the size of the image (0,5 = half the size of the img)
    int marginalization;         //!< 0 no marginalization, 1 marginalization
    bool sparsification;         //!< 0 no sparsification, 1 sparsification
    std::string pose_estimator;  //!< Type of pose estimator
    std::string tracker;         //!< Type of tracking (matcher or klt)
    int min_kf_number;           //!< Minimum KF for optimization
    int max_kf_number;           //!< Size maximum of the sliding windown
    int fixed_frame_number;      //!< Number of fixed frame for gauge fixing
    float min_lmk_number;        //!< Below this number of landmark, a KF is voted
    float min_movement_parallax; //!< Below this parallax, no motion is considered
    float max_movement_parallax; //!< Over this parallax, a KF is voted
    bool mesh3D;                 //!< 0 no 3D mesh, 1 3D mesh
    double ZNCC_tsh;             //!< Threshold on ZNCC for triangle filtering
    double max_length_tsh;       //!< Threshold on maximum length for triangle filtering

    // Dense SGBM mesh pipeline — fully independent of mesh3d (sparse ZNCC mesh)
    bool        dense_depth          = false;         // enable dense SGBM pipeline
    std::string dense_mesh_method   = "none";        // "none"=depth only, "gp"=GP, "pd"=PD, "zncc"=SGBM+Delaunay+ZNCC
    double      stereo_depth_scale     = 1.0;        // SGBM image scale factor
    double      stereo_depth_max_depth = 20.0;       // metres; dense cloud points beyond this are dropped
    int         stereo_depth_num_disp   = 64;
    int         stereo_depth_block_size = 5;
    int         stereo_depth_stride     = 2;
    int         stereo_depth_uniqueness_ratio      = 10;
    int         stereo_depth_speckle_window_size   = 100;
    int         stereo_depth_speckle_range         = 32;
    int         stereo_depth_disp12_max_diff       = 1;
    int         stereo_depth_pre_filter_cap        = 0;
    bool        stereo_depth_publish_sgbm_images   = false;
    std::string stereo_depth_matcher      = "sgbm";  // "sgbm" or "ffs" (Fast-FoundationStereo TensorRT engine)
    std::string stereo_depth_ffs_engine;              // TensorRT engine path for "ffs"
    double      stereo_depth_ffs_lr_check = 0.0;      // "ffs" left-right check threshold in px (0 = off)

    // Dense GP mesh controls, matching the SLAMesh local GP reconstruction.
    double      dense_gp_cell_size        = 1.0;
    int         dense_gp_num_test         = 6;
    int         dense_gp_min_pts_per_cell = 8;
    double      dense_gp_kernel_length    = 1.2;
    double      dense_gp_variance_sensor  = 0.1;
    double      dense_gp_max_variance     = 0.5;
    bool        dense_gp_full_cover       = false;
    double      dense_gp_eigen_1          = 48.0;
    double      dense_gp_eigen_2          = 0.95;
    double      dense_gp_eigen_3          = 0.2;
    bool        dense_gp_stitch_seams            = true;
    double      dense_gp_seam_max_variance       = 0.5;
    double      dense_gp_seam_max_edge_length    = 0.5;
    double      dense_gp_seam_max_prediction_gap = 0.25;
    double      dense_gp_seam_min_normal_cos     = 0.5;

    // Global GP map: SLAMesh map update (A) and optional frame-to-model registration (B).
    bool        dense_keep_all_keyframes           = false;   // queue every marginalised keyframe instead of only the latest
    bool        dense_gp_global_map                = false;   // A: fuse keyframes into one GP map (SLAMesh map update)
    double      dense_gp_variance_map_update       = 0.5;     // SLAMesh variance_map_update
    int         dense_gp_max_raw_points_per_cell   = 2000;    // raw points kept per non-surface cell
    bool        dense_gp_register                  = false;   // B: SLAMesh frame-to-model registration (needs dense_gp_global_map)
    int         dense_gp_register_times            = 5;       // SLAMesh register_times
    double      dense_gp_variance_register         = 0.1;     // SLAMesh variance_register
    int         dense_gp_cross_cell_overlap_length = 1;       // SLAMesh cross_cell_overlap_length
    double      dense_gp_register_converge_thr     = 1e-5;    // SLAMesh converge_thr
    double      dense_gp_register_huber            = 0.1;     // SLAMesh Huber loss delta
    int         dense_gp_register_min_matches      = 30;      // keep the VO pose below this many pairs
    double      dense_gp_register_max_translation  = 0.5;     // metres; reject larger corrections
    double      dense_gp_register_max_rotation_deg = 10.0;    // degrees; reject larger corrections
    bool        dense_gp_register_carry_correction = true;    // start each keyframe from the last accepted correction
    bool        dense_gp_register_depth_weighting  = false;   // weight residuals by min(1, (ref/z)^2)
    double      dense_gp_register_depth_ref        = 2.0;     // metres
    std::string dense_gp_global_mesh_path          = "log_slam/dense_gp_global_mesh.ply"; // PLY written every dense_gp_save_every keyframes
    int         dense_gp_save_every                = 5;       // keyframes between PLY saves (0 = never)

    // Dense primal-dual mesh controls from the inverse-depth optimization paper.
    int         dense_pd_steiner_spacing = 20;
    double      dense_pd_lambda          = 0.5;
    int         dense_pd_num_iterations  = 120;
    double      dense_pd_tau             = 0.01;
    double      dense_pd_sigma           = 0.1;
    double      dense_pd_theta           = 1.0;
    double      dense_pd_min_depth       = 0.5;

    std::vector<FeatureStruct> features_handled; //!< types of features the slam will work on separated with commas (,)
};

/*!
 * @brief A class that gathers most of the algorithmic blocks of the SLAM system that can be setup in the config file
 *
 * Some attributes are sets as unordered map because these depends on the feature type (e.g. matcher, detector...). Then
 * the proper blocks can be called using the feature label.
 */
class SLAMParameters {
  public:
    SLAMParameters(const std::string config_file);

    std::shared_ptr<ADataProvider> getDataProvider() { return _data_provider; }
    std::unordered_map<std::string, std::shared_ptr<AFeatureDetector>> getFeatureDetectors() { return _detector_map; }
    std::unordered_map<std::string, FeatureTrackerStruct> getFeatureTrackers() { return _tracker_map; }
    std::unordered_map<std::string, FeatureMatcherStruct> getFeatureMatchers() { return _matcher_map; }
    std::unordered_map<std::string, std::shared_ptr<ALandmarkInitializer>> getLandmarksInitializer() {
        return _lmk_init_map;
    };

    std::shared_ptr<APoseEstimator> getPoseEstimator() { return _pose_estimator; }
    std::shared_ptr<AOptimizer> getOptimizerFront() { return _optimizer_frontend; }
    std::shared_ptr<AOptimizer> getOptimizerBack() { return _optimizer_backend; }
    void readConfigFile(const std::string &path_config_folder);
    Config _config;

  private:
    std::shared_ptr<ADataProvider> _data_provider;
    std::unordered_map<std::string, std::shared_ptr<AFeatureDetector>> _detector_map;
    std::unordered_map<std::string, FeatureTrackerStruct> _tracker_map;
    std::unordered_map<std::string, FeatureMatcherStruct> _matcher_map;
    std::unordered_map<std::string, std::shared_ptr<ALandmarkInitializer>> _lmk_init_map;

    std::shared_ptr<APoseEstimator> _pose_estimator;
    std::shared_ptr<AOptimizer> _optimizer_frontend, _optimizer_backend;
    std::shared_ptr<LocalMap> _local_map;

    void createProvider();
    void createDetectors();
    void createTrackers();
    void createMatchers();
    void createPoseEstimator();
    void createLandmarkInitializers();
    void createOptimizer();
};

} // namespace isae

#endif
