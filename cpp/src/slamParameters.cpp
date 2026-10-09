#include "isaeslam/slamParameters.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "isaeslam/data/landmarks/BBox3d.h"
#include "isaeslam/data/landmarks/Line3D.h"
#include "isaeslam/data/landmarks/Point3D.h"

#include "isaeslam/featuredetectors/custom_detectors/Line2DFeatureDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/csvKeypointDetector.h"
#include "isaeslam/featuredetectors/custom_detectors/semanticBBoxFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvBRISKFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvFASTFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvGFTTFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvKAZEFeatureDetector.h"
#include "isaeslam/featuredetectors/opencv_detectors/cvORBFeatureDetector.h"

#include "isaeslam/featurematchers/Point2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Point2DFeatureTracker.h"
#include "isaeslam/featurematchers/Line2DFeatureMatcher.h"
#include "isaeslam/featurematchers/Line2DFeatureTracker.h"
#include "isaeslam/featurematchers/semanticBBoxFeatureMatcher.h"
#include "isaeslam/featurematchers/semanticBBoxFeatureTracker.h"

#include "isaeslam/landmarkinitializer/Line3DlandmarkInitializer.h"
#include "isaeslam/landmarkinitializer/Point3DlandmarkInitializer.h"
#include "isaeslam/landmarkinitializer/semanticBBoxlandmarkInitializer.h"

#include "isaeslam/estimator/EpipolarPoseEstimator.h"
#include "isaeslam/estimator/PnPPoseEstimator.h"

#include "isaeslam/optimizers/AngularAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESAnalytic.h"
#include "isaeslam/optimizers/BundleAdjustmentCERESNumeric.h"
#include "isaeslam/dataproviders/adataprovider.h"

std::vector<std::string>
isae::validateConfig(const Config &cfg, int ncam, bool has_imu, std::vector<std::string> &warnings) {
    std::vector<std::string> errors;
    const std::vector<std::string> modes      = {"bimono", "mono", "nofov", "bimonovio", "monovio"};
    const std::vector<std::string> optimizers = {"Analytic", "Numeric", "AngularAnalytic"};
    auto known = [](const std::vector<std::string> &list, const std::string &v) {
        return std::find(list.begin(), list.end(), v) != list.end();
    };

    if (!known(modes, cfg.slam_mode))
        errors.push_back("unknown slam_mode '" + cfg.slam_mode + "' (bimono, mono, nofov, bimonovio, monovio)");
    if (!known(optimizers, cfg.optimizer))
        errors.push_back("unknown optimizer '" + cfg.optimizer + "' (Analytic, Numeric, AngularAnalytic)");

    const bool vio    = (cfg.slam_mode == "bimonovio" || cfg.slam_mode == "monovio");
    const bool stereo = (cfg.slam_mode == "bimono" || cfg.slam_mode == "bimonovio" || cfg.slam_mode == "nofov");
    if (vio && !has_imu)
        errors.push_back("slam_mode '" + cfg.slam_mode + "' needs an imu block in the dataset yaml");
    if (stereo && ncam < 2)
        errors.push_back("slam_mode '" + cfg.slam_mode + "' needs 2 cameras, the dataset has " + std::to_string(ncam));
    if (!stereo && ncam < 1)
        errors.push_back("slam_mode '" + cfg.slam_mode + "' needs a camera");

    // The Numeric optimizer does not implement these (the base class versions do nothing)
    if (cfg.optimizer == "Numeric" && cfg.marginalization == 1)
        errors.push_back("marginalization is not implemented for the Numeric optimizer");
    if (cfg.optimizer == "Numeric" && vio && cfg.estimate_td)
        errors.push_back("estimate_td is not implemented for the Numeric optimizer");

    if (cfg.sparsification && cfg.marginalization != 1)
        warnings.push_back("sparsification has no effect without marginalization");
    if (cfg.estimate_td && !vio)
        warnings.push_back("estimate_td has no effect in slam_mode '" + cfg.slam_mode + "'");
    if (cfg.max_lost_frames < 0)
        errors.push_back("max_lost_frames must be >= 1, or 0 for the mode default");
    if (cfg.reinit_carry_state != 0 && cfg.reinit_carry_state != 1)
        errors.push_back("reinit_carry_state must be 0 or 1");
    if (!(cfg.reinit_carry_max_age_vio > 0) || !(cfg.reinit_carry_max_age_vo > 0))
        errors.push_back("reinit_carry_max_age_vio and reinit_carry_max_age_vo must be > 0 (s)");
    if (cfg.loop_closure != 0 && cfg.loop_closure != 1)
        errors.push_back("loop_closure must be 0 or 1");
    if (cfg.loop_closure == 1) {
        if (cfg.loop_detector != "bow" && cfg.loop_detector != "proximity" && cfg.loop_detector != "proximity_bow" &&
            cfg.loop_detector != "learned")
            errors.push_back("loop_detector must be bow, proximity, proximity_bow or learned");
        if (cfg.loop_detector == "learned") {
            if (!std::ifstream(cfg.loop_model).good())
                errors.push_back("loop_model: cannot read '" + cfg.loop_model + "' (needed by loop_detector learned)");
            if (cfg.loop_model_device != "CPU" && cfg.loop_model_device != "GPU")
                errors.push_back("loop_model_device must be CPU or GPU");
        }
        if ((cfg.loop_detector == "bow" || cfg.loop_detector == "proximity_bow") &&
            !std::ifstream(cfg.loop_vocabulary).good())
            errors.push_back("loop_vocabulary: cannot read '" + cfg.loop_vocabulary + "' (needed by loop_detector " +
                             cfg.loop_detector + ")");
        if (cfg.slam_mode == "mono")
            errors.push_back("loop_closure is not available in slam_mode mono (monocular VO needs a Sim3 pose graph)");
        if (cfg.loop_graph_dof != 0 && cfg.loop_graph_dof != 4 && cfg.loop_graph_dof != 6)
            errors.push_back("loop_graph_dof must be 0 (4 with an IMU, 6 without), 4 or 6");
        if (cfg.loop_graph_dof == 4 && cfg.slam_mode != "monovio" && cfg.slam_mode != "bimonovio")
            errors.push_back("loop_graph_dof 4 needs an IMU (roll and pitch observable)");
        if (cfg.loop_correct_window != 0 && cfg.loop_correct_window != 1)
            errors.push_back("loop_correct_window must be 0 or 1");
        if (!(cfg.loop_min_gap >= 0) || cfg.loop_min_inliers < 6 || !(cfg.loop_gate_radius > 0))
            errors.push_back("loop_min_gap must be >= 0, loop_min_inliers >= 6 and loop_gate_radius > 0");
    }

    return errors;
}

isae::SLAMParameters::SLAMParameters(const std::string config_folder_path) {
    std::cout << "------------------------------------" << std::endl;
    readConfigFile(config_folder_path);
    createProvider();

    // Refuse option combinations that cannot work, instead of silently doing nothing
    std::vector<std::string> warnings;
    std::vector<std::string> errors = validateConfig(
        _config, _data_provider->getNCam(), YAML::LoadFile(_config.dataset_path)["imu"].IsDefined(), warnings);
    for (const auto &w : warnings)
        std::cerr << "CONFIG WARNING: " << w << std::endl;
    if (!errors.empty()) {
        for (const auto &e : errors)
            std::cerr << "CONFIG ERROR: " << e << std::endl;
        throw std::runtime_error("invalid configuration (see CONFIG ERROR above)");
    }

    createDetectors();
    createMatchers();
    createTrackers();
    createLandmarkInitializers();
    createPoseEstimator();
    createOptimizer();
    std::cout << "------------------------------------" << std::endl;
}

void isae::SLAMParameters::readConfigFile(const std::string &path_config_folder) {
    YAML::Node yaml_file = YAML::LoadFile(path_config_folder + "/config.yaml");

    // Dataset ID
    _config.dataset_id     = yaml_file["dataset_id"].as<std::string>();
    _config.dataset_path   = path_config_folder + "/dataset/" + _config.dataset_id + ".yaml";
    _config.slam_mode      = yaml_file["slam_mode"].as<std::string>();
    _config.enable_visu    = yaml_file["enable_visu"].as<int>();
    _config.multithreading = yaml_file["multithreading"].as<int>();

    // Image processing
    _config.contrast_enhancer = yaml_file["contrast_enhancer"].as<int>();
    _config.clahe_clip        = yaml_file["clahe_clip"].as<float>();
    _config.downsampling      = yaml_file["downsampling"].as<float>();

    // SLAM parameters
    _config.pose_estimator        = yaml_file["pose_estimator"].as<std::string>();
    _config.optimizer             = yaml_file["optimizer"].as<std::string>();
    _config.tracker               = yaml_file["tracker"].as<std::string>();
    _config.estimate_td           = yaml_file["estimate_td"].as<int>();
    _config.max_lost_frames       = yaml_file["max_lost_frames"] ? yaml_file["max_lost_frames"].as<int>() : 0;
    _config.reinit_carry_state    = yaml_file["reinit_carry_state"] ? yaml_file["reinit_carry_state"].as<int>() : 1;
    _config.reinit_carry_max_age_vio =
        yaml_file["reinit_carry_max_age_vio"] ? yaml_file["reinit_carry_max_age_vio"].as<double>() : 2.0;
    _config.reinit_carry_max_age_vo =
        yaml_file["reinit_carry_max_age_vo"] ? yaml_file["reinit_carry_max_age_vo"].as<double>() : 2.0;
    _config.loop_closure     = yaml_file["loop_closure"] ? yaml_file["loop_closure"].as<int>() : 0;
    _config.loop_detector    = yaml_file["loop_detector"] ? yaml_file["loop_detector"].as<std::string>() : "bow";
    _config.loop_vocabulary  = yaml_file["loop_vocabulary"] ? yaml_file["loop_vocabulary"].as<std::string>() : "";
    _config.loop_model       = yaml_file["loop_model"] ? yaml_file["loop_model"].as<std::string>() : "";
    _config.loop_model_device =
        yaml_file["loop_model_device"] ? yaml_file["loop_model_device"].as<std::string>() : "CPU";
    _config.loop_model_threads = yaml_file["loop_model_threads"] ? yaml_file["loop_model_threads"].as<int>() : 8;
    _config.loop_min_gap     = yaml_file["loop_min_gap"] ? yaml_file["loop_min_gap"].as<double>() : 20.0;
    _config.loop_min_inliers = yaml_file["loop_min_inliers"] ? yaml_file["loop_min_inliers"].as<int>() : 12;
    _config.loop_gate_radius = yaml_file["loop_gate_radius"] ? yaml_file["loop_gate_radius"].as<double>() : 2.0;
    _config.loop_graph_dof   = yaml_file["loop_graph_dof"] ? yaml_file["loop_graph_dof"].as<int>() : 0;
    _config.loop_correct_window =
        yaml_file["loop_correct_window"] ? yaml_file["loop_correct_window"].as<int>() : 0;
    _config.min_kf_number         = yaml_file["min_kf_number"].as<int>();
    _config.max_kf_number         = yaml_file["max_kf_number"].as<int>();
    _config.fixed_frame_number    = yaml_file["fixed_frame_number"].as<int>();
    _config.min_lmk_number        = yaml_file["min_lmk_number"].as<float>();
    _config.min_movement_parallax = yaml_file["min_movement_parallax"].as<float>();
    _config.max_movement_parallax = yaml_file["max_movement_parallax"].as<float>();
    _config.marginalization       = yaml_file["marginalization"].as<int>();
    _config.sparsification        = yaml_file["sparsification"].as<int>();
    _config.mesh3D                = yaml_file["mesh3d"].as<int>();
    _config.ZNCC_tsh              = yaml_file["ZNCC_tsh"].as<double>();
    _config.max_length_tsh        = yaml_file["max_length_tsh"].as<double>();

    // dense_depth / dense_mesh3D / stereo_depth_enabled (backward compat aliases)
    if (yaml_file["dense_depth"])
        _config.dense_depth = yaml_file["dense_depth"].as<bool>();
    else if (yaml_file["dense_mesh3D"])
        _config.dense_depth = yaml_file["dense_mesh3D"].as<bool>();
    else if (yaml_file["stereo_depth_enabled"])
        _config.dense_depth = yaml_file["stereo_depth_enabled"].as<bool>();
    if (yaml_file["dense_mesh_method"])
        _config.dense_mesh_method     = yaml_file["dense_mesh_method"].as<std::string>();
    if (yaml_file["stereo_depth_scale"])
        _config.stereo_depth_scale    = yaml_file["stereo_depth_scale"].as<double>();
    if (yaml_file["stereo_depth_max_depth"])
        _config.stereo_depth_max_depth = yaml_file["stereo_depth_max_depth"].as<double>();
    if (yaml_file["stereo_depth_num_disp"])
        _config.stereo_depth_num_disp     = yaml_file["stereo_depth_num_disp"].as<int>();
    if (yaml_file["stereo_depth_block_size"])
        _config.stereo_depth_block_size   = yaml_file["stereo_depth_block_size"].as<int>();
    if (yaml_file["stereo_depth_stride"])
        _config.stereo_depth_stride       = yaml_file["stereo_depth_stride"].as<int>();
    if (yaml_file["stereo_depth_uniqueness_ratio"])
        _config.stereo_depth_uniqueness_ratio = yaml_file["stereo_depth_uniqueness_ratio"].as<int>();
    if (yaml_file["stereo_depth_speckle_window_size"])
        _config.stereo_depth_speckle_window_size = yaml_file["stereo_depth_speckle_window_size"].as<int>();
    if (yaml_file["stereo_depth_speckle_range"])
        _config.stereo_depth_speckle_range = yaml_file["stereo_depth_speckle_range"].as<int>();
    if (yaml_file["stereo_depth_disp12_max_diff"])
        _config.stereo_depth_disp12_max_diff = yaml_file["stereo_depth_disp12_max_diff"].as<int>();
    if (yaml_file["stereo_depth_pre_filter_cap"])
        _config.stereo_depth_pre_filter_cap = yaml_file["stereo_depth_pre_filter_cap"].as<int>();
    if (yaml_file["stereo_depth_publish_sgbm_images"])
        _config.stereo_depth_publish_sgbm_images = yaml_file["stereo_depth_publish_sgbm_images"].as<bool>();
    if (yaml_file["stereo_depth_matcher"])
        _config.stereo_depth_matcher = yaml_file["stereo_depth_matcher"].as<std::string>();
    if (yaml_file["stereo_depth_ffs_engine"])
        _config.stereo_depth_ffs_engine = yaml_file["stereo_depth_ffs_engine"].as<std::string>();
    if (yaml_file["stereo_depth_ffs_lr_check"])
        _config.stereo_depth_ffs_lr_check = yaml_file["stereo_depth_ffs_lr_check"].as<double>();
    if (yaml_file["stereo_depth_las2_model_dir"])
        _config.stereo_depth_las2_model_dir = yaml_file["stereo_depth_las2_model_dir"].as<std::string>();
    if (yaml_file["stereo_depth_las2_size"])
        _config.stereo_depth_las2_size = yaml_file["stereo_depth_las2_size"].as<std::string>();
    if (yaml_file["stereo_depth_las2_resolution"])
        _config.stereo_depth_las2_resolution = yaml_file["stereo_depth_las2_resolution"].as<std::string>();
    if (yaml_file["stereo_depth_las2_onnx"])
        _config.stereo_depth_las2_onnx = yaml_file["stereo_depth_las2_onnx"].as<std::string>();
    _config.stereo_depth_las2_onnx_path = !_config.stereo_depth_las2_onnx.empty()
        ? _config.stereo_depth_las2_onnx
        : _config.stereo_depth_las2_model_dir + "/las2_" + _config.stereo_depth_las2_size + "_" +
              _config.stereo_depth_las2_resolution + ".onnx";
    if (yaml_file["stereo_depth_las2_device"])
        _config.stereo_depth_las2_device = yaml_file["stereo_depth_las2_device"].as<std::string>();
    if (yaml_file["stereo_depth_las2_precision"])
        _config.stereo_depth_las2_precision = yaml_file["stereo_depth_las2_precision"].as<std::string>();
    if (yaml_file["stereo_depth_las2_threads"])
        _config.stereo_depth_las2_threads = yaml_file["stereo_depth_las2_threads"].as<int>();
    if (yaml_file["stereo_depth_las2_lr_check"])
        _config.stereo_depth_las2_lr_check = yaml_file["stereo_depth_las2_lr_check"].as<double>();
    if (yaml_file["dense_gp_cell_size"])
        _config.dense_gp_cell_size = yaml_file["dense_gp_cell_size"].as<double>();
    if (yaml_file["dense_gp_num_test"])
        _config.dense_gp_num_test = yaml_file["dense_gp_num_test"].as<int>();
    if (yaml_file["dense_gp_min_pts_per_cell"])
        _config.dense_gp_min_pts_per_cell = yaml_file["dense_gp_min_pts_per_cell"].as<int>();
    if (yaml_file["dense_gp_kernel_length"])
        _config.dense_gp_kernel_length = yaml_file["dense_gp_kernel_length"].as<double>();
    if (yaml_file["dense_gp_variance_sensor"])
        _config.dense_gp_variance_sensor = yaml_file["dense_gp_variance_sensor"].as<double>();
    if (yaml_file["dense_gp_max_variance"])
        _config.dense_gp_max_variance = yaml_file["dense_gp_max_variance"].as<double>();
    if (yaml_file["dense_gp_full_cover"])
        _config.dense_gp_full_cover = yaml_file["dense_gp_full_cover"].as<bool>();
    if (yaml_file["dense_gp_eigen_1"])
        _config.dense_gp_eigen_1 = yaml_file["dense_gp_eigen_1"].as<double>();
    if (yaml_file["dense_gp_eigen_2"])
        _config.dense_gp_eigen_2 = yaml_file["dense_gp_eigen_2"].as<double>();
    if (yaml_file["dense_gp_eigen_3"])
        _config.dense_gp_eigen_3 = yaml_file["dense_gp_eigen_3"].as<double>();
    if (yaml_file["dense_gp_stitch_seams"])
        _config.dense_gp_stitch_seams = yaml_file["dense_gp_stitch_seams"].as<bool>();
    if (yaml_file["dense_gp_seam_max_variance"])
        _config.dense_gp_seam_max_variance = yaml_file["dense_gp_seam_max_variance"].as<double>();
    if (yaml_file["dense_gp_seam_max_edge_length"])
        _config.dense_gp_seam_max_edge_length = yaml_file["dense_gp_seam_max_edge_length"].as<double>();
    if (yaml_file["dense_gp_seam_max_prediction_gap"])
        _config.dense_gp_seam_max_prediction_gap = yaml_file["dense_gp_seam_max_prediction_gap"].as<double>();
    if (yaml_file["dense_gp_seam_min_normal_cos"])
        _config.dense_gp_seam_min_normal_cos = yaml_file["dense_gp_seam_min_normal_cos"].as<double>();
    if (yaml_file["dense_keep_all_keyframes"])
        _config.dense_keep_all_keyframes = yaml_file["dense_keep_all_keyframes"].as<bool>();
    if (yaml_file["dense_gp_global_map"])
        _config.dense_gp_global_map = yaml_file["dense_gp_global_map"].as<bool>();
    if (yaml_file["dense_gp_variance_map_update"])
        _config.dense_gp_variance_map_update = yaml_file["dense_gp_variance_map_update"].as<double>();
    if (yaml_file["dense_gp_max_raw_points_per_cell"])
        _config.dense_gp_max_raw_points_per_cell = yaml_file["dense_gp_max_raw_points_per_cell"].as<int>();
    if (yaml_file["dense_gp_register"])
        _config.dense_gp_register = yaml_file["dense_gp_register"].as<bool>();
    if (yaml_file["dense_gp_register_times"])
        _config.dense_gp_register_times = yaml_file["dense_gp_register_times"].as<int>();
    if (yaml_file["dense_gp_variance_register"])
        _config.dense_gp_variance_register = yaml_file["dense_gp_variance_register"].as<double>();
    if (yaml_file["dense_gp_cross_cell_overlap_length"])
        _config.dense_gp_cross_cell_overlap_length = yaml_file["dense_gp_cross_cell_overlap_length"].as<int>();
    if (yaml_file["dense_gp_register_converge_thr"])
        _config.dense_gp_register_converge_thr = yaml_file["dense_gp_register_converge_thr"].as<double>();
    if (yaml_file["dense_gp_register_huber"])
        _config.dense_gp_register_huber = yaml_file["dense_gp_register_huber"].as<double>();
    if (yaml_file["dense_gp_register_min_matches"])
        _config.dense_gp_register_min_matches = yaml_file["dense_gp_register_min_matches"].as<int>();
    if (yaml_file["dense_gp_register_max_translation"])
        _config.dense_gp_register_max_translation = yaml_file["dense_gp_register_max_translation"].as<double>();
    if (yaml_file["dense_gp_register_max_rotation_deg"])
        _config.dense_gp_register_max_rotation_deg = yaml_file["dense_gp_register_max_rotation_deg"].as<double>();
    if (yaml_file["dense_gp_register_carry_correction"])
        _config.dense_gp_register_carry_correction = yaml_file["dense_gp_register_carry_correction"].as<bool>();
    if (yaml_file["dense_gp_register_depth_weighting"])
        _config.dense_gp_register_depth_weighting = yaml_file["dense_gp_register_depth_weighting"].as<bool>();
    if (yaml_file["dense_gp_register_depth_ref"])
        _config.dense_gp_register_depth_ref = yaml_file["dense_gp_register_depth_ref"].as<double>();
    if (yaml_file["dense_gp_global_mesh_path"])
        _config.dense_gp_global_mesh_path = yaml_file["dense_gp_global_mesh_path"].as<std::string>();
    if (yaml_file["dense_gp_save_every"])
        _config.dense_gp_save_every = yaml_file["dense_gp_save_every"].as<int>();
    if (yaml_file["dense_vdbgpdf_preset"])
        _config.dense_vdbgpdf_preset = yaml_file["dense_vdbgpdf_preset"].as<std::string>();
    _config.dense_vdbgpdf_preset_path = path_config_folder + "/vdbgpdf/" + _config.dense_vdbgpdf_preset + ".yaml";
    if (yaml_file["dense_vdbgpdf_stride"])
        _config.dense_vdbgpdf_stride = yaml_file["dense_vdbgpdf_stride"].as<int>();
    if (yaml_file["dense_vdbgpdf_mesh_every"])
        _config.dense_vdbgpdf_mesh_every = yaml_file["dense_vdbgpdf_mesh_every"].as<int>();
    if (yaml_file["dense_vdbgpdf_mesh_path"])
        _config.dense_vdbgpdf_mesh_path = yaml_file["dense_vdbgpdf_mesh_path"].as<std::string>();
    if (yaml_file["dense_pd_steiner_spacing"])
        _config.dense_pd_steiner_spacing = yaml_file["dense_pd_steiner_spacing"].as<int>();
    if (yaml_file["dense_pd_lambda"])
        _config.dense_pd_lambda = yaml_file["dense_pd_lambda"].as<double>();
    if (yaml_file["dense_pd_num_iterations"])
        _config.dense_pd_num_iterations = yaml_file["dense_pd_num_iterations"].as<int>();
    if (yaml_file["dense_pd_tau"])
        _config.dense_pd_tau = yaml_file["dense_pd_tau"].as<double>();
    if (yaml_file["dense_pd_sigma"])
        _config.dense_pd_sigma = yaml_file["dense_pd_sigma"].as<double>();
    if (yaml_file["dense_pd_theta"])
        _config.dense_pd_theta = yaml_file["dense_pd_theta"].as<double>();
    if (yaml_file["dense_pd_min_depth"])
        _config.dense_pd_min_depth = yaml_file["dense_pd_min_depth"].as<double>();

    // Features type
    YAML::Node features_node = yaml_file["features_handled"];

    for (YAML::iterator it = features_node.begin(); it != features_node.end(); ++it) {
        FeatureStruct feature_struct;
        feature_struct.label_feature            = (*it)["label_feature"].as<std::string>();
        feature_struct.detector_label           = (*it)["detector_label"].as<std::string>();
        feature_struct.number_detected_features = (*it)["number_detected_features"].as<int>();
        feature_struct.n_features_per_cell      = (*it)["n_features_per_cell"].as<int>();
        feature_struct.tracker_label            = (*it)["tracker_label"].as<std::string>();
        feature_struct.tracker_height           = (*it)["tracker_height"].as<int>();
        feature_struct.tracker_nlvls_pyramids   = (*it)["tracker_nlvls_pyramids"].as<int>();
        feature_struct.tracker_max_err          = (*it)["tracker_max_err"].as<double>();
        feature_struct.tracker_width            = (*it)["tracker_width"].as<int>();
        feature_struct.matcher_label            = (*it)["matcher_label"].as<std::string>();
        feature_struct.max_matching_dist        = (*it)["max_matching_dist"].as<double>();
        feature_struct.matcher_height           = (*it)["matcher_height"].as<int>();
        feature_struct.matcher_width            = (*it)["matcher_width"].as<int>();
        feature_struct.lmk_triangulator         = (*it)["lmk_triangulator"].as<std::string>();

        _config.features_handled.push_back(feature_struct);
    }
}

void isae::SLAMParameters::createProvider() {
    std::cout << "Create Data Provider" << std::endl;
    this->_data_provider = std::make_shared<ADataProvider>(_config.dataset_path, _config);
}

void isae::SLAMParameters::createDetectors() {

    std::cout << "Create Feature detectors" << std::endl;
    for (auto config_line : _config.features_handled) {
        if (config_line.detector_label == "cvORBFeatureDetector") {
            std::cout << "+ Adding cvORBFeatureDetector" << std::endl;
            isae::cvORBFeatureDetector orb_detector = isae::cvORBFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvORBFeatureDetector>(orb_detector);
        } else if (config_line.detector_label == "cvKAZEFeatureDetector") {
            std::cout << "+ Adding cvKAZEFeatureDetector" << std::endl;
            isae::cvKAZEFeatureDetector kaze_detector = isae::cvKAZEFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvKAZEFeatureDetector>(kaze_detector);
        } else if (config_line.detector_label == "cvBRISKFeatureDetector") {
            std::cout << "+ Adding cvBRISKFeatureDetector" << std::endl;
            isae::cvBRISKFeatureDetector brisk_detector = isae::cvBRISKFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvBRISKFeatureDetector>(brisk_detector);
        } else if (config_line.detector_label == "cvFASTFeatureDetector") {
            std::cout << "+ Adding cvFASTFeatureDetector" << std::endl;
            isae::cvFASTFeatureDetector fast_detector = isae::cvFASTFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvFASTFeatureDetector>(fast_detector);
        } else if (config_line.detector_label == "cvGFTTFeatureDetector") {
            std::cout << "+ Adding cvGFTTFeatureDetector" << std::endl;
            isae::cvGFTTFeatureDetector gftt_detector = isae::cvGFTTFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::cvGFTTFeatureDetector>(gftt_detector);
        } else if (config_line.detector_label == "cvCSVFeatureDetector") {
            std::cout << "+ Adding cvCSVFeatureDetector" << std::endl;
            isae::CsvKeypointDetector SIFT_detector = isae::CsvKeypointDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::CsvKeypointDetector>(SIFT_detector);
        } else if (config_line.detector_label == "Line2DFeatureDetector") {
            std::cout << "+ Adding Line2DFeatureDetector" << std::endl;
            isae::Line2DFeatureDetector lineDetector = isae::Line2DFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell, config_line.max_matching_dist);
            _detector_map[config_line.label_feature] = std::make_shared<isae::Line2DFeatureDetector>(lineDetector);
        } else if (config_line.detector_label == "semanticBBoxFeatureDetector") {
            std::cout << "+ Adding semanticBBoxFeatureDetector" << std::endl;
            isae::semanticBBoxFeatureDetector bboxFeatureDetector = isae::semanticBBoxFeatureDetector(
                config_line.number_detected_features, config_line.n_features_per_cell);
            _detector_map[config_line.label_feature] =
                std::make_shared<isae::semanticBBoxFeatureDetector>(bboxFeatureDetector);
        }
    }
}

void isae::SLAMParameters::createMatchers() {
    std::cout << "Create Feature Matchers" << std::endl;
    for (auto config_line : _config.features_handled) {

        isae::FeatureMatcherStruct matcher;
        matcher.matcher_height = config_line.matcher_height;
        matcher.matcher_width  = config_line.matcher_width;
        // Get the associated detector
        std::shared_ptr<AFeatureDetector> detector = _detector_map[config_line.label_feature];

        if (config_line.matcher_label == "Point2DFeatureMatcher") {
            std::cout << "+ Adding Point2DFeatureMatcher" << std::endl;
            isae::Point2DFeatureMatcher p2dMatcher(detector);
            matcher.feature_matcher                 = std::make_shared<Point2DFeatureMatcher>(p2dMatcher);
            _matcher_map[config_line.label_feature] = matcher;

        } else if (config_line.matcher_label == "Line2DFeatureMatcher") {
            std::cout << "+ Adding LineFeatureMatcher" << std::endl;
            isae::Line2DFeatureMatcher line2DMatcher(detector);
            matcher.feature_matcher                 = std::make_shared<Line2DFeatureMatcher>(line2DMatcher);
            _matcher_map[config_line.label_feature] = matcher;

        } else if (config_line.matcher_label == "semanticBBoxFeatureMatcher") {
            std::cout << "+ Adding semanticBBoxFeatureMatcher" << std::endl;
            isae::semanticBBoxFeatureMatcher BBoxMatcher(detector);
            matcher.feature_matcher                 = std::make_shared<semanticBBoxFeatureMatcher>(BBoxMatcher);
            _matcher_map[config_line.label_feature] = matcher;
        }
    }
}

void isae::SLAMParameters::createTrackers() {
    std::cout << "Create Feature Trackers" << std::endl;
    for (auto config_line : _config.features_handled) {

        FeatureTrackerStruct tracker;
        tracker.tracker_height         = config_line.tracker_height;
        tracker.tracker_width          = config_line.tracker_width;
        tracker.tracker_nlvls_pyramids = config_line.tracker_nlvls_pyramids;
        tracker.tracker_max_err        = config_line.tracker_max_err;

        // Get the associated detector
        std::shared_ptr<AFeatureDetector> detector = _detector_map[config_line.label_feature];

        if (config_line.tracker_label == "Point2DFeatureTracker") {
            std::cout << "+ Adding Point2DFeatureTracker" << std::endl;
            isae::Point2DFeatureTracker p2dTracker(detector);
            tracker.feature_tracker                 = std::make_shared<Point2DFeatureTracker>(p2dTracker);
            _tracker_map[config_line.label_feature] = tracker;
            
        } else if (config_line.tracker_label == "Line2DFeatureTracker") {
            std::cout << "+ Adding LineFeatureTracker" << std::endl;
            isae::Line2DFeatureTracker line2DTracker(detector);
            tracker.feature_tracker                 = std::make_shared<Line2DFeatureTracker>(line2DTracker);
            _tracker_map[config_line.label_feature] = tracker;

        } else if (config_line.tracker_label == "semanticBBoxFeatureTracker") {
            std::cout << "+ Adding semanticBBoxFeatureTracker" << std::endl;
            isae::semanticBBoxFeatureTracker BBoxTracker(detector);
            tracker.feature_tracker                 = std::make_shared<semanticBBoxFeatureTracker>(BBoxTracker);
            _tracker_map[config_line.label_feature] = tracker;
        }
    }
}

void isae::SLAMParameters::createLandmarkInitializers() {
    std::cout << "Create Landmarks Initializers" << std::endl;
    for (auto config_line : _config.features_handled) {

        if (config_line.lmk_triangulator == "Point3DLandmarkInitializer") {
            std::cout << "+ Adding Point3DLandmarkInitializer" << std::endl;
            _lmk_init_map[config_line.label_feature] = std::make_shared<Point3DLandmarkInitializer>();
        } else if (config_line.lmk_triangulator == "Line3DLandmarkInitializer") {
            std::cout << "+ Adding Line3DLandmarkInitializer " << std::endl;
            _lmk_init_map[config_line.label_feature] = std::make_shared<Line3DLandmarkInitializer>();
        } else if (config_line.lmk_triangulator == "semanticBBoxLandmarkInitializer") {
            std::cout << "+ Adding semanticBBoxLandmarkInitializer" << std::endl;
            _lmk_init_map[config_line.label_feature] = std::make_shared<semanticBBoxLandmarkInitializer>();
        }
    }
}

void isae::SLAMParameters::createPoseEstimator() {
    std::cout << "Create Interframe Pose Estimator" << std::endl;
    if (_config.pose_estimator == "epipolar") {
        std::cout << "+ Adding EpipolarPoseEstimator" << std::endl;
        isae::EpipolarPoseEstimator pose_estimator;
        _pose_estimator = std::make_shared<isae::EpipolarPoseEstimator>(pose_estimator);
    } else if (_config.pose_estimator == "imu") {
        std::cout << "+ Adding IMUPredictor -- TODO" << std::endl;

    } else if (_config.pose_estimator == "pnp") {
        std::cout << "+ Adding PnPPoseEstimator" << std::endl;
        isae::PnPPoseEstimator pose_estimator;
        _pose_estimator = std::make_shared<isae::PnPPoseEstimator>(pose_estimator);

    }
}

void isae::SLAMParameters::createOptimizer() {

    std::cout << "Create Optimizer" << std::endl;
    if (_config.optimizer == "Numeric") {
        std::cout << "+ Adding CERES optimizer with numerical jacobians" << std::endl;
        isae::BundleAdjustmentCERESNumeric ceres_ba;
        _optimizer_frontend = std::make_shared<isae::BundleAdjustmentCERESNumeric>(ceres_ba);
        _optimizer_backend  = std::make_shared<isae::BundleAdjustmentCERESNumeric>(ceres_ba);
    } else if (_config.optimizer == "Analytic") {
        std::cout << "+ Adding CERES optimizer with anlytical jacobians" << std::endl;
        isae::BundleAdjustmentCERESAnalytic ceres_ba;
        _optimizer_frontend = std::make_shared<isae::BundleAdjustmentCERESAnalytic>(ceres_ba);
        _optimizer_backend  = std::make_shared<isae::BundleAdjustmentCERESAnalytic>(ceres_ba);
    } else if (_config.optimizer == "AngularAnalytic") {
        std::cout << "+ Adding Angular error CERES optimizer with analytical jacobians" << std::endl;
        isae::AngularAdjustmentCERESAnalytic ceres_ba;
        _optimizer_frontend = std::make_shared<isae::AngularAdjustmentCERESAnalytic>(ceres_ba);
        _optimizer_backend  = std::make_shared<isae::AngularAdjustmentCERESAnalytic>(ceres_ba);
    }

    // With marginalization, VO also uses the robust visual loss (VIO always does)
    if (_optimizer_frontend && _optimizer_backend) {
        _optimizer_frontend->setRobustVisualVO(_config.marginalization == 1);
        _optimizer_backend->setRobustVisualVO(_config.marginalization == 1);
    }
}
