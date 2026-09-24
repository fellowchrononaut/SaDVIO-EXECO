#ifndef STEREO_MATCHER_H
#define STEREO_MATCHER_H

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <memory>
#include <string>

namespace isae {

// Dense disparity backends for MarginalDepthInjector (doc/dense_ffs_stereo.md).
struct StereoMatcherConfig {
    std::string matcher = "sgbm"; // "sgbm" = OpenCV SGBM, "ffs" = Fast-FoundationStereo (TensorRT)

    // SGBM
    int num_disparities     = 64;
    int block_size          = 5;
    int uniqueness_ratio    = 10;
    int speckle_window_size = 100;
    int speckle_range       = 32;
    int disp12_max_diff     = 1;
    int pre_filter_cap      = 0;

    // Fast-FoundationStereo: TensorRT engine built with trtexec from the upstream single-ONNX export
    std::string ffs_engine_path;
    double      ffs_lr_check_px = 0.0; // > 0: second (mirrored) inference, drop pixels with |dL - dR| above this
};

// Computes a disparity map (CV_32F, pixels, <= 0 = invalid) from a rectified stereo pair.
class StereoMatcher {
  public:
    virtual ~StereoMatcher() = default;
    virtual cv::Mat compute(const cv::Mat& rect_L, const cv::Mat& rect_R) = 0;
    virtual std::string name() const = 0;
};

class SGBMStereoMatcher : public StereoMatcher {
  public:
    explicit SGBMStereoMatcher(const StereoMatcherConfig& cfg);
    cv::Mat compute(const cv::Mat& rect_L, const cv::Mat& rect_R) override;
    std::string name() const override { return "SGBM"; }

  private:
    cv::Ptr<cv::StereoSGBM> _sgbm;
};

// Throws std::runtime_error for an unknown matcher, or "ffs" in a build without ISAESLAM_WITH_FFS.
std::unique_ptr<StereoMatcher> createStereoMatcher(const StereoMatcherConfig& cfg);

} // namespace isae
#endif
