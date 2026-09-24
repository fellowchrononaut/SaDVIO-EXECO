#include "isaeslam/stereo/StereoMatcher.h"
#include "isaeslam/stereo/FFSStereoMatcher.h"
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace isae {

SGBMStereoMatcher::SGBMStereoMatcher(const StereoMatcherConfig& cfg) {
    const int P1 = 8  * 1 * cfg.block_size * cfg.block_size;
    const int P2 = 32 * 1 * cfg.block_size * cfg.block_size;
    _sgbm = cv::StereoSGBM::create(0, cfg.num_disparities, cfg.block_size, P1, P2,
                                   cfg.disp12_max_diff, cfg.pre_filter_cap, cfg.uniqueness_ratio,
                                   cfg.speckle_window_size, cfg.speckle_range,
                                   cv::StereoSGBM::MODE_SGBM_3WAY);
}

cv::Mat SGBMStereoMatcher::compute(const cv::Mat& rect_L, const cv::Mat& rect_R) {
    cv::Mat gray_L = rect_L, gray_R = rect_R;
    if (rect_L.channels() == 3)
        cv::cvtColor(rect_L, gray_L, cv::COLOR_BGR2GRAY);
    if (rect_R.channels() == 3)
        cv::cvtColor(rect_R, gray_R, cv::COLOR_BGR2GRAY);

    // SGBM output is 16-bit fixed point (1/16 px); invalid pixels come out negative
    cv::Mat disp_raw, disp;
    _sgbm->compute(gray_L, gray_R, disp_raw);
    disp_raw.convertTo(disp, CV_32F, 1.0 / 16.0);
    return disp;
}

std::unique_ptr<StereoMatcher> createStereoMatcher(const StereoMatcherConfig& cfg) {
    if (cfg.matcher == "sgbm")
        return std::make_unique<SGBMStereoMatcher>(cfg);
    if (cfg.matcher == "ffs") {
#ifdef ISAESLAM_WITH_FFS
        return std::make_unique<FFSStereoMatcher>(cfg);
#else
        throw std::runtime_error("[StereoMatcher] stereo_depth_matcher=ffs needs a build with "
                                 "-DISAESLAM_WITH_FFS=ON (TensorRT + CUDA runtime)");
#endif
    }
    throw std::runtime_error("[StereoMatcher] unknown stereo_depth_matcher '" + cfg.matcher +
                             "' (expected sgbm or ffs)");
}

} // namespace isae
