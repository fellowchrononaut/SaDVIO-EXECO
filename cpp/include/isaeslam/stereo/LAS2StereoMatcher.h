#ifndef LAS2_STEREO_MATCHER_H
#define LAS2_STEREO_MATCHER_H

#include "isaeslam/stereo/StereoMatcher.h"
#include <memory>

namespace isae {

// Lite Any Stereo V2 (Jing et al., arXiv:2606.24457) disparity on the CPU through OpenVINO.
//
// The ONNX model is exported outside SaDVIO by upstream export_onnx.py (inputs left/right 1x3xHxW float
// RGB 0-255, the network normalises internally; output disparity at input resolution, static H x W,
// max_disp 192). Pre/post-processing matches FFSStereoMatcher: uniform downscale (never up) then centred
// replicate padding up to the model size (upstream InputPadder), disparity cropped and rescaled back,
// pixels whose match falls left of the right image dropped, optional mirrored left-right check.
// Only available in builds with ISAESLAM_WITH_LAS2 (OpenVINO runtime).
class LAS2StereoMatcher : public StereoMatcher {
  public:
    explicit LAS2StereoMatcher(const StereoMatcherConfig& cfg);
    ~LAS2StereoMatcher() override;
    cv::Mat compute(const cv::Mat& rect_L, const cv::Mat& rect_R) override;
    std::string name() const override { return "LAS2"; }

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    double _lr_check_px;
};

} // namespace isae
#endif
