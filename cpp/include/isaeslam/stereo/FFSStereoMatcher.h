#ifndef FFS_STEREO_MATCHER_H
#define FFS_STEREO_MATCHER_H

#include "isaeslam/stereo/StereoMatcher.h"
#include <memory>

namespace isae {

// Fast-FoundationStereo (Wen et al., CVPR 2026) disparity through a TensorRT engine.
//
// The engine is built outside SaDVIO: upstream scripts/make_single_onnx.py exports the network with
// ImageNet normalisation stripped (inputs left_image/right_image 1x3xHxW float, output disparity
// 1x1xHxW), and trtexec turns it into an engine for the target GPU. Pre/post-processing follows
// upstream scripts/run_demo.py: RGB 0-255 -> ImageNet normalisation, replicate padding split
// evenly on both sides up to the engine size (uniform downscale first if the image is larger),
// disparity cropped back, pixels whose match falls left of the right image dropped.
// Only available in builds with ISAESLAM_WITH_FFS (TensorRT + CUDA runtime).
class FFSStereoMatcher : public StereoMatcher {
  public:
    explicit FFSStereoMatcher(const StereoMatcherConfig& cfg);
    ~FFSStereoMatcher() override;
    cv::Mat compute(const cv::Mat& rect_L, const cv::Mat& rect_R) override;
    std::string name() const override { return "FFS"; }

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
    double _lr_check_px;
};

} // namespace isae
#endif
