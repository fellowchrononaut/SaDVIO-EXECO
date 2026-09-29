// Compiled to nothing unless the library is built with -DISAESLAM_WITH_LAS2=ON (cmake/ISAESLAM_LAS2.cmake).
#ifdef ISAESLAM_WITH_LAS2

#include "isaeslam/stereo/LAS2StereoMatcher.h"
#include <openvino/openvino.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace isae {

namespace {

// Names of the upstream export (export_onnx.py)
constexpr const char* kLeft  = "left";
constexpr const char* kRight = "right";
constexpr const char* kDisp  = "disparity";

ov::element::Type precisionFromName(const std::string& p) {
    if (p == "f32")
        return ov::element::f32;
    if (p == "f16")
        return ov::element::f16;
    if (p == "bf16")
        return ov::element::bf16;
    throw std::runtime_error("[LAS2] unknown stereo_depth_las2_precision '" + p + "' (expected f32, f16 or bf16)");
}

} // namespace

struct LAS2StereoMatcher::Impl {
    ov::Core          core;
    ov::CompiledModel compiled;
    ov::InferRequest  request;
    int H = 0, W = 0;

    // Image placement inside the model input: uniform downscale (never up) then centred replicate padding
    struct Placement {
        double scale;
        int sw, sh, pad_x, pad_y;
    };

    explicit Impl(const StereoMatcherConfig& cfg) {
        const std::string& path = cfg.las2_onnx_path;
        if (!std::filesystem::is_regular_file(path))
            throw std::runtime_error("[LAS2] ONNX model not found: " + path +
                                     " (check stereo_depth_las2_model_dir / _size / _resolution)");
        std::shared_ptr<ov::Model> model = core.read_model(path);

        for (const char* name : {kLeft, kRight}) {
            const ov::Output<ov::Node> in = model->input(name); // throws if missing
            if (in.get_element_type() != ov::element::f32 || in.get_partial_shape().is_dynamic())
                throw std::runtime_error(std::string("[LAS2] input '") + name +
                                         "' must be static float32 (expected the upstream export_onnx.py model)");
        }
        const ov::Shape in = model->input(kLeft).get_shape();
        if (in.size() != 4 || in[0] != 1 || in[1] != 3)
            throw std::runtime_error("[LAS2] expected a 1x3xHxW input");
        H = static_cast<int>(in[2]);
        W = static_cast<int>(in[3]);
        if (ov::shape_size(model->output(kDisp).get_shape()) != static_cast<size_t>(H) * W)
            throw std::runtime_error("[LAS2] output 'disparity' is not H x W");

        ov::AnyMap props{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                         ov::hint::inference_precision(precisionFromName(cfg.las2_precision))};
        if (cfg.las2_threads > 0)
            props.insert(ov::inference_num_threads(cfg.las2_threads));
        compiled = core.compile_model(model, cfg.las2_device, props);
        request  = compiled.create_infer_request();
    }

    Placement place(const cv::Size& src) const {
        Placement p;
        p.scale = std::min({1.0, static_cast<double>(W) / src.width, static_cast<double>(H) / src.height});
        p.sw    = std::max(1, static_cast<int>(std::lround(src.width * p.scale)));
        p.sh    = std::max(1, static_cast<int>(std::lround(src.height * p.scale)));
        p.pad_x = (W - p.sw) / 2;
        p.pad_y = (H - p.sh) / 2;
        return p;
    }

    // RGB 0-255, CHW, straight into the request's input tensor
    void toInput(const cv::Mat& img, const Placement& p, const char* name) {
        if (img.depth() != CV_8U)
            throw std::runtime_error("[LAS2] expects 8-bit images");
        cv::Mat rgb;
        if (img.channels() == 1)
            cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        if (p.sw != img.cols || p.sh != img.rows)
            cv::resize(rgb, rgb, cv::Size(p.sw, p.sh), 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(rgb, rgb, p.pad_y, H - p.sh - p.pad_y, p.pad_x, W - p.sw - p.pad_x,
                           cv::BORDER_REPLICATE);
        float* chw = request.get_tensor(name).data<float>();
        const size_t plane = static_cast<size_t>(H) * W;
        for (int v = 0; v < H; ++v) {
            const uint8_t* row = rgb.ptr<uint8_t>(v);
            for (int u = 0; u < W; ++u)
                for (int c = 0; c < 3; ++c)
                    chw[c * plane + static_cast<size_t>(v) * W + u] = row[3 * u + c];
        }
    }

    // Disparity in source-image pixels at source resolution
    cv::Mat infer(const cv::Mat& L, const cv::Mat& R) {
        const Placement p = place(L.size());
        toInput(L, p, kLeft);
        toInput(R, p, kRight);
        request.infer();
        const cv::Mat full(H, W, CV_32F, request.get_tensor(kDisp).data<float>());

        cv::Mat disp = full(cv::Rect(p.pad_x, p.pad_y, p.sw, p.sh)).clone();
        if (p.sw != L.cols || p.sh != L.rows) {
            cv::resize(disp, disp, L.size(), 0, 0, cv::INTER_NEAREST);
            disp *= static_cast<double>(L.cols) / p.sw;
        }
        return disp;
    }
};

LAS2StereoMatcher::LAS2StereoMatcher(const StereoMatcherConfig& cfg)
    : _impl(std::make_unique<Impl>(cfg)), _lr_check_px(cfg.las2_lr_check_px) {
    // First inference allocates the CPU plugin's scratch memory; pay it before the first keyframe
    const cv::Mat blank(_impl->H, _impl->W, CV_8UC1, cv::Scalar(0));
    _impl->infer(blank, blank);
    const int threads = _impl->compiled.get_property(ov::inference_num_threads);
    const ov::element::Type prec = _impl->compiled.get_property(ov::hint::inference_precision);
    std::cout << "[LAS2] model " << cfg.las2_onnx_path << " (" << _impl->W << "x" << _impl->H << ") on "
              << cfg.las2_device << ", precision " << prec << ", " << threads << " threads, left-right check "
              << (_lr_check_px > 0 ? std::to_string(_lr_check_px) + " px" : "off") << std::endl;
}

LAS2StereoMatcher::~LAS2StereoMatcher() = default;

cv::Mat LAS2StereoMatcher::compute(const cv::Mat& rect_L, const cv::Mat& rect_R) {
    cv::Mat disp = _impl->infer(rect_L, rect_R);

    // Mirrored pair (flipped right as reference) gives the right-image disparity for the consistency check
    cv::Mat disp_R;
    if (_lr_check_px > 0) {
        cv::Mat fL, fR;
        cv::flip(rect_L, fL, 1);
        cv::flip(rect_R, fR, 1);
        cv::flip(_impl->infer(fR, fL), disp_R, 1);
    }

    for (int v = 0; v < disp.rows; ++v) {
        float* d = disp.ptr<float>(v);
        const float* dr = disp_R.empty() ? nullptr : disp_R.ptr<float>(v);
        for (int u = 0; u < disp.cols; ++u) {
            const float ur = u - d[u];
            if (!(d[u] > 0.f) || ur < 0.f) { // match left of the right image
                d[u] = -1.f;
                continue;
            }
            if (dr) {
                const int iu = static_cast<int>(std::lround(ur));
                if (iu >= disp.cols || std::abs(d[u] - dr[iu]) > _lr_check_px)
                    d[u] = -1.f;
            }
        }
    }
    return disp;
}

} // namespace isae

#endif // ISAESLAM_WITH_LAS2
