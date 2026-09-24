// Compiled to nothing unless the library is built with -DISAESLAM_WITH_FFS=ON (cmake/ISAESLAM_FFS.cmake).
#ifdef ISAESLAM_WITH_FFS

#include "isaeslam/stereo/FFSStereoMatcher.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace isae {

namespace {

class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[FFS][TRT] " << msg << std::endl;
    }
};

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("[FFS] ") + what + ": " + cudaGetErrorString(err));
}

// Names and normalisation of the upstream single-ONNX export (scripts/make_single_onnx.py)
constexpr const char* kLeft  = "left_image";
constexpr const char* kRight = "right_image";
constexpr const char* kDisp  = "disparity";
constexpr float kMean[3] = {123.675f, 116.28f, 103.53f}; // RGB, 0-255 scale
constexpr float kStd[3]  = {58.395f, 57.12f, 57.375f};

} // namespace

struct FFSStereoMatcher::Impl {
    TrtLogger                                    logger;
    std::unique_ptr<nvinfer1::IRuntime>          runtime;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    void *d_left = nullptr, *d_right = nullptr, *d_disp = nullptr;
    int H = 0, W = 0;
    std::vector<float> h_left, h_right;

    // Image placement inside the engine input: uniform downscale (never up) then centred replicate padding
    struct Placement {
        double scale;
        int sw, sh, pad_x, pad_y;
    };

    explicit Impl(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f.good())
            throw std::runtime_error("[FFS] cannot open TensorRT engine " + path);
        std::vector<char> blob(static_cast<size_t>(f.tellg()));
        f.seekg(0);
        f.read(blob.data(), static_cast<std::streamsize>(blob.size()));

        runtime.reset(nvinfer1::createInferRuntime(logger));
        engine.reset(runtime ? runtime->deserializeCudaEngine(blob.data(), blob.size()) : nullptr);
        if (!engine)
            throw std::runtime_error("[FFS] cannot deserialize " + path +
                                     " (engine must be built by this TensorRT version on this GPU)");
        context.reset(engine->createExecutionContext());
        if (!context)
            throw std::runtime_error("[FFS] cannot create an execution context for " + path);

        for (const char* name : {kLeft, kRight, kDisp})
            if (engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
                throw std::runtime_error(std::string("[FFS] engine tensor '") + name +
                                         "' missing or not float32 (expected the single-ONNX export)");
        const nvinfer1::Dims in = engine->getTensorShape(kLeft);
        if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3)
            throw std::runtime_error("[FFS] expected a 1x3xHxW static input");
        H = static_cast<int>(in.d[2]);
        W = static_cast<int>(in.d[3]);

        const size_t in_bytes = 3 * static_cast<size_t>(H) * W * sizeof(float);
        check(cudaStreamCreate(&stream), "cudaStreamCreate");
        check(cudaMalloc(&d_left, in_bytes), "cudaMalloc left");
        check(cudaMalloc(&d_right, in_bytes), "cudaMalloc right");
        check(cudaMalloc(&d_disp, static_cast<size_t>(H) * W * sizeof(float)), "cudaMalloc disparity");
        context->setTensorAddress(kLeft, d_left);
        context->setTensorAddress(kRight, d_right);
        context->setTensorAddress(kDisp, d_disp);
        h_left.resize(3 * static_cast<size_t>(H) * W);
        h_right.resize(h_left.size());
    }

    ~Impl() {
        for (void* p : {d_left, d_right, d_disp})
            if (p)
                cudaFree(p);
        if (stream)
            cudaStreamDestroy(stream);
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

    void toInput(const cv::Mat& img, const Placement& p, std::vector<float>& chw) const {
        if (img.depth() != CV_8U)
            throw std::runtime_error("[FFS] expects 8-bit images");
        cv::Mat rgb;
        if (img.channels() == 1)
            cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
        else
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
        if (p.sw != img.cols || p.sh != img.rows)
            cv::resize(rgb, rgb, cv::Size(p.sw, p.sh), 0, 0, cv::INTER_LINEAR);
        cv::copyMakeBorder(rgb, rgb, p.pad_y, H - p.sh - p.pad_y, p.pad_x, W - p.sw - p.pad_x,
                           cv::BORDER_REPLICATE);
        const size_t plane = static_cast<size_t>(H) * W;
        for (int v = 0; v < H; ++v) {
            const uint8_t* row = rgb.ptr<uint8_t>(v);
            for (int u = 0; u < W; ++u)
                for (int c = 0; c < 3; ++c)
                    chw[c * plane + static_cast<size_t>(v) * W + u] = (row[3 * u + c] - kMean[c]) / kStd[c];
        }
    }

    // Disparity in source-image pixels at source resolution
    cv::Mat infer(const cv::Mat& L, const cv::Mat& R) {
        const Placement p = place(L.size());
        toInput(L, p, h_left);
        toInput(R, p, h_right);
        const size_t in_bytes = h_left.size() * sizeof(float);
        check(cudaMemcpyAsync(d_left, h_left.data(), in_bytes, cudaMemcpyHostToDevice, stream), "upload left");
        check(cudaMemcpyAsync(d_right, h_right.data(), in_bytes, cudaMemcpyHostToDevice, stream), "upload right");
        if (!context->enqueueV3(stream))
            throw std::runtime_error("[FFS] TensorRT enqueue failed");
        cv::Mat full(H, W, CV_32F);
        check(cudaMemcpyAsync(full.data, d_disp, static_cast<size_t>(H) * W * sizeof(float),
                              cudaMemcpyDeviceToHost, stream), "download disparity");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        cv::Mat disp = full(cv::Rect(p.pad_x, p.pad_y, p.sw, p.sh)).clone();
        if (p.sw != L.cols || p.sh != L.rows) {
            cv::resize(disp, disp, L.size(), 0, 0, cv::INTER_NEAREST);
            disp *= static_cast<double>(L.cols) / p.sw;
        }
        return disp;
    }
};

FFSStereoMatcher::FFSStereoMatcher(const StereoMatcherConfig& cfg)
    : _impl(std::make_unique<Impl>(cfg.ffs_engine_path)), _lr_check_px(cfg.ffs_lr_check_px) {
    // First enqueue is slow (lazy CUDA module loading); pay it before the first keyframe
    const cv::Mat blank(_impl->H, _impl->W, CV_8UC1, cv::Scalar(0));
    _impl->infer(blank, blank);
    std::cout << "[FFS] engine " << cfg.ffs_engine_path << " (" << _impl->W << "x" << _impl->H
              << "), left-right check " << (_lr_check_px > 0 ? std::to_string(_lr_check_px) + " px" : "off")
              << std::endl;
}

FFSStereoMatcher::~FFSStereoMatcher() = default;

cv::Mat FFSStereoMatcher::compute(const cv::Mat& rect_L, const cv::Mat& rect_R) {
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
            if (!(d[u] > 0.f) || ur < 0.f) { // upstream remove_invisible: match left of the right image
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

#endif // ISAESLAM_WITH_FFS
