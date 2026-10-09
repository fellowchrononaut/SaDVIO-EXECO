#include "isaeslam/loopclosure/GlobalDescriptor.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#ifdef ISAESLAM_VPR_OPENVINO
#include <openvino/openvino.hpp>
#endif
#ifdef ISAESLAM_VPR_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace isae {

Eigen::VectorXf GlobalDescriptor::describe(const cv::Mat &img) {
    cv::Mat gray = img;
    if (gray.depth() == CV_16U)
        gray.convertTo(gray, CV_8U, 1.0 / 256);
    if (gray.channels() == 3)
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
    cv::Mat r;
    cv::resize(gray, r, cv::Size(_W, _H), 0, 0, cv::INTER_CUBIC);

    // RGB normalised with the ImageNet statistics; a grey image has the same value on the three channels
    static const float mean[3] = {0.485f, 0.456f, 0.406f}, stdev[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> chw(3 * static_cast<size_t>(_H) * _W);
    for (int c = 0; c < 3; c++) {
        float *dst = chw.data() + static_cast<size_t>(c) * _H * _W;
        for (int y = 0; y < _H; y++) {
            const uchar *src = r.ptr<uchar>(y);
            for (int x = 0; x < _W; x++)
                dst[y * _W + x] = (src[x] / 255.f - mean[c]) / stdev[c];
        }
    }
    std::vector<float> out;
    infer(chw, out);
    Eigen::VectorXf d = Eigen::Map<Eigen::VectorXf>(out.data(), static_cast<Eigen::Index>(out.size()));
    const float n     = d.norm();
    return n > 0 ? Eigen::VectorXf(d / n) : d;
}

#ifdef ISAESLAM_VPR_OPENVINO
namespace {
class OpenVinoDescriptor : public GlobalDescriptor {
  public:
    OpenVinoDescriptor(const std::string &path, int threads) {
        std::shared_ptr<ov::Model> model = core.read_model(path);
        const ov::Shape in               = model->input().get_shape();
        if (in.size() != 4 || in[0] != 1 || in[1] != 3)
            throw std::runtime_error("loop closure model " + path + ": expected an input [1, 3, H, W]");
        _H = static_cast<int>(in[2]);
        _W = static_cast<int>(in[3]);
        _D = static_cast<int>(ov::shape_size(model->output().get_shape()));
        ov::AnyMap props{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)};
        if (threads > 0)
            props.insert(ov::inference_num_threads(threads));
        compiled = core.compile_model(model, "CPU", props);
        request  = compiled.create_infer_request();
    }

  protected:
    void infer(const std::vector<float> &chw, std::vector<float> &out) override {
        ov::Tensor input(ov::element::f32, ov::Shape{1, 3, static_cast<size_t>(_H), static_cast<size_t>(_W)},
                         const_cast<float *>(chw.data()));
        request.set_input_tensor(input);
        request.infer();
        const ov::Tensor o = request.get_output_tensor();
        out.assign(o.data<float>(), o.data<float>() + o.get_size());
    }

  private:
    ov::Core core;
    ov::CompiledModel compiled;
    ov::InferRequest request;
};
} // namespace
#endif

#ifdef ISAESLAM_VPR_TENSORRT
namespace {
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity s, const char *msg) noexcept override {
        if (s <= Severity::kWARNING)
            std::cerr << "[loop closure TensorRT] " << msg << std::endl;
    }
};

void cudaCheck(cudaError_t e, const char *what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("loop closure model: ") + what + ": " + cudaGetErrorString(e));
}

class TensorRtDescriptor : public GlobalDescriptor {
  public:
    TensorRtDescriptor(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        runtime.reset(nvinfer1::createInferRuntime(logger));
        engine.reset(runtime ? runtime->deserializeCudaEngine(blob.data(), blob.size()) : nullptr);
        if (!engine)
            throw std::runtime_error("loop closure model: cannot deserialize the TensorRT engine " + path);
        context.reset(engine->createExecutionContext());
        for (int k = 0; k < engine->getNbIOTensors(); k++) {
            const char *name = engine->getIOTensorName(k);
            (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? in_name : out_name) = name;
        }
        const nvinfer1::Dims in = engine->getTensorShape(in_name.c_str());
        const nvinfer1::Dims o  = engine->getTensorShape(out_name.c_str());
        if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3)
            throw std::runtime_error("loop closure model " + path + ": expected an input [1, 3, H, W]");
        _H = static_cast<int>(in.d[2]);
        _W = static_cast<int>(in.d[3]);
        _D = 1;
        for (int k = 0; k < o.nbDims; k++)
            _D *= static_cast<int>(o.d[k]);
        cudaCheck(cudaStreamCreate(&stream), "stream");
        cudaCheck(cudaMalloc(&d_in, 3 * sizeof(float) * _H * _W), "cudaMalloc input");
        cudaCheck(cudaMalloc(&d_out, sizeof(float) * _D), "cudaMalloc output");
        context->setTensorAddress(in_name.c_str(), d_in);
        context->setTensorAddress(out_name.c_str(), d_out);
    }
    ~TensorRtDescriptor() override {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaStreamDestroy(stream);
    }

  protected:
    void infer(const std::vector<float> &chw, std::vector<float> &out) override {
        out.resize(_D);
        cudaCheck(cudaMemcpyAsync(d_in, chw.data(), chw.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
                  "upload");
        if (!context->enqueueV3(stream))
            throw std::runtime_error("loop closure model: TensorRT enqueue failed");
        cudaCheck(cudaMemcpyAsync(out.data(), d_out, _D * sizeof(float), cudaMemcpyDeviceToHost, stream), "download");
        cudaCheck(cudaStreamSynchronize(stream), "synchronize");
    }

  private:
    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::string in_name, out_name;
    cudaStream_t stream = nullptr;
    void *d_in = nullptr, *d_out = nullptr;
};
} // namespace
#endif

std::unique_ptr<GlobalDescriptor> GlobalDescriptor::create(const std::string &model, const std::string &device,
                                                           int threads) {
    (void)threads; // OpenVINO only
    if (device == "CPU") {
#ifdef ISAESLAM_VPR_OPENVINO
        return std::make_unique<OpenVinoDescriptor>(model, threads);
#else
        throw std::runtime_error("loop closure: a CPU model needs the build option ISAESLAM_WITH_VPR_OPENVINO");
#endif
    }
    if (device == "GPU") {
#ifdef ISAESLAM_VPR_TENSORRT
        return std::make_unique<TensorRtDescriptor>(model);
#else
        throw std::runtime_error("loop closure: a GPU model needs the build option ISAESLAM_WITH_VPR_TENSORRT");
#endif
    }
    throw std::runtime_error("loop closure: loop_model_device must be CPU or GPU, not " + device);
}

} // namespace isae
