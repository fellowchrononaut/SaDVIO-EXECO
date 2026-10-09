#ifndef GLOBALDESCRIPTOR_H
#define GLOBALDESCRIPTOR_H

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace isae {

/*!
 * @brief Learned global image descriptor for loop detection (e.g. MegaLoc exported to ONNX by
 * doc/loop_closure/tools/export_megaloc_onnx.py).
 *
 * The model takes one RGB image [1, 3, H, W] normalised with the ImageNet mean and standard deviation and returns
 * one descriptor [1, D]. The input size is read from the model: images are resized to it (grey images are repeated
 * on the three channels). Backends, each compiled only with its build option (cmake/ISAESLAM_VPR.cmake):
 *   - "CPU": OpenVINO, from the .onnx (or OpenVINO IR) file (ISAESLAM_WITH_VPR_OPENVINO)
 *   - "GPU": TensorRT, from an engine built with trtexec from the .onnx file (ISAESLAM_WITH_VPR_TENSORRT)
 */
class GlobalDescriptor {
  public:
    /*!
     * @brief Load a model for a device; throws if the backend is not compiled in or the model does not fit
     */
    static std::unique_ptr<GlobalDescriptor> create(const std::string &model, const std::string &device, int threads);
    virtual ~GlobalDescriptor() = default;

    /*!
     * @brief L2-normalised descriptor of an 8-bit grey or BGR image
     */
    Eigen::VectorXf describe(const cv::Mat &img);

    int dim() const { return _D; }
    int height() const { return _H; }
    int width() const { return _W; }

  protected:
    virtual void infer(const std::vector<float> &chw, std::vector<float> &out) = 0;
    int _H = 0, _W = 0, _D = 0;
};

} // namespace isae

#endif // GLOBALDESCRIPTOR_H
