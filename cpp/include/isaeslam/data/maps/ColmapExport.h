#ifndef COLMAPEXPORT_H
#define COLMAPEXPORT_H

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace isae {

class Frame;
class ImageSensor;
class LoopClosure;
struct cam_config;

/*!
 * @brief COLMAP text export of the final map (config export_colmap; doc/nofovvio/NOFOVVIO_LEDGER.md, item 5.1)
 *
 * Each keyframe is recorded when it leaves the sliding window (its final window estimate, before its features are
 * cleaned): its pose, and per camera the pixels of its features that hold an initialized landmark. At the end of the
 * run, write() produces in <dir>/:
 *   cameras.txt   one camera per sensor of the rig, at the dataset's original resolution, COLMAP pixel centres (+0.5):
 *                 pinhole radial-tangential as OPENCV (the original, distorted image; feature pixels mapped back from
 *                 the undistorted image), kb4 as OPENCV_FISHEYE; models COLMAP lacks (double sphere, equidistant,
 *                 omni) as an OPENCV_FISHEYE fitted to the camera's rays, with a message saying so and the fit's error
 *   images.txt    one image per keyframe and camera, named camN/data/<timestamp>.png (EuRoC layout), pose world to
 *                 camera from the loop-corrected keyframe pose when the loop closure ran, else the window's, with its
 *                 2D points
 *   points3D.txt  landmarks seen by at least 2 recorded images, moved by the loop correction of the last keyframe
 *                 that saw them
 *   rigs.txt, frames.txt   with 2 or more cameras (COLMAP 3.12+ / 4 text format): one rig, camera 1 as reference,
 *                 the others with their camera-from-rig poses; one frame per keyframe
 */
class ColmapExport {
  public:
    //! @param downsampling the working resolution's scale (feature pixels are at it)
    ColmapExport(const std::vector<std::shared_ptr<cam_config>> &cams, double downsampling);

    //! Record a keyframe leaving the window (each keyframe once)
    void record(const std::shared_ptr<Frame> &kf);

    //! Write the COLMAP files to dir (created); lc may be null (no loop closure)
    void write(const std::string &dir, const std::shared_ptr<LoopClosure> &lc);

  private:
    struct Obs {
        double u, v;   // pixel at the working resolution
        int lmk;       // landmark id
    };
    struct KF {
        unsigned long long ts;
        Eigen::Affine3d T_w_f;                // window estimate at hand-over (frame to world)
        std::vector<Eigen::Affine3d> T_s_f;   // per camera: frame to camera
        std::vector<std::vector<Obs>> obs;    // per camera
    };

    //! COLMAP model name and parameters of camera c (original resolution, COLMAP pixel centres)
    void cameraModel(size_t c, std::string &model, std::vector<double> &params);

    //! A feature pixel of camera c (working resolution, SaDVIO pixel centres) in the exported image and model
    Eigen::Vector2d exportPixel(size_t c, double u, double v) const;

    std::vector<std::shared_ptr<cam_config>> _cams;
    double _downsampling;
    std::vector<std::shared_ptr<ImageSensor>> _sensors; // one sensor per camera (rays for the fitted models)
    std::vector<KF> _kfs;
    std::map<int, Eigen::Vector3d> _lmk_pos;            // latest world position of each landmark seen
    std::map<int, unsigned long long> _lmk_last_kf;     // last keyframe that saw it
    unsigned long long _last_ts = 0;
    std::vector<std::string> _notes; // messages written to README.txt (fitted camera models)
    std::mutex _mtx;
};

} // namespace isae

#endif // COLMAPEXPORT_H
