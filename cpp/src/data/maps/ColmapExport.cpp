#include "isaeslam/data/maps/ColmapExport.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/loopclosure/LoopClosure.h"

namespace isae {

namespace {

// Rotation as COLMAP's quaternion order: w x y z
void writePose(std::ostream &o, const Eigen::Affine3d &T) {
    const Eigen::Quaterniond q(T.rotation());
    o << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << T.translation().x() << " "
      << T.translation().y() << " " << T.translation().z();
}

} // namespace

ColmapExport::ColmapExport(const std::vector<std::shared_ptr<cam_config>> &cams, double downsampling)
    : _cams(cams), _downsampling(downsampling) {}

void ColmapExport::record(const std::shared_ptr<Frame> &kf) {
    std::lock_guard<std::mutex> lock(_mtx);
    if (!kf || kf->getSensors().empty() || kf->getTimestamp() <= _last_ts)
        return;
    _last_ts = kf->getTimestamp();
    if (_sensors.empty())
        _sensors = kf->getSensors();

    KF r;
    r.ts    = kf->getTimestamp();
    r.T_w_f = kf->getFrame2WorldTransform();
    for (auto &cam : kf->getSensors()) {
        r.T_s_f.push_back(cam->getFrame2SensorTransform());
        std::vector<Obs> obs;
        for (auto &feat : cam->getFeatures("pointxd")) {
            std::shared_ptr<ALandmark> lmk = feat->getLandmark().lock();
            if (!lmk || !lmk->isInitialized() || lmk->isOutlier() || feat->getPoints().empty())
                continue;
            const Eigen::Vector2d px = feat->getPoints().at(0);
            obs.push_back({px.x(), px.y(), lmk->_id});
            _lmk_pos[lmk->_id]     = lmk->getPose().translation();
            _lmk_last_kf[lmk->_id] = r.ts;
        }
        r.obs.push_back(std::move(obs));
    }
    _kfs.push_back(std::move(r));
}

Eigen::Vector2d ColmapExport::exportPixel(size_t c, double u, double v) const {
    // Working resolution -> original image (pixel centres at integers, as OpenCV's resize maps them)
    const Eigen::Vector2d p((u + 0.5) / _downsampling - 0.5, (v + 0.5) / _downsampling - 0.5);
    const cam_config &cfg = *_cams.at(c);
    if (cfg.proj_model == "pinhole" && cfg.undistort) {
        // The SLAM saw the undistorted image (K_new): back to the original, distorted image (K_raw, radial-tangential)
        const double x = (p.x() - cfg.K(0, 2)) / cfg.K(0, 0), y = (p.y() - cfg.K(1, 2)) / cfg.K(1, 1);
        const double k1 = cfg.d(0), k2 = cfg.d(1), p1 = cfg.d(2), p2 = cfg.d(3);
        const double r2 = x * x + y * y, radial = 1 + k1 * r2 + k2 * r2 * r2;
        const double xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
        const double yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
        return Eigen::Vector2d(cfg.K_raw(0, 0) * xd + cfg.K_raw(0, 2), cfg.K_raw(1, 1) * yd + cfg.K_raw(1, 2)) +
               Eigen::Vector2d(0.5, 0.5);
    }
    return p + Eigen::Vector2d(0.5, 0.5); // COLMAP: pixel centres at +0.5
}

void ColmapExport::cameraModel(size_t c, std::string &model, std::vector<double> &params) {
    const cam_config &cfg = *_cams.at(c);
    if (cfg.proj_model == "pinhole") {
        const Eigen::Matrix3d &K = cfg.undistort ? cfg.K_raw : cfg.K;
        params = {K(0, 0), K(1, 1), K(0, 2) + 0.5, K(1, 2) + 0.5};
        if (cfg.undistort) {
            model = "OPENCV";
            params.insert(params.end(), {cfg.d(0), cfg.d(1), cfg.d(2), cfg.d(3)});
        } else {
            model = "PINHOLE";
        }
        return;
    }
    model = "OPENCV_FISHEYE";
    const Eigen::Matrix3d &K = cfg.K;
    if (cfg.proj_model == "kb4") {
        params = {K(0, 0), K(1, 1), K(0, 2) + 0.5, K(1, 2) + 0.5, cfg.d(0), cfg.d(1), cfg.d(2), cfg.d(3)};
        return;
    }

    // COLMAP has no model for this camera: fit an OPENCV_FISHEYE (theta_d = theta (1 + k1 theta^2 + ... + k4 theta^8))
    // to the camera's own rays over a grid of its pixels. The principal point and the aspect ratio stay the camera's;
    // the focal length and k1..k4 come from a linear least-squares fit of the radius (in x pixels),
    // rho = F theta + F k1 theta^3 + F k2 theta^5 + F k3 theta^7 + F k4 theta^9
    std::vector<double> theta, rho;
    std::shared_ptr<ImageSensor> s = c < _sensors.size() ? _sensors[c] : nullptr;
    const double cx = K(0, 2), cy = K(1, 2), aspect = K(0, 0) / K(1, 1);
    if (s) {
        const int w = static_cast<int>(std::round(cfg.width * _downsampling));
        const int h = static_cast<int>(std::round(cfg.height * _downsampling));
        for (int v = 0; v < h; v += std::max(1, h / 60))
            for (int u = 0; u < w; u += std::max(1, w / 60)) {
                const Eigen::Vector3d ray = s->getRayCamera(Eigen::Vector2d(u, v));
                if (!ray.allFinite() || ray.norm() < 1e-9)
                    continue;
                const double th = std::atan2(ray.head<2>().norm(), ray.z());
                const Eigen::Vector2d p((u + 0.5) / _downsampling - 0.5, (v + 0.5) / _downsampling - 0.5);
                const double r = std::hypot(p.x() - cx, (p.y() - cy) * aspect);
                // Up to 89 deg: COLMAP projects through x / z
                if (th < 1e-3 || th > 89 * M_PI / 180 || !std::isfinite(r))
                    continue;
                theta.push_back(th);
                rho.push_back(r);
            }
    }
    Eigen::Vector4d k = Eigen::Vector4d::Zero();
    double F = K(0, 0), rms = -1;
    if (theta.size() >= 10) {
        Eigen::MatrixXd A(theta.size(), 5);
        Eigen::VectorXd b(theta.size());
        for (size_t i = 0; i < theta.size(); i++) {
            const double t = theta[i], t2 = t * t;
            A.row(i) << t, t * t2, t * t2 * t2, t * t2 * t2 * t2, t * t2 * t2 * t2 * t2;
            b(i) = rho[i];
        }
        const Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
        F                       = x(0);
        k                       = x.tail<4>() / F;
        rms                     = std::sqrt((A * x - b).squaredNorm() / theta.size());
    }
    params = {F, F / aspect, cx + 0.5, cy + 0.5};
    params.insert(params.end(), {k(0), k(1), k(2), k(3)});
    std::ostringstream msg;
    msg << "COLMAP export: camera " << c + 1 << " (" << cfg.proj_model
        << ") has no COLMAP model; written as OPENCV_FISHEYE (Kannala-Brandt: focal length and k1..k4) fitted to its rays over "
        << theta.size() << " pixels, RMS " << std::setprecision(3) << rms
        << " px. The 2D points are the observed pixels; the fitted model approximates the camera.";
    std::cout << msg.str() << std::endl;
    _notes.push_back(msg.str());
}

void ColmapExport::write(const std::string &dir, const std::shared_ptr<LoopClosure> &lc) {
    std::lock_guard<std::mutex> lock(_mtx);
    std::filesystem::create_directories(dir);
    const size_t ncam = _cams.size();
    _notes.clear();

    // Final keyframe poses: loop-corrected when the loop closure ran
    std::vector<Eigen::Affine3d> T_w_f(_kfs.size());
    std::map<unsigned long long, Eigen::Affine3d> correction;
    for (size_t k = 0; k < _kfs.size(); k++) {
        Eigen::Affine3d C = Eigen::Affine3d::Identity();
        if (lc && !lc->keyframeCorrection(_kfs[k].ts, C))
            C = Eigen::Affine3d::Identity();
        correction[_kfs[k].ts] = C;
        T_w_f[k]               = C * _kfs[k].T_w_f;
    }

    // Landmark positions, moved by the correction of the last keyframe that saw them
    std::map<int, Eigen::Vector3d> X;
    for (auto &l : _lmk_pos) {
        auto it    = correction.find(_lmk_last_kf[l.first]);
        X[l.first] = (it != correction.end() ? it->second : Eigen::Affine3d::Identity()) * l.second;
    }

    // Tracks: an observation joins its landmark's track if the landmark is in front of that camera: COLMAP's models
    // project through x / z (its OPENCV_FISHEYE takes atan(r) of it), so a fisheye's rays beyond 90 deg from the axis
    // cannot be represented (kept as 2D points without a 3D point). Landmarks seen by at least 2 images are exported
    std::map<int, std::vector<std::pair<int, int>>> tracks; // landmark -> (image id, point2D index)
    std::vector<std::vector<std::vector<bool>>> linked(_kfs.size());
    size_t n_behind = 0;
    for (size_t k = 0; k < _kfs.size(); k++) {
        linked[k].resize(_kfs[k].obs.size());
        for (size_t c = 0; c < _kfs[k].obs.size(); c++) {
            const Eigen::Affine3d T_c_w = _kfs[k].T_s_f[c] * T_w_f[k].inverse();
            for (size_t j = 0; j < _kfs[k].obs[c].size(); j++) {
                const auto it = X.find(_kfs[k].obs[c][j].lmk);
                bool ok       = it != X.end();
                if (ok) {
                    const Eigen::Vector3d p = T_c_w * it->second;
                    ok                      = p.z() > std::cos(89.0 * M_PI / 180) * p.norm();
                    n_behind += ok ? 0 : 1;
                }
                linked[k][c].push_back(ok);
                if (ok)
                    tracks[_kfs[k].obs[c][j].lmk].push_back({static_cast<int>(k * ncam + c + 1), static_cast<int>(j)});
            }
        }
    }
    std::set<int> exported;
    for (auto &t : tracks)
        if (t.second.size() >= 2)
            exported.insert(t.first);

    // cameras.txt
    std::ofstream fc(dir + "/cameras.txt");
    fc << std::setprecision(12) << "# Camera list with one line of data per camera:\n"
       << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: " << ncam << "\n";
    for (size_t c = 0; c < ncam; c++) {
        std::string model;
        std::vector<double> params;
        cameraModel(c, model, params);
        fc << c + 1 << " " << model << " " << _cams[c]->width << " " << _cams[c]->height;
        for (double p : params)
            fc << " " << p;
        fc << "\n";
    }

    // images.txt
    std::ofstream fi(dir + "/images.txt");
    fi << std::setprecision(12) << "# Image list with two lines of data per image:\n"
       << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
       << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n# Number of images: " << _kfs.size() * ncam << "\n";
    for (size_t k = 0; k < _kfs.size(); k++) {
        for (size_t c = 0; c < ncam && c < _kfs[k].T_s_f.size(); c++) {
            fi << k * ncam + c + 1 << " ";
            writePose(fi, _kfs[k].T_s_f[c] * T_w_f[k].inverse());
            fi << " " << c + 1 << " cam" << c << "/data/" << _kfs[k].ts << ".png\n";
            for (size_t j = 0; j < _kfs[k].obs[c].size(); j++) {
                const Obs &o            = _kfs[k].obs[c][j];
                const Eigen::Vector2d p = exportPixel(c, o.u, o.v);
                fi << (j ? " " : "") << p.x() << " " << p.y() << " "
                   << (linked[k][c][j] && exported.count(o.lmk) ? o.lmk + 1 : -1);
            }
            fi << "\n";
        }
    }

    // points3D.txt: moved by the correction of the last keyframe that saw them
    std::ofstream fp(dir + "/points3D.txt");
    fp << std::setprecision(12) << "# 3D point list with one line of data per point:\n"
       << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
       << "# Number of points: " << exported.size() << "\n";
    for (int id : exported) {
        const Eigen::Vector3d &x = X[id];
        fp << id + 1 << " " << x.x() << " " << x.y() << " " << x.z() << " 128 128 128 0";
        for (auto &t : tracks[id])
            fp << " " << t.first << " " << t.second;
        fp << "\n";
    }

    // rigs.txt, frames.txt: the cameras of a keyframe as one rig (camera 1 as reference)
    if (ncam >= 2 && !_kfs.empty()) {
        std::ofstream fr(dir + "/rigs.txt");
        fr << std::setprecision(12) << "# Rig calib list with one line of data per calib:\n"
           << "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, SENSOR_ID, HAS_POSE, "
              "[QW, QX, QY, QZ, TX, TY, TZ])\n# Number of rigs: 1\n"
           << "1 " << ncam << " CAMERA 1";
        const Eigen::Affine3d T_c0_f = _kfs[0].T_s_f[0];
        for (size_t c = 1; c < ncam; c++) {
            fr << " CAMERA " << c + 1 << " 1 ";
            writePose(fr, _kfs[0].T_s_f[c] * T_c0_f.inverse()); // camera c from rig (= camera 1)
        }
        fr << "\n";
        std::ofstream ff(dir + "/frames.txt");
        ff << std::setprecision(12) << "# Frame list with one line of data per frame:\n"
           << "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, DATA_IDS[] as "
              "(SENSOR_TYPE, SENSOR_ID, DATA_ID)\n# Number of frames: "
           << _kfs.size() << "\n";
        for (size_t k = 0; k < _kfs.size(); k++) {
            ff << k + 1 << " 1 ";
            writePose(ff, _kfs[k].T_s_f[0] * T_w_f[k].inverse());
            ff << " " << ncam;
            for (size_t c = 0; c < ncam; c++)
                ff << " CAMERA " << c + 1 << " " << k * ncam + c + 1;
            ff << "\n";
        }
    }

    std::ofstream fn(dir + "/README.txt");
    fn << "SaDVIO COLMAP export: " << _kfs.size() << " keyframes, " << ncam << " camera(s), " << exported.size()
       << " points (landmarks seen by >= 2 images).\n" << n_behind
       << " observations beyond 90 deg from their camera's axis are 2D points without a 3D point (COLMAP's camera "
          "models project through x / z).\nPoses: " << (lc ? "loop-corrected" : "sliding-window")
       << " keyframe poses. Images: camN/data/<timestamp>.png of the dataset (EuRoC layout), original resolution, "
          "COLMAP pixel centres (+0.5).\n";
    for (const std::string &n : _notes)
        fn << n << "\n";
    std::cout << "COLMAP export: " << _kfs.size() << " keyframes, " << exported.size() << " points in " << dir
              << std::endl;
}

} // namespace isae
