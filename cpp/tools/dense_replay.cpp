// Offline replay of the dense map from a keyframe dump (EXECO_DENSE_DUMP, MarginalDepthInjector::dumpKeyframe), to
// probe the fusion without re-running the SLAM (doc/loop_closure/LOOP_CLOSURE_LEDGER.md, long-horizon TSDF fusion).
// The keyframes are integrated into VDB volumes at chosen poses: their odometry poses (from the dump), loop-corrected
// poses (results_loop.csv), or ground-truth poses (an EuRoC state_groundtruth_estimate0/data.csv; analysis only, never
// in the SLAM). Volumes: one for the whole run (--chunk 0), or chunks of N keyframes, each in its first keyframe's
// frame, written as a union of layers or merged into one volume (--merge). Writes the mesh in the poses' world frame.
//
// usage: dense_replay <dump dir> <out.ply> --preset <vdbgpdf preset yaml> [--poses odometry|loop:<results_loop.csv>|
//        gt:<data.csv>] [--chunk N] [--merge] [--range m] [--carve 0|1] [--trunc m] [--voxel m] [--min-weight w]
//        [--range-weight ref_m] [--range-exp e] [--max-kf N] [--every K]
#include "isaeslam/stereo/DenseMesh.h"
#include "isaeslam/stereo/VDBGPDFMap.h"

#include <Eigen/Geometry>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace isae;

namespace {

struct Keyframe {
    unsigned long long ts;
    int segment, epoch;
    Eigen::Affine3d T_w_rectcam, T_w_f;
    std::vector<Eigen::Vector3d> pts; // rectified camera frame
};

bool readKeyframe(const std::string& path, Keyframe& k) {
    std::ifstream f(path, std::ios::binary);
    uint32_t n = 0;
    Eigen::Matrix4d A, B;
    f.read(reinterpret_cast<char*>(&k.ts), sizeof(k.ts));
    f.read(reinterpret_cast<char*>(&k.segment), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(&k.epoch), sizeof(int32_t));
    f.read(reinterpret_cast<char*>(A.data()), 16 * sizeof(double));
    f.read(reinterpret_cast<char*>(B.data()), 16 * sizeof(double));
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    std::vector<float> v(3 * static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(float));
    if (!f)
        return false;
    k.T_w_rectcam.matrix() = A;
    k.T_w_f.matrix()       = B;
    k.pts.resize(n);
    for (uint32_t i = 0; i < n; i++)
        k.pts[i] = Eigen::Vector3d(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
    return true;
}

// Frame poses by timestamp (ns): results_loop.csv (ts, nframes, 3x4 row-major T_wf) or EuRoC ground truth
// (ts, p, q wxyz, ...), interpolated linearly (position) and by slerp
struct PoseTable {
    std::vector<unsigned long long> ts;
    std::vector<Eigen::Vector3d> p;
    std::vector<Eigen::Quaterniond> q;
    bool lookup(unsigned long long t, Eigen::Affine3d& T, double max_gap_s) const {
        auto it = std::lower_bound(ts.begin(), ts.end(), t);
        if (it != ts.end() && *it == t) { // exact (keyframe poses by their own timestamps)
            const size_t i = it - ts.begin();
            T              = Eigen::Translation3d(p[i]) * q[i];
            return true;
        }
        if (it == ts.end() || it == ts.begin())
            return false;
        const size_t i1 = it - ts.begin(), i0 = i1 - 1;
        if ((ts[i1] - ts[i0]) * 1e-9 > max_gap_s)
            return false;
        const double a = double(t - ts[i0]) / double(ts[i1] - ts[i0]);
        T              = Eigen::Translation3d((1 - a) * p[i0] + a * p[i1]) * q[i0].slerp(a, q[i1]);
        return true;
    }
};

PoseTable readPoses(const std::string& path, bool gt) {
    PoseTable t;
    std::ifstream f(path);
    std::string line;
    std::map<unsigned long long, std::pair<Eigen::Vector3d, Eigen::Quaterniond>> rows; // last write per timestamp
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || !std::isdigit(static_cast<unsigned char>(line[0])))
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream s(line);
        unsigned long long ts;
        s >> ts;
        if (gt) {
            double px, py, pz, qw, qx, qy, qz;
            s >> px >> py >> pz >> qw >> qx >> qy >> qz;
            rows[ts] = {Eigen::Vector3d(px, py, pz), Eigen::Quaterniond(qw, qx, qy, qz).normalized()};
        } else {
            double nf, m[12];
            s >> nf;
            for (double& x : m)
                s >> x;
            Eigen::Matrix3d R;
            R << m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10];
            rows[ts] = {Eigen::Vector3d(m[3], m[7], m[11]), Eigen::Quaterniond(R).normalized()};
        }
    }
    for (auto& r : rows) {
        t.ts.push_back(r.first);
        t.p.push_back(r.second.first);
        t.q.push_back(r.second.second);
    }
    return t;
}

void append(DenseMesh& out, const DenseMesh& m, const Eigen::Affine3d& T) {
    const int base = static_cast<int>(out.vertices.size());
    for (const auto& v : m.vertices)
        out.vertices.push_back(T * v);
    for (const auto& f : m.faces)
        out.faces.push_back(f + Eigen::Vector3i::Constant(base));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: dense_replay <dump dir> <out.ply> --preset <yaml> [options]\n";
        return 1;
    }
    const std::string dump = argv[1], out = argv[2];
    std::string preset, poses = "odometry";
    int chunk = 0, max_kf = 0, every = 1;
    bool merge = false;
    double range = -1, trunc = -1, voxel = -1, min_weight = -1, rw_ref = -1, rw_exp = -1;
    int carve = -1;
    for (int i = 3; i < argc; i++) {
        const std::string a = argv[i];
        auto next           = [&]() { return std::string(argv[++i]); };
        if (a == "--preset")
            preset = next();
        else if (a == "--poses")
            poses = next();
        else if (a == "--chunk")
            chunk = std::stoi(next());
        else if (a == "--merge")
            merge = true;
        else if (a == "--range")
            range = std::stod(next());
        else if (a == "--carve")
            carve = std::stoi(next());
        else if (a == "--trunc")
            trunc = std::stod(next());
        else if (a == "--voxel")
            voxel = std::stod(next());
        else if (a == "--min-weight")
            min_weight = std::stod(next());
        else if (a == "--range-weight")
            rw_ref = std::stod(next());
        else if (a == "--range-exp")
            rw_exp = std::stod(next());
        else if (a == "--max-kf")
            max_kf = std::stoi(next());
        else if (a == "--every")
            every = std::stoi(next());
    }
    VDBGPDFMapConfig cfg;
    if (!preset.empty())
        loadVDBGPDFPreset(preset, cfg);
    if (range > 0)
        cfg.max_scan_range = static_cast<float>(range);
    if (carve >= 0)
        cfg.space_carving = carve == 1;
    if (trunc > 0)
        cfg.sdf_trunc = static_cast<float>(trunc);
    if (voxel > 0)
        cfg.voxel_size_global = static_cast<float>(voxel);
    if (min_weight >= 0)
        cfg.recon_min_weight = static_cast<float>(min_weight);
    if (rw_ref > 0)
        cfg.range_weight_ref = static_cast<float>(rw_ref);
    if (rw_exp > 0)
        cfg.range_weight_exp = static_cast<float>(rw_exp);

    std::vector<std::string> files;
    for (const auto& e : std::filesystem::directory_iterator(dump))
        if (e.path().extension() == ".bin")
            files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    std::vector<Keyframe> kfs;
    for (size_t i = 0; i < files.size(); i += std::max(1, every)) {
        Keyframe k;
        if (readKeyframe(files[i], k))
            kfs.push_back(std::move(k));
        if (max_kf > 0 && static_cast<int>(kfs.size()) >= max_kf)
            break;
    }
    std::sort(kfs.begin(), kfs.end(), [](const Keyframe& a, const Keyframe& b) { return a.ts < b.ts; });

    // Poses of the rectified cameras: odometry from the dump, or frame poses from a table times the dumped
    // frame-to-rectified-camera extrinsic
    PoseTable table;
    const bool from_table = poses != "odometry";
    if (from_table) {
        const bool gt = poses.rfind("gt:", 0) == 0;
        table         = readPoses(poses.substr(poses.find(':') + 1), gt);
    }
    std::vector<Eigen::Affine3d> T_w_c;
    std::vector<int> use;
    for (size_t i = 0; i < kfs.size(); i++) {
        Eigen::Affine3d T = kfs[i].T_w_rectcam;
        if (from_table) {
            Eigen::Affine3d T_w_f;
            if (!table.lookup(kfs[i].ts, T_w_f, 0.05))
                continue;
            T = T_w_f * (kfs[i].T_w_f.inverse() * kfs[i].T_w_rectcam);
        }
        T_w_c.push_back(T);
        use.push_back(static_cast<int>(i));
    }
    std::cout << "dense_replay: " << use.size() << " of " << kfs.size() << " keyframes posed (" << poses
              << "), fusion " << cfg.fusion << ", voxel " << cfg.voxel_size_global << " m, trunc " << cfg.sdf_trunc
              << " m, carve " << cfg.space_carving << ", range " << cfg.max_scan_range << " m, min weight "
              << cfg.recon_min_weight << ", range weight " << cfg.range_weight_ref << "^" << cfg.range_weight_exp
              << ", chunk " << chunk << (merge ? " merged" : "") << std::endl;

    DenseMesh mesh;
    if (chunk <= 0) {
        VDBGPDFMap vol(cfg);
        for (size_t j = 0; j < use.size(); j++) {
            std::vector<Eigen::Vector3d> pw;
            pw.reserve(kfs[use[j]].pts.size());
            for (const auto& p : kfs[use[j]].pts)
                pw.push_back(T_w_c[j] * p);
            vol.integrate(pw, T_w_c[j].translation());
        }
        mesh = vol.mesh();
    } else {
        std::unique_ptr<VDBGPDFMap> fused = merge ? std::make_unique<VDBGPDFMap>(cfg) : nullptr;
        for (size_t s = 0; s < use.size(); s += chunk) {
            // A chunk breaks at a re-initialization (unrelated worlds with odometry poses)
            VDBGPDFMap vol(cfg);
            const Eigen::Affine3d T_w_a = T_w_c[s];
            size_t j                    = s;
            for (; j < std::min(use.size(), s + chunk); j++) {
                if (!from_table && kfs[use[j]].segment != kfs[use[s]].segment)
                    break;
                const Eigen::Affine3d T_a_c = T_w_a.inverse() * T_w_c[j];
                std::vector<Eigen::Vector3d> pa;
                pa.reserve(kfs[use[j]].pts.size());
                for (const auto& p : kfs[use[j]].pts)
                    pa.push_back(T_a_c * p);
                vol.integrate(pa, T_a_c.translation());
            }
            if (merge)
                vol.accumulateInto(*fused, T_w_a, 1.f);
            else
                append(mesh, vol.mesh(), T_w_a);
            s = j - chunk; // next chunk starts where this one stopped
        }
        if (merge)
            mesh = fused->mesh();
    }
    if (!writeDenseMeshPly(mesh, out, "dense_replay " + poses))
        return 1;
    std::cout << "dense_replay: " << mesh.vertices.size() << " vertices, " << mesh.faces.size() << " faces -> " << out
              << std::endl;
    return 0;
}
