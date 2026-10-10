#include "isaeslam/loopclosure/LoopClosure.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/sensors/ASensor.h"

namespace isae {

namespace {

using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// DBoW3 keeps its loader of ORB-SLAM's text vocabularies protected
struct TxtVocabulary : public DBoW3::Vocabulary {
    using DBoW3::Vocabulary::load_fromtxt;
};

void writeRow(std::ofstream &f, unsigned long long ts, const Eigen::Affine3d &T, int segment, int group) {
    const Eigen::Matrix3d R = T.linear();
    const Eigen::Vector3d t = T.translation();
    f << ts << ",0," << R(0, 0) << "," << R(0, 1) << "," << R(0, 2) << "," << t.x() << "," << R(1, 0) << ","
      << R(1, 1) << "," << R(1, 2) << "," << t.y() << "," << R(2, 0) << "," << R(2, 1) << "," << R(2, 2) << ","
      << t.z() << "," << segment << "," << group << "\n";
}

const char *kTrajHeader = "timestamp (ns), nframes, T_wf(00), T_wf(01), T_wf(02), T_wf(03), T_wf(10), T_wf(11), "
                          "T_wf(12), T_wf(13), T_wf(20), T_wf(21), T_wf(22), T_wf(23), segment, group\n";

// ORB's orientation (intensity centroid over its 31 x 31 circular patch), in degrees as cv::KeyPoint::angle
float icAngle(const cv::Mat &img, const cv::Point2f &p) {
    const int R = 15;
    double m01 = 0, m10 = 0;
    for (int v = -R; v <= R; v++) {
        const int d = static_cast<int>(std::round(std::sqrt(double(R * R - v * v))));
        for (int u = -d; u <= d; u++) {
            const int x = cvRound(p.x) + u, y = cvRound(p.y) + v;
            if (x < 0 || y < 0 || x >= img.cols || y >= img.rows)
                continue;
            const double I = img.at<uchar>(y, x);
            m10 += u * I;
            m01 += v * I;
        }
    }
    const float a = static_cast<float>(std::atan2(m01, m10) * 180 / CV_PI);
    return a < 0 ? a + 360 : a;
}

} // namespace

LoopClosure::LoopClosure(const Options &opt) : _opt(opt), _graph(opt.four_dof, 4, opt.sim3) {
    const bool bow = _opt.detector == "bow" || _opt.detector == "proximity_bow";
    if (bow) {
        const auto t0 = Clock::now();
        const std::string &v = _opt.vocabulary;
        if (v.size() > 4 && v.substr(v.size() - 4) == ".txt") {
            TxtVocabulary tv;
            tv.load_fromtxt(v);
            _voc = tv;
        } else {
            _voc.load(v);
        }
        if (_voc.empty())
            throw std::runtime_error("loop closure: cannot load the vocabulary " + v);
        _db.setVocabulary(_voc, false, 0);
        std::cout << "Loop closure: vocabulary of " << _voc.size() << " words loaded in " << msSince(t0) / 1e3
                  << " s" << std::endl;
    }
    if (_opt.detector == "learned") {
        const auto t0 = Clock::now();
        _vpr          = GlobalDescriptor::create(_opt.model, _opt.model_device, _opt.model_threads);
        _vpr->describe(cv::Mat(_vpr->height(), _vpr->width(), CV_8U, cv::Scalar(128))); // first inference allocates
        std::cout << "Loop closure: model " << _opt.model << " on " << _opt.model_device << ", input "
                  << _vpr->height() << " x " << _vpr->width() << ", descriptor " << _vpr->dim() << ", ready in "
                  << msSince(t0) / 1e3 << " s" << std::endl;
    }
    _orb = cv::ORB::create(_opt.n_orb, 1.2f, 8);
    std::filesystem::create_directories("log_slam");

    std::ofstream("log_slam/loops.csv", std::ofstream::trunc)
        << "timestamp query (ns), timestamp match (ns), candidates, inliers query->match, inliers match->query, "
           "kept by pose graph, t_x, t_y, t_z (match frame), ms, scale ratio (sim3: match map length per query map "
           "length)\n";
    std::ofstream("log_slam/loop_attempts.csv", std::ofstream::trunc)
        << "timestamp query (ns), timestamp candidate (ns), query landmarks, matches query->candidate, inliers, "
           "candidate landmarks, matches candidate->query, inliers, disagreement (m; sim3: fraction of the candidate's "
           "scene depth), disagreement (deg), accepted\n";
    std::ofstream("log_slam/results_loop.csv", std::ofstream::trunc) << kTrajHeader;
    std::ofstream("log_slam/results_loop_online.csv", std::ofstream::trunc) << kTrajHeader;

    if (_opt.async)
        _thread = std::thread(&LoopClosure::loop, this);
}

LoopClosure::~LoopClosure() {
    if (_thread.joinable()) {
        _stop = true;
        _queue_cv.notify_all();
        _thread.join();
    }
}

void LoopClosure::addKeyframe(const std::shared_ptr<Frame> &f, int segment) {
    if (!f || f->getSensors().empty())
        return;
    Input in;
    in.ts      = f->getTimestamp();
    in.segment = segment;
    in.T_w_f   = f->getFrame2WorldTransform();
    in.T_w_f_handover = in.T_w_f;
    in.cam     = f->getSensors().at(0);
    in.T_s_f   = in.cam->getFrame2SensorTransform();

    // Camera 0's landmarks in camera 0, with the front end's descriptors (ORB only: they are matched to ORB)
    const Eigen::Affine3d T_c_w = in.T_s_f * f->getWorld2FrameTransform();
    for (auto &feat : in.cam->getFeatures("pointxd")) {
        std::shared_ptr<ALandmark> lmk = feat->getLandmark().lock();
        if (!lmk || !lmk->isInitialized() || lmk->isOutlier() || feat->getPoints().empty())
            continue;
        const Eigen::Vector3d p_c = T_c_w * lmk->getPose().translation();
        if (p_c.z() < 0.1)
            continue;
        const Eigen::Vector2d px = feat->getPoints().at(0);
        in.lmk_c.push_back(p_c);
        in.lmk_px.emplace_back(static_cast<float>(px.x()), static_cast<float>(px.y()));
    }

    if (_opt.async) {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _queue.push_back(std::move(in));
        _queue_cv.notify_one();
    } else {
        process(in);
    }
}

void LoopClosure::loop() {
    while (true) {
        Input in;
        {
            std::unique_lock<std::mutex> lock(_queue_mutex);
            _queue_cv.wait(lock, [&] { return _stop || !_queue.empty(); });
            if (_queue.empty())
                return;
            in = std::move(_queue.front());
            _queue.pop_front();
        }
        process(in);
    }
}

Eigen::Affine3d LoopClosure::correct(const Eigen::Affine3d &T_w_f, int segment) const {
    std::lock_guard<std::mutex> lock(_graph_mutex);
    Eigen::Affine3d T = _graph.correction(segment) * T_w_f;
    T.linear() /= std::cbrt(T.linear().determinant()); // sim3: the similarity's scale is not part of the pose
    return T;
}

bool LoopClosure::takeCorrection(int segment, Eigen::Affine3d &C) {
    std::lock_guard<std::mutex> lock_q(_queue_mutex);
    std::lock_guard<std::mutex> lock(_graph_mutex);
    if (!_pending_correction || _opt.sim3)
        return false;
    _pending_correction = false;
    C                   = _graph.correction(segment);
    if (_opt.window_gravity) {
        // Yaw about the world z axis (gravity) closest to the correction's rotation, and the translation that puts
        // the segment's latest KF where the full correction puts it
        int last = -1;
        for (int i = static_cast<int>(_graph.size()) - 1; i >= 0 && last < 0; i--)
            if (_graph.segment(i) == segment)
                last = i;
        if (last < 0)
            return false;
        const Eigen::Matrix3d &R   = C.linear();
        const double yaw           = std::atan2(R(1, 0) - R(0, 1), R(0, 0) + R(1, 1));
        const Eigen::Vector3d p    = _graph.odomPose(last).translation();
        Eigen::Affine3d C4         = Eigen::Affine3d::Identity();
        C4.linear()                = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        C4.translation()           = C * p - C4.linear() * p;
        C                          = C4;
    }
    if (C.translation().norm() < 1e-3 && Eigen::AngleAxisd(C.rotation()).angle() < 1e-4)
        return false;
    _graph.reanchor(segment, C);
    for (auto &in : _queue)
        if (in.segment == segment)
            in.T_w_f = C * in.T_w_f;
    return true;
}

bool LoopClosure::display(Display &out, unsigned long known) const {
    std::lock_guard<std::mutex> lock(_graph_mutex);
    if (_version == known)
        return false;
    out           = Display();
    out.version   = _version;
    const int n   = static_cast<int>(_graph.size());
    for (int i = 1; i < n; i++)
        if (_graph.segment(i) == _graph.segment(i - 1))
            out.traj.push_back({_graph.pose(i - 1).translation(), _graph.pose(i).translation()});
    for (const auto &l : _graph.loops())
        if (std::get<2>(l))
            out.loops.push_back({_graph.pose(std::get<0>(l)).translation(), _graph.pose(std::get<1>(l)).translation()});
    for (const auto &l : _last_loops)
        out.new_loops.push_back({_graph.pose(l.first).translation(), _graph.pose(l.second).translation()});
    return true;
}

bool LoopClosure::keyframeCorrection(unsigned long long ts, Eigen::Affine3d &C) const {
    std::lock_guard<std::mutex> lock(_graph_mutex);
    const auto it = _handover.find(ts);
    if (it == _handover.end())
        return false;
    C = _graph.pose(it->second.first) * it->second.second.inverse();
    return true;
}

LoopClosure::Stats LoopClosure::stats() const {
    std::lock_guard<std::mutex> lock(_graph_mutex);
    return _stats;
}

void LoopClosure::process(Input &in) {
    const auto t0 = Clock::now();

    // Pyramid ORB of camera 0, kept as normalized coordinates of their rays (any camera model)
    cv::Mat img = in.cam->getRawData();
    if (img.depth() == CV_16U)
        img.convertTo(img, CV_8U, 1.0 / 256);
    if (img.channels() == 3)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    std::vector<cv::KeyPoint> kps;
    cv::Mat desc;
    _orb->detectAndCompute(img, cv::noArray(), kps, desc);

    Record r;
    r.ts      = in.ts;
    r.segment = in.segment;
    r.T_s_f   = in.T_s_f;
    r.focal   = in.cam->getFocal();
    for (size_t k = 0; k < kps.size(); k++) {
        const Eigen::Vector3d ray = in.cam->getRayCamera(Eigen::Vector2d(kps[k].pt.x, kps[k].pt.y));
        if (!(ray.z() > 0.2 * ray.norm()))
            continue;
        r.kp_n.emplace_back(ray.x() / ray.z(), ray.y() / ray.z());
        r.desc.push_back(desc.row(static_cast<int>(k)));
    }
    in.cam.reset(); // the image is no longer needed
    if (!_voc.empty() && !r.desc.empty())
        _voc.transform(r.desc, r.bow);
    double ms_describe = 0;
    if (_vpr) {
        const auto td = Clock::now();
        r.gdesc       = _vpr->describe(img);
        ms_describe   = msSince(td);
    }

    // Oriented ORB at the landmarks' pixels (same extractor as the keyframe's ORB, so the two match)
    std::vector<cv::KeyPoint> lkps;
    for (size_t k = 0; k < in.lmk_px.size(); k++)
        lkps.emplace_back(in.lmk_px[k], 31.f, icAngle(img, in.lmk_px[k]), 0.f, 0, static_cast<int>(k));
    cv::Mat ldesc;
    if (!lkps.empty())
        _orb->compute(img, lkps, ldesc); // drops points too close to the border
    for (size_t k = 0; k < lkps.size(); k++) {
        r.lmk_c.push_back(in.lmk_c[lkps[k].class_id]);
        r.lmk_desc.push_back(ldesc.row(static_cast<int>(k)));
    }
    {
        std::lock_guard<std::mutex> lock(_graph_mutex);
        r.node = _graph.addNode(in.T_w_f, in.segment);
        _handover[in.ts] = {r.node, in.T_w_f_handover};
    }

    // Detection and verification: up to max_loops_per_kf verified candidates, from different times (a recent
    // revisit and an old one constrain different drift), then one pose graph solve
    const std::vector<int> cands = candidates(r);
    bool optimized               = false;
    std::vector<std::pair<int, std::array<int, 2>>> accepted;
    std::vector<Eigen::Affine3d> rel;
    const unsigned long long gap = static_cast<unsigned long long>(_opt.min_gap * 1e9);
    for (int c : cands) {
        if (static_cast<int>(accepted.size()) >= _opt.max_loops_per_kf)
            break;
        bool distinct = true;
        for (auto &a : accepted) {
            const unsigned long long t1 = _records[a.first].ts, t2 = _records[c].ts;
            distinct &= (t1 > t2 ? t1 - t2 : t2 - t1) >= gap;
        }
        if (!distinct)
            continue;
        Eigen::Affine3d T_fc_fq;
        std::array<int, 2> inl = {0, 0};
        if (!verify(r, _records[c], T_fc_fq, inl[0], inl[1]))
            continue;
        accepted.push_back({c, inl});
        rel.push_back(T_fc_fq);
    }
    if (!accepted.empty()) {
        std::lock_guard<std::mutex> lock(_graph_mutex);
        _last_loops.clear();
        for (size_t k = 0; k < accepted.size(); k++) {
            _graph.addLoop(_records[accepted[k].first].node, r.node, rel[k]);
            _last_loops.push_back({_records[accepted[k].first].node, r.node});
        }
        const auto t1    = Clock::now();
        const int reject = _graph.optimize();
        const double dt  = msSince(t1);
        optimized        = true;
        _pending_correction = _opt.correct_window;
        _stats.n_verified += static_cast<int>(accepted.size());
        _stats.n_loops++;
        _stats.n_rejected_by_graph += reject;
        _stats.avg_ms_optimize += (dt - _stats.avg_ms_optimize) / _stats.n_loops;
        std::ofstream lf("log_slam/loops.csv", std::ofstream::app);
        for (size_t k = 0; k < accepted.size(); k++)
            lf << r.ts << "," << _records[accepted[k].first].ts << "," << cands.size() << ","
               << accepted[k].second[0] << "," << accepted[k].second[1] << "," << (reject == 0 ? 1 : 0) << ","
               << rel[k].translation().x() << "," << rel[k].translation().y() << "," << rel[k].translation().z()
               << "," << dt << "," << std::cbrt(rel[k].linear().determinant()) << "\n";
    }

    _records.push_back(std::move(r));
    if (!_voc.empty())
        _db.add(_records.back().bow);
    writeTrajectory();
    if (optimized) {
        // The whole corrected trajectory changes after a solve
        std::lock_guard<std::mutex> lock(_graph_mutex);
        std::ofstream f("log_slam/results_loop.csv", std::ofstream::trunc);
        f << kTrajHeader;
        for (const auto &rec : _records)
            writeRow(f, rec.ts, _graph.pose(rec.node), rec.segment, _graph.group(rec.segment));
    }

    std::lock_guard<std::mutex> lock(_graph_mutex);
    _version++;
    _stats.n_keyframes++;
    _stats.n_candidates += static_cast<int>(cands.size());
    _stats.avg_ms_describe += (ms_describe - _stats.avg_ms_describe) / _stats.n_keyframes;
    _stats.avg_ms_keyframe += (msSince(t0) - _stats.avg_ms_keyframe) / _stats.n_keyframes;
}

void LoopClosure::writeTrajectory() {
    // The new keyframe at its current corrected pose: the online output (results_loop_online.csv, never rewritten)
    // and the running trajectory (results_loop.csv, rewritten after each pose graph solve)
    // Labels: rows of segments joined by loops are in one frame (column group). In the online trajectory a segment
    // gets a new label (segment column: 1000 x segment + frame changes) each time its frame changes (when it joins a
    // group), so that rows written in different frames are never aligned together
    const Record &r = _records.back();
    Eigen::Affine3d T;
    int group;
    {
        std::lock_guard<std::mutex> lock(_graph_mutex);
        T     = _graph.pose(r.node);
        group = _graph.group(r.segment);
    }
    auto it = _online_label.find(r.segment);
    if (it == _online_label.end())
        it = _online_label.emplace(r.segment, std::make_pair(group, 0)).first;
    else if (it->second.first != group)
        it->second = {group, it->second.second + 1};
    std::ofstream on("log_slam/results_loop_online.csv", std::ofstream::app);
    writeRow(on, r.ts, T, r.segment * 1000 + it->second.second, group);
    std::ofstream run("log_slam/results_loop.csv", std::ofstream::app);
    writeRow(run, r.ts, T, r.segment, group);
}

std::vector<int> LoopClosure::candidates(const Record &q) {
    // Only keyframes at least min_gap older than the query
    const unsigned long long gap = static_cast<unsigned long long>(_opt.min_gap * 1e9);
    if (q.ts < gap)
        return {};
    const int n_old = static_cast<int>(
        std::upper_bound(_records.begin(), _records.end(), q.ts - gap,
                         [](unsigned long long t, const Record &rec) { return t < rec.ts; }) -
        _records.begin());
    if (n_old == 0)
        return {};

    std::vector<int> out;
    auto push = [&](int i) {
        if (std::find(out.begin(), out.end(), i) == out.end())
            out.push_back(i);
    };
    // Ranked candidates, keeping at most `quota` that are at least min_gap apart in time: otherwise a place visited
    // for a while fills the list with its own recent KFs and an older visit (the one that bounds the drift: e.g. the
    // start of TUM-VI magistrale2, seen again at its end) never gets verified
    auto pushDiverse = [&](const std::vector<int> &ranked, int quota) {
        std::vector<int> chosen;
        for (int i : ranked) {
            if (static_cast<int>(chosen.size()) >= quota)
                break;
            bool far = true;
            for (int c : chosen) {
                const unsigned long long a = _records[i].ts, b = _records[c].ts;
                far &= (a > b ? a - b : b - a) >= gap;
            }
            if (far)
                chosen.push_back(i);
        }
        for (int i : chosen)
            push(i);
    };

    if (_opt.detector == "proximity" || _opt.detector == "proximity_bow") {
        // Gate on the corrected poses: position within gate_radius, viewing directions within gate_angle
        std::vector<std::pair<double, int>> gate;
        {
            std::lock_guard<std::mutex> lock(_graph_mutex);
            const Eigen::Affine3d Tq   = _graph.pose(q.node);
            const Eigen::Vector3d zq   = Tq.rotation() * q.T_s_f.rotation().transpose() * Eigen::Vector3d::UnitZ();
            const double cos_max       = std::cos(_opt.gate_angle * M_PI / 180);
            for (int i = 0; i < n_old; i++) {
                const Eigen::Affine3d Ti = _graph.pose(_records[i].node);
                const double d           = (Ti.translation() - Tq.translation()).norm();
                const Eigen::Vector3d zi = Ti.rotation() * _records[i].T_s_f.rotation().transpose() * Eigen::Vector3d::UnitZ();
                if (d < _opt.gate_radius && zi.dot(zq) > cos_max)
                    gate.push_back({d, i});
            }
        }
        if (_opt.detector == "proximity_bow" && !q.bow.empty()) {
            for (auto &g : gate)
                g.first = -_voc.score(q.bow, _records[g.second].bow); // rank by appearance inside the gate
        }
        std::sort(gate.begin(), gate.end());
        for (int k = 0; k < std::min<int>(_opt.n_candidates, gate.size()); k++)
            push(gate[k].second);
    }
    if (_opt.detector == "learned" && q.gdesc.size() > 0) {
        // Cosine similarity of the global descriptors (unit norm)
        std::vector<std::pair<float, int>> sims;
        sims.reserve(n_old);
        for (int i = 0; i < n_old; i++)
            sims.push_back({-_records[i].gdesc.dot(q.gdesc), i});
        const int k = std::min<int>(4 * _opt.n_candidates, n_old);
        std::partial_sort(sims.begin(), sims.begin() + k, sims.end());
        std::vector<int> ranked;
        for (int j = 0; j < k; j++)
            ranked.push_back(sims[j].second);
        pushDiverse(ranked, _opt.n_candidates);
    }
    if ((_opt.detector == "bow" || _opt.detector == "proximity_bow") && !q.bow.empty()) {
        // Global search, also beyond the gate (drift larger than the gate)
        DBoW3::QueryResults res;
        _db.query(q.bow, res, 4 * _opt.n_candidates, n_old - 1);
        const int n_global = _opt.detector == "bow" ? _opt.n_candidates : std::max(1, _opt.n_candidates - 1);
        std::vector<int> ranked;
        for (const auto &rr : res)
            ranked.push_back(static_cast<int>(rr.Id));
        pushDiverse(ranked, n_global);
    }
    return out;
}

bool LoopClosure::pnp(const std::vector<Eigen::Vector3d> &pts, const cv::Mat &pts_desc, const Record &target,
                      Eigen::Affine3d &T_t_src, int &inliers, int &matches,
                      std::vector<std::pair<int, int>> *inlier_pairs) const {
    inliers = matches = 0;
    if (static_cast<int>(pts.size()) < _opt.min_inliers || target.desc.empty())
        return false;
    // Targets: the target's landmarks (projected into its camera: the same kind of corner as the source's
    // landmarks, described the same way) and its pyramid ORB
    std::vector<cv::Point2f> tgt_n;
    cv::Mat tgt_desc;
    for (size_t k = 0; k < target.lmk_c.size(); k++) {
        const Eigen::Vector3d &p = target.lmk_c[k];
        tgt_n.emplace_back(p.x() / p.z(), p.y() / p.z());
    }
    tgt_n.insert(tgt_n.end(), target.kp_n.begin(), target.kp_n.end());
    if (!target.lmk_desc.empty())
        cv::vconcat(target.lmk_desc, target.desc, tgt_desc);
    else
        tgt_desc = target.desc;

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(pts_desc, tgt_desc, knn, 2);
    std::vector<cv::Point3f> obj;
    std::vector<cv::Point2f> img;
    std::vector<std::pair<int, int>> pairs; // (source point, target) of each correspondence
    for (const auto &m : knn) {
        // A landmark detected again by the ORB has two near-identical targets: no ratio test between them
        if (m.empty() || m[0].distance > 50)
            continue;
        const bool twin = m.size() > 1 && cv::norm(tgt_n[m[0].trainIdx] - tgt_n[m[1].trainIdx]) * target.focal < 3;
        if (m.size() > 1 && !twin && m[0].distance > 0.8f * m[1].distance)
            continue;
        const Eigen::Vector3d &p = pts[m[0].queryIdx];
        obj.emplace_back(p.x(), p.y(), p.z());
        img.push_back(tgt_n[m[0].trainIdx]);
        pairs.emplace_back(m[0].queryIdx, m[0].trainIdx);
    }
    matches = static_cast<int>(obj.size());
    if (matches < _opt.min_inliers)
        return false;
    cv::Mat rvec, tvec;
    std::vector<int> idx;
    const double thr = 3.0 / std::max(target.focal, 1.0); // 3 px in normalized coordinates
    if (!cv::solvePnPRansac(obj, img, cv::Mat::eye(3, 3, CV_64F), cv::Mat(), rvec, tvec, false, 200, thr, 0.99, idx,
                            cv::SOLVEPNP_EPNP))
        return false;
    inliers = static_cast<int>(idx.size());
    if (inliers < _opt.min_inliers)
        return false;
    if (inlier_pairs) {
        // Targets below target.lmk_c.size() are the target's landmarks
        inlier_pairs->clear();
        for (int k : idx)
            inlier_pairs->push_back(pairs[k]);
    }
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    T_t_src = Eigen::Affine3d::Identity();
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++)
            T_t_src.linear()(a, b) = R.at<double>(a, b);
        T_t_src.translation()(a) = tvec.at<double>(a);
    }
    return true;
}

bool LoopClosure::verify(const Record &q, const Record &c, Eigen::Affine3d &T_fc_fq, int &inl_q, int &inl_c) {
    // The query's landmarks seen in the candidate's image, and the candidate's landmarks in the query's image
    Eigen::Affine3d T_cc_cq, T_cq_cc;
    inl_q = inl_c = 0;
    int m_q = 0, m_c = 0;
    std::vector<std::pair<int, int>> pairs_q, pairs_c;
    const bool ok_q   = pnp(q.lmk_c, q.lmk_desc, c, T_cc_cq, inl_q, m_q, &pairs_q);
    const bool both   = static_cast<int>(c.lmk_c.size()) >= _opt.min_inliers;
    const bool ok_c   = both && pnp(c.lmk_c, c.lmk_desc, q, T_cq_cc, inl_c, m_c, &pairs_c);
    bool accepted     = false;
    double e_t = -1, e_deg = -1;
    double s_cq = 1; // sim3: the candidate map's length per query map length
    if (_opt.sim3) {
        // Scale ratio from the landmarks matched between the two maps: a query landmark brought into the candidate's
        // camera (query units) against the candidate landmark it matched (candidate units), and the reverse. The
        // two directions then agree once the query's translation is scaled, within 10% of the candidate's scene
        // depth (0.3 m at 3 m, the metric threshold)
        if (ok_q && ok_c) {
            std::vector<double> lq, lc, depth_c;
            auto logRatio = [](double num, double den, std::vector<double> &out, double sign) {
                if (num > 1e-6 && den > 1e-6)
                    out.push_back(sign * std::log(num / den));
            };
            for (const auto &m : pairs_q)
                if (m.second < static_cast<int>(c.lmk_c.size()))
                    logRatio(c.lmk_c[m.second].norm(), (T_cc_cq * q.lmk_c[m.first]).norm(), lq, 1);
            for (const auto &m : pairs_c)
                if (m.second < static_cast<int>(q.lmk_c.size()))
                    logRatio(q.lmk_c[m.second].norm(), (T_cq_cc * c.lmk_c[m.first]).norm(), lc, -1);
            for (const auto &p : c.lmk_c)
                depth_c.push_back(p.norm());
            auto median = [](std::vector<double> v) {
                std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
                return v[v.size() / 2];
            };
            std::vector<double> all = lq;
            all.insert(all.end(), lc.begin(), lc.end());
            const bool directions_agree =
                lq.size() < 3 || lc.size() < 3 || std::abs(median(lq) - median(lc)) < std::log(1.15);
            if (all.size() >= 5 && directions_agree) {
                s_cq                  = std::exp(median(all));
                Eigen::Affine3d T_s   = T_cc_cq;
                T_s.translation()    *= s_cq;
                const Eigen::Affine3d E = T_s * T_cq_cc;
                e_t                     = E.translation().norm() / median(depth_c);
                e_deg                   = Eigen::AngleAxisd(E.rotation()).angle() * 180 / M_PI;
                accepted                = e_t < 0.1 && e_deg < 5;
            }
        }
    } else if (ok_q && ok_c) {
        // The two directions must agree
        const Eigen::Affine3d E = T_cc_cq * T_cq_cc;
        e_t                     = E.translation().norm();
        e_deg                   = Eigen::AngleAxisd(E.rotation()).angle() * 180 / M_PI;
        accepted                = e_t < 0.3 && e_deg < 5;
    } else if (ok_q && !both) {
        accepted = inl_q >= 1.5 * _opt.min_inliers; // the candidate has too few landmarks for the reverse check
    }
    // Every verification attempt, for tuning (log_slam/loop_attempts.csv)
    std::ofstream("log_slam/loop_attempts.csv", std::ofstream::app)
        << q.ts << "," << c.ts << "," << q.lmk_c.size() << "," << m_q << "," << inl_q << "," << c.lmk_c.size() << ","
        << m_c << "," << inl_c << "," << e_t << "," << e_deg << "," << (accepted ? 1 : 0) << "\n";
    if (!accepted)
        return false;
    Eigen::Affine3d S_cc_cq = T_cc_cq; // sim3: a similarity (linear part s R), else rigid
    S_cc_cq.linear()       *= s_cq;
    S_cc_cq.translation()  *= s_cq;
    T_fc_fq = c.T_s_f.inverse() * S_cc_cq * q.T_s_f;
    return true;
}

} // namespace isae
