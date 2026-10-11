#include "isaeslam/stereo/AnchoredMaps.h"
#include <Eigen/Geometry>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unordered_set>

namespace isae {

DenseMesh transformMesh(const DenseMesh& m, const Eigen::Affine3d& T) {
    DenseMesh out;
    out.vertices.reserve(m.vertices.size());
    for (const auto& v : m.vertices)
        out.vertices.push_back(T * v);
    out.faces = m.faces;
    out.face_normals.reserve(m.face_normals.size());
    for (const auto& n : m.face_normals)
        out.face_normals.push_back(T.linear() * n);
    out.vertex_variance    = m.vertex_variance;
    out.gaussian_curvature = m.gaussian_curvature;
    out.mean_curvature     = m.mean_curvature;
    return out;
}

AnchoredMaps::AnchoredMaps(const AnchoredMapsConfig& cfg) : _cfg(cfg) {
    const std::filesystem::path mesh_path(cfg.mesh_path);
    const std::filesystem::path dir =
        mesh_path.has_parent_path() ? mesh_path.parent_path() : std::filesystem::path(".");
    _dir = (dir / "dense_anchors").string();
    std::filesystem::create_directories(_dir);
}

void AnchoredMaps::setKeyframeCorrection(KeyframeCorrection fn) {
    std::lock_guard<std::mutex> lock(_corr_mtx);
    _corr = std::move(fn);
}

void AnchoredMaps::writeAnchorPly(const Anchor& a) const {
    char name[32];
    std::snprintf(name, sizeof(name), "/anchor_%05d.ply", a.id);
    if (!writeDenseMeshPly(a.mesh, _dir + name, "SaDVIO dense geometry, anchor frame (anchors.csv)"))
        std::cerr << "[DenseMesh] could not write " << _dir + name << std::endl;
}

void AnchoredMaps::addKeyframe(unsigned long long ts, int segment, int epoch, const Eigen::Affine3d& T_w_cam,
                               DenseMesh mesh_cam) {
    auto a        = std::make_unique<Anchor>();
    a->id         = static_cast<int>(_anchors.size());
    a->ts         = ts;
    a->segment    = segment;
    a->epoch      = epoch;
    a->T_w_anchor = T_w_cam;
    a->n_kf       = 1;
    a->closed     = true;
    a->mesh       = std::move(mesh_cam);
    writeAnchorPly(*a);
    _anchors.push_back(std::move(a));
    if (_cfg.fused_view != 0 && _cfg.view_from_geometry)
        addGeometryToView(*_anchors.back());
}

void AnchoredMaps::addGeometryToView(const Anchor& a) {
    if (a.mesh.vertices.empty())
        return;
    View* v = _views.empty() || _views.back().closed ? nullptr : &_views.back();
    if (v && (v->n_kf >= _cfg.submap_kfs || v->segment != a.segment || v->epoch != a.epoch)) {
        v->closed = true;
        v         = nullptr;
    }
    if (!v) {
        View nv;
        nv.anchor  = a.id;
        nv.segment = a.segment;
        nv.epoch   = a.epoch;
        nv.vol     = std::make_unique<VDBGPDFMap>(_cfg.view_cfg);
        _views.push_back(std::move(nv));
        v = &_views.back();
    }
    // The geometry in the chunk's frame, integrated from the anchor's camera (its own keyframe's camera for the
    // per-keyframe methods, the submap's first camera for the GP map)
    // Thinned to half a voxel first (dense GP meshes have tens of thousands of vertices per keyframe)
    const Eigen::Affine3d T_v_a = _anchors[v->anchor]->T_w_anchor.inverse() * a.T_w_anchor;
    const double cell           = 0.5 * _cfg.view_cfg.voxel_size_global;
    std::unordered_set<long long> seen;
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(a.mesh.vertices.size());
    for (const auto& p : a.mesh.vertices) {
        const Eigen::Vector3i k = (p / cell).array().floor().cast<int>();
        const long long key     = (static_cast<long long>(k.x()) * 73856093LL) ^
                              (static_cast<long long>(k.y()) * 19349663LL) ^ (static_cast<long long>(k.z()) * 83492791LL);
        if (seen.insert(key).second)
            pts.push_back(T_v_a * p);
    }
    v->vol->integrate(pts, T_v_a.translation());
    v->n_kf += a.n_kf;
    if (v->n_kf >= _cfg.submap_kfs)
        v->closed = true;
}

AnchoredMaps::Anchor& AnchoredMaps::submapFor(unsigned long long ts, int segment, int epoch,
                                              const Eigen::Affine3d& T_w_cam, Eigen::Affine3d& T_a_c,
                                              const Mesher& mesher) {
    Anchor* cur = _anchors.empty() || _anchors.back()->closed ? nullptr : _anchors.back().get();
    // A submap's keyframes must share one odometry world for their relative poses to hold
    if (cur && (cur->n_kf >= _cfg.submap_kfs || cur->segment != segment || cur->epoch != epoch)) {
        closeOpenSubmap(mesher);
        cur = nullptr;
    }
    if (!cur) {
        auto a        = std::make_unique<Anchor>();
        a->id         = static_cast<int>(_anchors.size());
        a->ts         = ts;
        a->segment    = segment;
        a->epoch      = epoch;
        a->T_w_anchor = T_w_cam;
        _anchors.push_back(std::move(a));
        cur = _anchors.back().get();
    }
    T_a_c = cur->T_w_anchor.inverse() * T_w_cam;
    cur->n_kf++;
    return *cur;
}

void AnchoredMaps::closeOpenSubmap(const Mesher& mesher) {
    if (_anchors.empty() || _anchors.back()->closed)
        return;
    Anchor& a = *_anchors.back();
    a.mesh    = mesher(a);
    a.closed  = true;
    writeAnchorPly(a);
    if (_cfg.fused_view != 0 && a.vdb && !_cfg.view_from_geometry) {
        // vdbgpdf: the submap's volume itself goes to the fused view
        View v;
        v.anchor  = a.id;
        v.segment = a.segment;
        v.epoch   = a.epoch;
        v.n_kf    = a.n_kf;
        v.closed  = true;
        v.vol     = std::move(a.vdb);
        _views.push_back(std::move(v));
    } else if (_cfg.fused_view != 0 && _cfg.view_from_geometry) {
        addGeometryToView(a);
    }
    a.vdb.reset(); // a closed submap keeps only its mesh (and, for the fused view, its volume)
    a.gp.reset();
}

bool AnchoredMaps::keyframeDone(const std::function<void(Anchor&)>& refresh_open) {
    ++_keyframes;
    if (_global.vertices.empty() || (_cfg.save_every > 0 && _keyframes % _cfg.save_every == 0)) {
        if (refresh_open && !_anchors.empty() && !_anchors.back()->closed)
            refresh_open(*_anchors.back());
        save();
        return true;
    }
    return false;
}

Eigen::Affine3d AnchoredMaps::correction(const Anchor& a, const KeyframeCorrection& corr) const {
    Eigen::Affine3d C = Eigen::Affine3d::Identity();
    if (corr && !corr(a.ts, C))
        C = Eigen::Affine3d::Identity();
    return C;
}

AnchoredMaps::KeyframeCorrection AnchoredMaps::currentCorrection() {
    std::lock_guard<std::mutex> lock(_corr_mtx);
    return _corr;
}

static bool movedBeyond(const Eigen::Affine3d& A, const Eigen::Affine3d& B, double t, double deg) {
    const Eigen::Affine3d D = A.inverse() * B;
    return D.translation().norm() > t || Eigen::AngleAxisd(D.linear()).angle() * 180 / M_PI > deg;
}

void AnchoredMaps::updateFused(const KeyframeCorrection& corr, bool full) {
    if (full || !_fused) {
        _fused = std::make_unique<VDBGPDFMap>(_cfg.view_cfg);
        for (auto& v : _views)
            v.projected = false;
    }
    int added = 0, moved = 0;
    for (auto& v : _views) {
        if (!v.closed || !v.vol)
            continue;
        const Anchor& a         = *_anchors[v.anchor];
        const Eigen::Affine3d T = correction(a, corr) * a.T_w_anchor;
        if (!v.projected) {
            v.vol->accumulateInto(*_fused, T, 1.f);
            added++;
        } else if (movedBeyond(v.T_proj, T, _cfg.fused_move_t, _cfg.fused_move_deg)) {
            v.vol->accumulateInto(*_fused, v.T_proj, -1.f); // exact: the same splat, negated
            v.vol->accumulateInto(*_fused, T, 1.f);
            moved++;
        } else {
            continue;
        }
        v.projected = true;
        v.T_proj    = T;
    }
    DenseMesh out = _fused->mesh();
    // Not merged yet: the open volume (an open vdbgpdf submap, or the open chunk), as a layer of its own
    auto append = [&](const DenseMesh& m, const Eigen::Affine3d& T) {
        const int base = static_cast<int>(out.vertices.size());
        for (const auto& p : m.vertices)
            out.vertices.push_back(T * p);
        for (const auto& f : m.faces)
            out.faces.push_back(f + Eigen::Vector3i::Constant(base));
        for (const auto& n : m.face_normals)
            out.face_normals.push_back(T.linear() * n);
    };
    if (!_anchors.empty() && !_anchors.back()->closed)
        append(_anchors.back()->mesh, correction(*_anchors.back(), corr) * _anchors.back()->T_w_anchor);
    if (!_views.empty() && !_views.back().closed && _views.back().vol) {
        const Anchor& a = *_anchors[_views.back().anchor];
        append(_views.back().vol->mesh(), correction(a, corr) * a.T_w_anchor);
    }
    out.vertex_variance.clear();
    if (out.face_normals.size() != out.faces.size())
        out.face_normals.clear();
    _fused_mesh = std::move(out);
    if (!writeDenseMeshPly(_fused_mesh, _cfg.fused_mesh_path, "SaDVIO dense fused view (anchored volumes merged)"))
        std::cerr << "[DenseMesh] could not write " << _cfg.fused_mesh_path << std::endl;
    std::cout << "[DenseMesh] fused view" << (full ? " (full rebuild)" : "") << ": " << added << " volumes added, "
              << moved << " moved, " << _fused->activeVoxels() << " voxels, " << _fused_mesh.faces.size() << " faces"
              << std::endl;
}

void AnchoredMaps::finish() {
    if (!_views.empty())
        _views.back().closed = true;
    const KeyframeCorrection corr = currentCorrection();
    writeAnchors(corr);
    if (_cfg.fused_view == 1 || _cfg.fused_view == 2)
        updateFused(corr, true);
}

void AnchoredMaps::save() {
    const KeyframeCorrection corr = currentCorrection();
    writeAnchors(corr);
    if (_cfg.fused_view == 1)
        updateFused(corr, false);
}

void AnchoredMaps::writeAnchors(const KeyframeCorrection& corr) {
    // anchors.csv: the anchor's pose in the odometry world it was built in, and corrected (equal without a correction)
    std::ofstream csv(_dir + "/anchors.csv", std::ios::trunc);
    csv << "anchor,anchor_timestamp_ns,segment,keyframes,closed,"
           "x,y,z,qx,qy,qz,qw,corrected_x,corrected_y,corrected_z,corrected_qx,corrected_qy,corrected_qz,corrected_qw\n";
    auto pose = [&](const Eigen::Affine3d& T) {
        const Eigen::Quaterniond q(T.linear());
        csv << "," << T.translation().x() << "," << T.translation().y() << "," << T.translation().z() << ","
            << q.x() << "," << q.y() << "," << q.z() << "," << q.w();
    };
    DenseMesh out;
    for (const auto& ap : _anchors) {
        const Anchor& a         = *ap;
        const Eigen::Affine3d T = correction(a, corr) * a.T_w_anchor;
        csv << std::setprecision(10) << a.id << "," << a.ts << "," << a.segment << "," << a.n_kf << ","
            << (a.closed ? 1 : 0);
        pose(a.T_w_anchor);
        pose(T);
        csv << "\n";
        const int base = static_cast<int>(out.vertices.size());
        for (const auto& v : a.mesh.vertices)
            out.vertices.push_back(T * v);
        for (const auto& f : a.mesh.faces)
            out.faces.push_back(f + Eigen::Vector3i::Constant(base));
        for (const auto& n : a.mesh.face_normals)
            out.face_normals.push_back(T.linear() * n);
        if (a.mesh.vertex_variance.size() == a.mesh.vertices.size())
            out.vertex_variance.insert(out.vertex_variance.end(), a.mesh.vertex_variance.begin(),
                                       a.mesh.vertex_variance.end());
    }
    if (out.vertex_variance.size() != out.vertices.size())
        out.vertex_variance.clear();
    if (out.face_normals.size() != out.faces.size())
        out.face_normals.clear();
    _global = std::move(out);
    if (!writeDenseMeshPly(_global, _cfg.mesh_path, "SaDVIO dense anchored map (anchors at corrected poses)"))
        std::cerr << "[DenseMesh] could not write " << _cfg.mesh_path << std::endl;
}

} // namespace isae
