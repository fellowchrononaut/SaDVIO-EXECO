#ifndef ANCHORED_MAPS_H
#define ANCHORED_MAPS_H

#include "isaeslam/stereo/DenseMesh.h"
#include "isaeslam/stereo/GPGlobalMap.h"
#include "isaeslam/stereo/VDBGPDFMap.h"
#include <Eigen/Geometry>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace isae {

// Dense geometry anchored to keyframes, so that it follows the loop closure (doc/dense_anchored_maps.md). Each anchor
// holds geometry in the frame of its anchor keyframe's rectified left camera, the odometry pose that frame had when
// the geometry was built, and the anchor's timestamp; the global mesh places every anchor at its corrected pose.
// Per-keyframe methods (zncc, pd, per-keyframe gp, none) anchor every keyframe on its own; fused methods (gp global
// map, vdbgpdf) fuse submaps of submap_kfs keyframes, each in its first keyframe's frame.
struct AnchoredMapsConfig {
    std::string mesh_path   = "log_slam/dense_anchored_mesh.ply"; // global mesh; anchors in <its dir>/dense_anchors/
    int         save_every  = 5;    // keyframes between global mesh saves (0: only at the end)
    int         submap_kfs  = 10;   // fused methods: keyframes per submap

    // Fused global view (option): the placed volumes merged into one VDB volume by weighted voxel averaging, so that
    // revisits are averaged instead of layered. vdbgpdf merges its submap volumes; the other methods first
    // integrate their geometry into TSDF chunks of submap_kfs keyframes (view_from_geometry), which are merged the
    // same way. 0: off; 1: at every save, re-projecting the closed volumes whose correction moved by more than
    // fused_move_t / fused_move_deg, plus a full rebuild at the end; 2: only the full rebuild at the end
    int              fused_view         = 0;
    std::string      fused_mesh_path    = "log_slam/dense_fused_mesh.ply";
    double           fused_move_t       = 0.02;
    double           fused_move_deg     = 0.5;
    bool             view_from_geometry = false;
    VDBGPDFMapConfig view_cfg; // the fused volume, and the chunks of view_from_geometry
};

class AnchoredMaps {
  public:
    // Correction of a keyframe (corrected world <- world it was handed over in) by timestamp; false: identity
    using KeyframeCorrection = std::function<bool(unsigned long long, Eigen::Affine3d&)>;

    struct Anchor {
        int id = 0;
        unsigned long long ts = 0;       // anchor keyframe
        int segment = 0, epoch = 0;      // odometry world it was built in (re-initializations, window corrections)
        Eigen::Affine3d T_w_anchor = Eigen::Affine3d::Identity(); // odometry pose of the anchor frame
        int n_kf = 0;
        bool closed = false;
        DenseMesh mesh;                  // in the anchor frame (vertices only for point clouds)
        std::unique_ptr<VDBGPDFMap> vdb; // fused methods: the open submap's volume
        std::unique_ptr<GPGlobalMap> gp;
    };

    explicit AnchoredMaps(const AnchoredMapsConfig& cfg);

    void setKeyframeCorrection(KeyframeCorrection fn);

    // Per-keyframe methods: the keyframe's geometry in its own (rectified camera) frame; closed at once
    void addKeyframe(unsigned long long ts, int segment, int epoch, const Eigen::Affine3d& T_w_cam, DenseMesh mesh_cam);

    // Fused methods: the open submap this keyframe goes into. When the current one is full or was built in another
    // odometry world it is closed (mesher gives its final mesh) and a new one opened. T_a_c: the keyframe's camera in
    // the submap's anchor frame. The caller creates the submap's volume when it is new (vdb / gp empty) and
    // integrates into it
    using Mesher = std::function<DenseMesh(Anchor&)>;
    Anchor& submapFor(unsigned long long ts, int segment, int epoch, const Eigen::Affine3d& T_w_cam,
                      Eigen::Affine3d& T_a_c, const Mesher& mesher);

    // Fused methods: close the open submap (its final mesh from mesher, its PLY written, its volume freed)
    void closeOpenSubmap(const Mesher& mesher);

    // A keyframe was added: every save_every keyframes the global mesh is rebuilt and written. Returns true if so
    bool keyframeDone(const std::function<void(Anchor&)>& refresh_open);

    // Global mesh for publishing: the fused view when it is refreshed during the run (fused_view 1), else every anchor
    // at its corrected pose
    const DenseMesh& globalMesh() const {
        return _cfg.fused_view == 1 && !_fused_mesh.vertices.empty() ? _fused_mesh : _global;
    }
    // Writes the anchors at their corrected poses (and anchors.csv), and the fused view with fused_view 1
    void save();
    // End of the run (after the last submap is closed): save, and the full rebuild of the fused view
    void finish();

    size_t size() const { return _anchors.size(); }

  private:
    // A volume of the fused view, placed by an anchor: a vdbgpdf submap's own volume, or a TSDF chunk of geometry
    struct View {
        int anchor = 0; // placing anchor (its corrected pose places the volume)
        int segment = 0, epoch = 0, n_kf = 0;
        bool closed = false, projected = false;
        Eigen::Affine3d T_proj = Eigen::Affine3d::Identity(); // pose it is merged at in _fused
        std::unique_ptr<VDBGPDFMap> vol;
    };

    Eigen::Affine3d correction(const Anchor& a, const KeyframeCorrection& corr) const;
    void writeAnchorPly(const Anchor& a) const;
    void addGeometryToView(const Anchor& a);
    void updateFused(const KeyframeCorrection& corr, bool full);
    void writeAnchors(const KeyframeCorrection& corr);
    KeyframeCorrection currentCorrection();

    AnchoredMapsConfig _cfg;
    std::string _dir;
    std::vector<std::unique_ptr<Anchor>> _anchors;
    DenseMesh _global;
    std::vector<View> _views;
    std::unique_ptr<VDBGPDFMap> _fused;
    DenseMesh _fused_mesh;
    int _keyframes = 0;
    std::mutex _corr_mtx;
    KeyframeCorrection _corr;
};

DenseMesh transformMesh(const DenseMesh& m, const Eigen::Affine3d& T);

} // namespace isae
#endif
