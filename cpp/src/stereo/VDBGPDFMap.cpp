#include "isaeslam/stereo/VDBGPDFMap.h"
#include <yaml-cpp/yaml.h>
#include <stdexcept>

// loadVDBGPDFPreset only needs yaml-cpp; the map itself is compiled only with -DISAESLAM_WITH_VDBGPDF=ON
// (cmake/ISAESLAM_VDBGPDF.cmake).
namespace isae {

void loadVDBGPDFPreset(const std::string& path, VDBGPDFMapConfig& c) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        throw std::runtime_error("[VDB-GPDF] cannot read preset " + path + ": " + e.what());
    }
    // accept the upstream ROS 2 layout as well as a flat file
    const YAML::Node n = root["vdb_gpdf_mapping_node"] && root["vdb_gpdf_mapping_node"]["ros__parameters"]
                             ? root["vdb_gpdf_mapping_node"]["ros__parameters"]
                             : root;
    auto get = [&](const char* key, auto& v) {
        if (n[key])
            v = n[key].as<std::remove_reference_t<decltype(v)>>();
    };
    get("fusion", c.fusion);
    get("sdf_trunc", c.sdf_trunc);
    get("space_carving", c.space_carving);
    get("distance_method", c.distance_method);
    get("voxel_size_local", c.voxel_size_local);
    get("voxel_overlapping", c.voxel_overlapping);
    get("voxel_downsample", c.voxel_downsample);
    get("voxel_size_global", c.voxel_size_global);
    get("variance_method", c.variance_method);
    get("variance_cap", c.variance_cap);
    get("variance_on_surface", c.variance_on_surface);
    get("surface_normal_method", c.surface_normal_method);
    get("surface_normal_num", c.surface_normal_num);
    get("surface_value", c.surface_value);
    get("query_iterval", c.query_iterval);
    get("query_trunc_in", c.query_trunc_in);
    get("query_trunc_out", c.query_trunc_out);
    get("freespace_iterval", c.freespace_iterval);
    get("freespace_trunc_out", c.freespace_trunc_out);
    get("query_downsample", c.query_downsample);
    get("map_lambda_scale", c.map_lambda_scale);
    get("map_noise", c.map_noise);
    get("smooth_param", c.smooth_param);
    get("min_scan_range", c.min_scan_range);
    get("max_scan_range", c.max_scan_range);
    get("range_weight_ref", c.range_weight_ref);
    get("range_weight_exp", c.range_weight_exp);
    get("fill_holes", c.fill_holes);
    get("recon_min_weight", c.recon_min_weight);
    get("debug_print", c.debug_print);
    if (c.fusion != "gpdf" && c.fusion != "tsdf")
        throw std::runtime_error("[VDB-GPDF] fusion must be gpdf or tsdf, got '" + c.fusion + "' in " + path);
}

} // namespace isae

#ifdef ISAESLAM_WITH_VDBGPDF

#include "VDBVolume.h"
#include <algorithm>
#include <cmath>
#include "utils.h"
#include <iostream>

namespace isae {

struct VDBGPDFMap::Impl {
    explicit Impl(const vdb_gpdf::VDBVolumeParams& p) : volume(p) {}
    vdb_gpdf::VDBVolume volume;
};

VDBGPDFMap::VDBGPDFMap(const VDBGPDFMapConfig& cfg) : _cfg(cfg) {
    openvdb::initialize();
    vdb_gpdf::VDBVolumeParams p;
    p.debug_print           = cfg.debug_print;
    p.use_color             = false; // SaDVIO feeds grey stereo; colour GPs are not trained
    p.sdf_trunc             = cfg.sdf_trunc;
    p.space_carving         = cfg.space_carving;
    p.distance_method       = cfg.distance_method;
    p.voxel_size_local      = cfg.voxel_size_local;
    p.voxel_overlapping     = cfg.voxel_overlapping;
    p.voxel_downsample      = cfg.voxel_downsample;
    p.voxel_size_global     = cfg.voxel_size_global;
    p.variance_method       = cfg.variance_method;
    p.variance_cap          = cfg.variance_cap;
    p.variance_on_surface   = cfg.variance_on_surface;
    p.surface_normal_method = cfg.surface_normal_method;
    p.surface_normal_num    = cfg.surface_normal_num;
    p.surface_value         = cfg.surface_value;
    p.query_iterval         = cfg.query_iterval;
    p.query_trunc_in        = cfg.query_trunc_in;
    p.query_trunc_out       = cfg.query_trunc_out;
    p.freespace_iterval     = cfg.freespace_iterval;
    p.freespace_trunc_out   = cfg.freespace_trunc_out;
    p.query_downsample      = cfg.query_downsample;
    p.map_lambda_scale      = cfg.map_lambda_scale;
    p.map_noise             = cfg.map_noise;
    p.smooth_param          = cfg.smooth_param;
    _impl = std::make_unique<Impl>(p);
    std::cout << "[VDB-GPDF] fusion=" << cfg.fusion << " voxel local/global " << cfg.voxel_size_local << "/"
              << cfg.voxel_size_global << " m, lambda " << cfg.map_lambda_scale << ", range "
              << cfg.min_scan_range << "-" << cfg.max_scan_range << " m" << std::endl;
}

VDBGPDFMap::~VDBGPDFMap() = default;

void VDBGPDFMap::integrate(const std::vector<Eigen::Vector3d>& points_world, const Eigen::Vector3d& origin) {
    // upstream mapper range gate (min/max_scan_range around the sensor)
    std::vector<Eigen::Vector3d> pts;
    pts.reserve(points_world.size());
    const double r2_min = static_cast<double>(_cfg.min_scan_range) * _cfg.min_scan_range;
    const double r2_max = static_cast<double>(_cfg.max_scan_range) * _cfg.max_scan_range;
    for (const auto& p : points_world) {
        const double r2 = (p - origin).squaredNorm();
        if (r2 > r2_min && r2 < r2_max)
            pts.push_back(p);
    }
    if (pts.empty())
        return;
    // upstream indexes colors[i] for every point (CreateLocalDisantceField), so pass one (unused) colour each
    const std::vector<openvdb::Vec3i> colors(pts.size(), openvdb::Vec3i(0, 0, 0));
    if (_cfg.fusion == "tsdf") {
        std::vector<float> w;
        if (_cfg.range_weight_ref > 0) {
            w.reserve(pts.size());
            for (const auto& p : pts)
                w.push_back(std::min(1.0f, static_cast<float>(std::pow(_cfg.range_weight_ref / (p - origin).norm(),
                                                                        _cfg.range_weight_exp))));
        }
        _impl->volume.Integrate(pts, colors, origin, common::WeightFunction::constant_weight, w.empty() ? nullptr : &w);
        return;
    }
    // Integrate appends to these outputs, so they must be fresh for every scan (as in the upstream mapper)
    std::vector<Eigen::Vector3d> global_gp_points, local_gp_points, local_query_points;
    std::vector<openvdb::Vec3i> global_gp_colors;
    std::vector<double> local_query_dis;
    _impl->volume.Integrate(pts, colors, origin, common::WeightFunction::constant_weight, global_gp_points,
                            global_gp_colors, local_gp_points, local_query_points, local_query_dis);
}

void VDBGPDFMap::accumulateInto(VDBGPDFMap& dst, const Eigen::Affine3d& T_dst_src, float sign) const {
    const auto& src_vol = _impl->volume;
    auto& dst_vol       = dst._impl->volume;
    auto src_d          = src_vol.gsdf_->getConstAccessor();
    auto dst_d          = dst_vol.gsdf_->getAccessor();
    auto dst_w          = dst_vol.weights_->getAccessor();
    const openvdb::math::Transform& src_xf = src_vol.weights_->transform();
    const openvdb::math::Transform& dst_xf = dst_vol.weights_->transform();
    // Only the band around the surfaces is merged: the saturated free space (|d| at the truncation, written along
    // every ray by space carving with a weight per ray) of one volume would otherwise erase the surfaces of the
    // volumes it overlaps
    const float band = 0.999f * _cfg.sdf_trunc;
    for (auto it = src_vol.weights_->cbeginValueOn(); it; ++it) {
        const float w = *it;
        if (!(w > 0.f))
            continue;
        if (std::abs(src_d.getValue(it.getCoord())) >= band)
            continue;
        // Voxel centres follow the vendored convention (GetVoxelCenter, marching cubes): indexToWorld + half a voxel
        const openvdb::Coord& ijk = it.getCoord();
        const float d             = src_d.getValue(ijk);
        const openvdb::Vec3d p    = src_xf.indexToWorld(ijk) + 0.5 * src_xf.voxelSize()[0];
        const Eigen::Vector3d q   = T_dst_src * Eigen::Vector3d(p.x(), p.y(), p.z());
        const openvdb::Vec3d x    = dst_xf.worldToIndex(openvdb::Vec3d(q.x(), q.y(), q.z()) - 0.5 * dst_xf.voxelSize()[0]);
        const openvdb::Coord base(static_cast<int>(std::floor(x.x())), static_cast<int>(std::floor(x.y())),
                                  static_cast<int>(std::floor(x.z())));
        const double fx = x.x() - base.x(), fy = x.y() - base.y(), fz = x.z() - base.z();
        for (int c = 0; c < 8; c++) {
            const double a = ((c & 1) ? fx : 1 - fx) * ((c & 2) ? fy : 1 - fy) * ((c & 4) ? fz : 1 - fz);
            if (a < 1e-6)
                continue;
            const openvdb::Coord n = base.offsetBy(c & 1, (c >> 1) & 1, (c >> 2) & 1);
            const float dw         = sign * static_cast<float>(a) * w;
            const float W          = dst_w.isValueOn(n) ? dst_w.getValue(n) : 0.f;
            const float W2         = W + dw;
            // Weights count observations (one per ray or query, constant weighting) and reach hundreds: float
            // cancellation leaves ~1e-4 of them after a removal. Below 1% of one observation a voxel is empty
            if (W2 <= 1e-2f) {
                dst_w.setValueOff(n, 0.f);
                dst_d.setValueOff(n, dst_vol.gsdf_->background());
                continue;
            }
            const float D = W > 0.f ? dst_d.getValue(n) : 0.f;
            dst_d.setValueOn(n, (D * W + d * dw) / W2);
            dst_w.setValueOn(n, W2);
        }
    }
}

size_t VDBGPDFMap::activeVoxels() const {
    return static_cast<size_t>(_impl->volume.weights_->activeVoxelCount());
}

DenseMesh VDBGPDFMap::mesh() const {
    auto [vertices, triangles, colors] = _impl->volume.ExtractTriangleMesh(_cfg.fill_holes, _cfg.recon_min_weight);
    (void)colors;
    DenseMesh out;
    out.vertices = std::move(vertices);
    out.faces    = std::move(triangles);
    out.computeFaceNormals();
    return out;
}

} // namespace isae

#else // !ISAESLAM_WITH_VDBGPDF

#include <stdexcept>

namespace isae {

// Without VDB-GPDF the map cannot be built (MarginalDepthInjector refuses dense_mesh_method=vdbgpdf with a clear
// error); these definitions only let the library link, since MarginalDepthInjector references the class.
struct VDBGPDFMap::Impl {};

VDBGPDFMap::VDBGPDFMap(const VDBGPDFMapConfig& cfg) : _cfg(cfg) {
    throw std::runtime_error("VDBGPDFMap needs a build with -DISAESLAM_WITH_VDBGPDF=ON");
}

VDBGPDFMap::~VDBGPDFMap() = default;

void VDBGPDFMap::integrate(const std::vector<Eigen::Vector3d>&, const Eigen::Vector3d&) {
    throw std::runtime_error("VDBGPDFMap needs a build with -DISAESLAM_WITH_VDBGPDF=ON");
}

void VDBGPDFMap::accumulateInto(VDBGPDFMap&, const Eigen::Affine3d&, float) const {
    throw std::runtime_error("VDBGPDFMap needs a build with -DISAESLAM_WITH_VDBGPDF=ON");
}

size_t VDBGPDFMap::activeVoxels() const {
    return 0;
}

DenseMesh VDBGPDFMap::mesh() const {
    throw std::runtime_error("VDBGPDFMap needs a build with -DISAESLAM_WITH_VDBGPDF=ON");
}

} // namespace isae

#endif // ISAESLAM_WITH_VDBGPDF
