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
    get("fill_holes", c.fill_holes);
    get("recon_min_weight", c.recon_min_weight);
    get("debug_print", c.debug_print);
    if (c.fusion != "gpdf" && c.fusion != "tsdf")
        throw std::runtime_error("[VDB-GPDF] fusion must be gpdf or tsdf, got '" + c.fusion + "' in " + path);
}

} // namespace isae

#ifdef ISAESLAM_WITH_VDBGPDF

#include "VDBVolume.h"
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
        _impl->volume.Integrate(pts, colors, origin, common::WeightFunction::constant_weight);
        return;
    }
    // Integrate appends to these outputs, so they must be fresh for every scan (as in the upstream mapper)
    std::vector<Eigen::Vector3d> global_gp_points, local_gp_points, local_query_points;
    std::vector<openvdb::Vec3i> global_gp_colors;
    std::vector<double> local_query_dis;
    _impl->volume.Integrate(pts, colors, origin, common::WeightFunction::constant_weight, global_gp_points,
                            global_gp_colors, local_gp_points, local_query_points, local_query_dis);
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

#endif // ISAESLAM_WITH_VDBGPDF
