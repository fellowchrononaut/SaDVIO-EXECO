#include <gtest/gtest.h>

#include <filesystem>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "isaeslam/data/landmarks/Point3D.h"
#include "isaeslam/data/mesh/mesh.h"
#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/stereo/AnchoredMaps.h"
#include "isaeslam/stereo/VDBGPDFMap.h"
#include "utilities/geometry.h"

namespace isae {

class MeshTest : public testing::Test {

    /* Toy example of a mesh with a single triangular face
     *       lmk_0 --- lmk_1
     *         \       /
     *          \     /
     *           lmk_2
     */

  public:
    void SetUp() override {
        std::srand(12345u); // same random state for every test, whatever the run order
        srand(12345u); // fixed seed: reproducible tests (was time-seeded)

        // Set Frame
        _frame0 = std::shared_ptr<Frame>(new Frame());
        Eigen::Affine3d T_f0_w = Eigen::Affine3d::Identity();
        T_f0_w.translation() << 0.05, 0.0, 3.0;
        Eigen::Affine3d T_f_camright = Eigen::Affine3d::Identity();
        T_f_camright.translation() << 0.1, 0.0, 0.0;

        // Set Sensors
        _K        = Eigen::Matrix3d::Identity();
        _K(0, 0)  = 320;
        _K(1, 1)  = 320;
        _K(0, 2)  = 320;
        _K(1, 2)  = 240;
        cv::Mat img_left = cv::imread("../tests/img_left.png", cv::IMREAD_GRAYSCALE);
        cv::Mat img_right = cv::imread("../tests/img_right.png", cv::IMREAD_GRAYSCALE);
        _sensor0l = std::shared_ptr<Camera>(new Camera(img_left, _K));
        _sensor0l->setFrame2SensorTransform(Eigen::Affine3d::Identity());
        _sensor0r = std::shared_ptr<Camera>(new Camera(img_right, _K));
        _sensor0r->setFrame2SensorTransform(T_f_camright.inverse());
        std::vector<std::shared_ptr<ImageSensor>> sensors_frame0;
        sensors_frame0.push_back(_sensor0l);
        sensors_frame0.push_back(_sensor0r);
        _frame0->init(sensors_frame0, 0);
        _frame0->setWorld2FrameTransform(T_f0_w);

        // Set landmark 0
        Eigen::Affine3d T_w_l0 = Eigen::Affine3d::Identity();
        T_w_l0.translation()   = Eigen::Vector3d(1, 1, -1);
        std::vector<std::shared_ptr<AFeature>> feat_vec_0;
        _lmk_0 = std::shared_ptr<Point3D>(new Point3D(T_w_l0, feat_vec_0));
        std::vector<Eigen::Vector2d> projection_0_0_l, projection_0_0_r;
        _sensor0l->project(_lmk_0->getPose(), _lmk_0->getModel(), projection_0_0_l);
        _sensor0r->project(_lmk_0->getPose(), _lmk_0->getModel(), projection_0_0_r);
        _feat_0 = std::shared_ptr<Point2D>(new Point2D(projection_0_0_l));
        _lmk_0->addFeature(_feat_0);
        _sensor0l->addFeature("pointxd", _feat_0);

        // Set landmark 1
        Eigen::Affine3d T_w_l1 = Eigen::Affine3d::Identity();
        T_w_l1.translation()   = Eigen::Vector3d(-1, -1, -1);
        std::vector<std::shared_ptr<AFeature>> feat_vec_1;
        _lmk_1 = std::shared_ptr<Point3D>(new Point3D(T_w_l1, feat_vec_1));
        std::vector<Eigen::Vector2d> projection_0_1_l, projection_0_1_r;
        _sensor0l->project(_lmk_1->getPose(), _lmk_1->getModel(), projection_0_1_l);
        _sensor0r->project(_lmk_1->getPose(), _lmk_1->getModel(), projection_0_1_r);
        _feat_1 = std::shared_ptr<Point2D>(new Point2D(projection_0_1_l));
        _lmk_1->addFeature(_feat_1);
        _sensor0l->addFeature("pointxd", _feat_1);

        // Set landmark 2
        Eigen::Affine3d T_w_l2 = Eigen::Affine3d::Identity();
        T_w_l2.translation()   = Eigen::Vector3d(1, -1, -1);
        std::vector<std::shared_ptr<AFeature>> feat_vec_2;
        _lmk_2 = std::shared_ptr<Point3D>(new Point3D(T_w_l2, feat_vec_2));
        std::vector<Eigen::Vector2d> projection_0_2_l, projection_0_2_r;
        _sensor0l->project(_lmk_2->getPose(), _lmk_2->getModel(), projection_0_2_l);
        _sensor0r->project(_lmk_2->getPose(), _lmk_2->getModel(), projection_0_2_r);
        _feat_2 = std::shared_ptr<Point2D>(new Point2D(projection_0_2_l));
        _lmk_2->addFeature(_feat_2);
        _sensor0l->addFeature("pointxd", _feat_2);

        // Set landmark 3
        Eigen::Affine3d T_w_l3 = Eigen::Affine3d::Identity();
        T_w_l3.translation()   = Eigen::Vector3d(-1, 1, -1);
        std::vector<std::shared_ptr<AFeature>> feat_vec_3;
        _lmk_3 = std::shared_ptr<Point3D>(new Point3D(T_w_l3, feat_vec_3));
        std::vector<Eigen::Vector2d> projection_0_3_l, projection_0_3_r;
        _sensor0l->project(_lmk_3->getPose(), _lmk_3->getModel(), projection_0_3_l);
        _sensor0r->project(_lmk_2->getPose(), _lmk_2->getModel(), projection_0_3_r);
        _feat_3 = std::shared_ptr<Point2D>(new Point2D(projection_0_3_l));
        _lmk_3->addFeature(_feat_3);
        _sensor0l->addFeature("pointxd", _feat_3);

        // Init the 2D mesh
        _2D_triangle.push_back(_feat_0);
        _2D_triangle.push_back(_feat_1);
        _2D_triangle.push_back(_feat_2);

        // Init the new 2D mesh for update
        _new_2D_triangle.push_back(_feat_0);
        _new_2D_triangle.push_back(_feat_1);
        _new_2D_triangle.push_back(_feat_3);
        
        // Uncomment to check if the triangle is well projected
        // cv::Mat img_colorl, img_colorr;
        // cv::cvtColor(img_left.clone(), img_colorl, cv::COLOR_GRAY2RGB);
        // cv::cvtColor(img_right.clone(), img_colorr, cv::COLOR_GRAY2RGB);
        // cv::circle(img_colorl, cv::Point2d(projection_0_1_l.at(0)(0), projection_0_1_l.at(0)(1)), 11, cv::Scalar(0, 255, 0));
        // cv::circle(img_colorr, cv::Point2d(projection_0_1_r.at(0)(0), projection_0_1_r.at(0)(1)), 11, cv::Scalar(0, 255, 0));
        // cv::circle(img_colorl, cv::Point2d(projection_0_0_l.at(0)(0), projection_0_0_l.at(0)(1)), 11, cv::Scalar(255, 0, 0));
        // cv::circle(img_colorr, cv::Point2d(projection_0_0_r.at(0)(0), projection_0_0_r.at(0)(1)), 11, cv::Scalar(255, 0, 0));
        // cv::circle(img_colorl, cv::Point2d(projection_0_2_l.at(0)(0), projection_0_2_l.at(0)(1)), 11, cv::Scalar(0, 0, 255));
        // cv::circle(img_colorr, cv::Point2d(projection_0_2_r.at(0)(0), projection_0_2_r.at(0)(1)), 11, cv::Scalar(0, 0, 255));
        // cv::imwrite("_0.png", img_colorl);
        // cv::imwrite("_1.png", img_colorr);
    }

    std::shared_ptr<Frame> _frame0;
    std::shared_ptr<Point2D> _feat_0;
    std::shared_ptr<Point3D> _lmk_0;
    std::shared_ptr<Point2D> _feat_1;
    std::shared_ptr<Point3D> _lmk_1;
    std::shared_ptr<Point2D> _feat_2;
    std::shared_ptr<Point3D> _lmk_2;
    std::shared_ptr<Point2D> _feat_3;
    std::shared_ptr<Point3D> _lmk_3;

    FeatPolygon _2D_triangle;
    FeatPolygon _new_2D_triangle;

    Eigen::Matrix3d _K;
    std::shared_ptr<ImageSensor> _sensor0l;
    std::shared_ptr<ImageSensor> _sensor0r;
};

TEST_F(MeshTest, MeshTestBase) {

    std::vector<FeatPolygon> mesh_2d;
    mesh_2d.push_back(_2D_triangle);
    Mesh3D mesh_3d;
    mesh_3d.updateMesh(mesh_2d, _frame0);

    // The size of the mesh is correct
    ASSERT_EQ(mesh_3d.getPolygonVector().size(), 1);

    // Try checkPolygonTri
    FeatPolygon feat_tri2d;
    feat_tri2d.push_back(_feat_0);
    feat_tri2d.push_back(_feat_1);
    feat_tri2d.push_back(_feat_2);
    bool tri_valid = mesh_3d.checkPolygonTri(mesh_3d.getPolygonVector().at(0), feat_tri2d);

    ASSERT_EQ(tri_valid, true);

    // The normal is correct
    Eigen::Vector3d normal_gt(0, 0, -1);
    Eigen::Vector3d normal_mesh = mesh_3d.getPolygonVector().at(0)->getPolygonNormal();
    ASSERT_EQ((normal_gt.cross(normal_mesh)).norm(), 0);

    // Update of the mesh
    // Test if we have one new polygon
    std::vector<FeatPolygon> new_mesh_2d;
    new_mesh_2d.push_back(_new_2D_triangle);
    mesh_3d.updateMesh(new_mesh_2d, _frame0);
    ASSERT_EQ(mesh_3d.getPolygonVector().size(), 2);
    ASSERT_EQ(mesh_3d.getMap().at(_lmk_1)->getPolygons().size(), 2);
    ASSERT_EQ(mesh_3d.getMap().at(_lmk_2)->getPolygons().size(), 1);

    // Test if we have a marginalized landmark and no new polygons
    _lmk_2->setMarg();
    new_mesh_2d.clear();
    mesh_3d.updateMesh(new_mesh_2d, _frame0);

    // Test if there is only one polygon left
    ASSERT_EQ(mesh_3d.getPolygonVector().size(), 1);

    // Test if the good polygon is remaining
    std::vector<std::shared_ptr<Vertex>> vertices;
    ASSERT_EQ(mesh_3d.getPolygonVector().at(0)->getVertices().at(0)->getLmk(), _lmk_0);
    ASSERT_EQ(mesh_3d.getPolygonVector().at(0)->getVertices().at(1)->getLmk(), _lmk_1);
    ASSERT_EQ(mesh_3d.getPolygonVector().at(0)->getVertices().at(2)->getLmk(), _lmk_3);

    // Test if the vertices have only one remaining polygon
    ASSERT_EQ(mesh_3d.getMap().at(_lmk_1)->getPolygons().size(), 1);
    ASSERT_EQ(mesh_3d.getMap().at(_lmk_1)->getPolygons().at(0), mesh_3d.getPolygonVector().at(0));
}

TEST_F(MeshTest, pointInTriangleTest) {
    // Build a random triangle
    Eigen::Vector3d A = Eigen::Vector3d::Random();
    Eigen::Vector3d B = Eigen::Vector3d::Random();
    Eigen::Vector3d C = Eigen::Vector3d::Random();
    std::vector<Eigen::Vector3d> triangle;
    triangle.push_back(A);
    triangle.push_back(B);
    triangle.push_back(C);

    // Build a point inside the triangle
    Eigen::Vector3d P = A + 0.4 * (B - A) + 0.5 * (C - A);
    ASSERT_EQ(geometry::pointInTriangle(P, triangle), true);

    // Build a point outside the triangle
    Eigen::Vector3d Pp = A + 0.8 * (B - A) + 0.5 * (C - A);
    ASSERT_EQ(geometry::pointInTriangle(Pp, triangle), false);
}

TEST_F(MeshTest, covTest) {

    // Build a random triangle
    Eigen::Vector3d A = Eigen::Vector3d::Random();
    Eigen::Vector3d B = Eigen::Vector3d::Random();
    Eigen::Vector3d C = Eigen::Vector3d::Random();
    std::vector<Eigen::Vector3d> triangle;
    triangle.push_back(A);
    triangle.push_back(B);
    triangle.push_back(C);

    // Turn it into a 2D triangle
    Eigen::Vector3d b1 = B - A;
    Eigen::Vector3d b2 = C - A;

    // Find 2D coordinates of the points
    Eigen::Vector3d b1_norm = b1.normalized();
    Eigen::Vector2d u1(b1.norm(), 0);
    Eigen::Vector2d u2(b1_norm.dot(b2), (b2 - b1_norm.dot(b2) * b1_norm).norm());

    // Check if the area of both triangle is the same
    double a_3d = 0.5 * (b1.cross(b2)).norm();
    Eigen::Matrix2d tri;
    tri << u1.x(), u2.x(), u1.y(), u2.y();
    double a_2d = 0.5 * tri.determinant();

    ASSERT_NEAR(a_3d, a_2d, 1e-6);

    // Now compute the numerical covariance of the triangle
    std::vector<Eigen::Vector2d> tri_2d;
    tri_2d.push_back(Eigen::Vector2d(0, 0));
    tri_2d.push_back(u1);
    tri_2d.push_back(u2);
    tri << u1.x(), u2.x(), u1.y(), u2.y();
    Eigen::Vector2d barycenter = (1.0 / 3.0) * (u1 + u2);

    Eigen::Matrix2d cov_num  = Eigen::Matrix2d::Zero();
    Eigen::Vector2d bary_num = Eigen::Vector2d::Zero();
    int n_pts                = 100000;

    for (int i = 0; i < n_pts; i++) {
        Eigen::Vector2d rand_pt = 2 * Eigen::Vector2d::Random();
        while (!geometry::pointInTriangle(rand_pt, tri_2d))
            rand_pt = 2 * Eigen::Vector2d::Random();

        cov_num = cov_num + (rand_pt - barycenter) * (rand_pt - barycenter).transpose();
        bary_num += rand_pt;
    }

    // Try to compute the barycenter
    bary_num /= n_pts;
    cov_num = cov_num / n_pts;

    ASSERT_NEAR((bary_num - barycenter).norm(), 0, 1e-1);

    // Compute covariance
    Eigen::Matrix2d M = geometry::cov2dTriangle(tri_2d);
    std::cout << cov_num << "\n" << std::endl;
    std::cout << M << std::endl;
    ASSERT_NEAR((M - cov_num).sum(), 0, 1e-2);

    // Try equi triangle (the eigen values must be equal)
    u1 << std::cos(M_PI / 3), std::sin(M_PI / 3);
    u2 << 1, 0;
    tri_2d.clear();
    tri_2d.push_back(Eigen::Vector2d(0, 0));
    tri_2d.push_back(u1);
    tri_2d.push_back(u2);
    M = geometry::cov2dTriangle(tri_2d);
    std::cout << geometry::cov2dTriangle(tri_2d) << std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double ratio = svd.singularValues()(0) / svd.singularValues()(1);
    ASSERT_NEAR(ratio,1, 1e-3);
}

namespace {

AnchoredMapsConfig anchoredConfig(const std::string &name) {
    AnchoredMapsConfig c;
    c.mesh_path  = (std::filesystem::temp_directory_path() / name / "mesh.ply").string();
    c.save_every = 0;
    c.submap_kfs = 2;
    return c;
}

Eigen::Affine3d pose(double x, double yaw) {
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.linear()        = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    T.translation()   = Eigen::Vector3d(x, 0.5, 0);
    return T;
}

} // namespace

// Anchored maps (doc/dense_anchored_maps.md): a keyframe's geometry, kept in its own frame, is placed at its corrected
// pose: the loop closure's correction of that keyframe times the odometry pose it was built at
TEST(AnchoredMapsTest, anchorsArePlacedAtTheirCorrectedPoses) {
    AnchoredMaps maps(anchoredConfig("anchored_place"));
    DenseMesh tri;
    tri.vertices = {Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(1, 0, 1), Eigen::Vector3d(0, 1, 1)};
    tri.faces    = {Eigen::Vector3i(0, 1, 2)};
    tri.computeFaceNormals();
    const Eigen::Affine3d T1 = pose(0, 0), T2 = pose(2, 0.3);
    maps.addKeyframe(100, 0, 0, T1, tri);
    maps.addKeyframe(200, 0, 0, T2, tri);
    Eigen::Affine3d C = Eigen::Affine3d::Identity(); // the loop closure moved keyframe 200 by 0.4 m and 5 deg
    C.linear()        = Eigen::AngleAxisd(0.087, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    C.translation()   = Eigen::Vector3d(0.4, -0.1, 0);
    maps.setKeyframeCorrection([&](unsigned long long ts, Eigen::Affine3d &out) {
        if (ts != 200)
            return false;
        out = C;
        return true;
    });
    maps.save();
    const DenseMesh &g = maps.globalMesh();
    ASSERT_EQ(g.vertices.size(), 6u);
    ASSERT_EQ(g.faces.size(), 2u);
    EXPECT_EQ(g.faces[1], Eigen::Vector3i(3, 4, 5));
    for (int k = 0; k < 3; k++) {
        EXPECT_LT((g.vertices[k] - T1 * tri.vertices[k]).norm(), 1e-12) << "no correction: odometry pose";
        EXPECT_LT((g.vertices[3 + k] - C * T2 * tri.vertices[k]).norm(), 1e-12) << "corrected pose";
    }
    EXPECT_LT((g.face_normals[1] - (C * T2).linear() * tri.face_normals[0]).norm(), 1e-12);
}

// Fused methods: a submap takes submap_kfs keyframes, and a new one starts whenever the odometry world changes
// (re-initialization: segment, window moved by a loop: epoch), since its keyframes' relative poses must hold
TEST(AnchoredMapsTest, submapsStartWhenFullOrWhenTheWorldChanges) {
    AnchoredMaps maps(anchoredConfig("anchored_submaps"));
    int meshed = 0;
    const AnchoredMaps::Mesher mesher = [&](AnchoredMaps::Anchor &a) {
        meshed++;
        return a.mesh;
    };
    Eigen::Affine3d T_a_c;
    std::vector<int> ids;
    const int segment[] = {0, 0, 0, 0, 1, 1};
    const int epoch[]   = {0, 0, 0, 1, 1, 1};
    for (int k = 0; k < 6; k++)
        ids.push_back(maps.submapFor(10 * k, segment[k], epoch[k], pose(k, 0.1 * k), T_a_c, mesher).id);
    // KF 0, 1: submap 0 (full); KF 2: submap 1; KF 3: new epoch, submap 2; KF 4: new segment, submap 3; KF 5: same
    EXPECT_EQ(ids, (std::vector<int>{0, 0, 1, 2, 3, 3}));
    EXPECT_EQ(meshed, 3) << "each submap left behind was closed with its final mesh";
    // The keyframe's camera in the anchor's frame: the relative odometry pose
    EXPECT_LT((T_a_c.matrix() - (pose(4, 0.4).inverse() * pose(5, 0.5)).matrix()).norm(), 1e-12);
    maps.closeOpenSubmap(mesher);
    EXPECT_EQ(meshed, 4);
}

namespace {

VDBGPDFMapConfig tsdfConfig() {
    VDBGPDFMapConfig c;
    c.fusion            = "tsdf";
    c.sdf_trunc         = 0.1f;
    c.space_carving     = false;
    c.voxel_size_global = 0.05f;
    c.min_scan_range    = 0.05f;
    c.max_scan_range    = 20.f;
    c.recon_min_weight  = 0.1f;
    return c;
}

// A 2 m x 2 m wall 2 m in front of the origin (z = 2), 1 cm grid, as a camera at the origin sees it
std::vector<Eigen::Vector3d> wall() {
    std::vector<Eigen::Vector3d> pts;
    for (double x = -1; x <= 1; x += 0.01)
        for (double y = -1; y <= 1; y += 0.01)
            pts.emplace_back(x, y, 2.0);
    return pts;
}

double medianDistanceToPlane(const DenseMesh &m, const Eigen::Affine3d &T) {
    // distance of the mesh vertices to the wall moved by T (normal T.R z, through T * (0, 0, 2))
    const Eigen::Vector3d n = T.linear() * Eigen::Vector3d::UnitZ(), c = T * Eigen::Vector3d(0, 0, 2);
    std::vector<double> d;
    for (const auto &v : m.vertices)
        d.push_back(std::abs(n.dot(v - c)));
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    return d[d.size() / 2];
}

} // namespace

// Fused view (doc/dense_anchored_maps.md): a submap volume merged into the fused volume at a pose lands there; merged
// twice at the same pose it is averaged (one surface, not two layers); removed with the same splat it leaves nothing
TEST(AnchoredMapsTest, volumesMergeAverageAndRemoveExactly) {
    std::unique_ptr<VDBGPDFMap> sub, fused;
    try {
        sub = std::make_unique<VDBGPDFMap>(tsdfConfig());
    } catch (const std::runtime_error &) {
        GTEST_SKIP() << "build without VDB-GPDF";
    }
    sub->integrate(wall(), Eigen::Vector3d::Zero());
    fused = std::make_unique<VDBGPDFMap>(tsdfConfig());
    Eigen::Affine3d T = Eigen::Affine3d::Identity(); // placed 0.37 m and 20 deg away (off the voxel lattice)
    T.linear()        = Eigen::AngleAxisd(0.35, Eigen::Vector3d(0.3, 1, 0.2).normalized()).toRotationMatrix();
    T.translation()   = Eigen::Vector3d(0.37, -0.21, 0.13);

    sub->accumulateInto(*fused, T, 1.f);
    const DenseMesh once = fused->mesh();
    ASSERT_GT(once.faces.size(), 1000u);
    EXPECT_LT(medianDistanceToPlane(once, T), 0.02) << "the merged surface is the wall at its new pose";

    sub->accumulateInto(*fused, T, 1.f); // a revisit at the same pose: averaged, the surface does not move
    const DenseMesh twice = fused->mesh();
    EXPECT_LT(medianDistanceToPlane(twice, T), 0.02);
    EXPECT_NEAR(static_cast<double>(twice.faces.size()), static_cast<double>(once.faces.size()),
                0.02 * once.faces.size())
        << "one surface, not two layers";

    sub->accumulateInto(*fused, T, -1.f);
    sub->accumulateInto(*fused, T, -1.f);
    EXPECT_EQ(fused->activeVoxels(), 0u) << "removal undoes the merge exactly";
}

namespace {

// A 6 x 5 x 3 m room (walls, floor, ceiling) as a camera at T_w_c sees it: points on its surfaces within 60 deg of
// the optical axis, in the camera frame
std::vector<Eigen::Vector3d> roomView(const Eigen::Affine3d &T_w_c) {
    std::vector<Eigen::Vector3d> pts;
    const Eigen::Vector3d o = T_w_c.translation();
    const Eigen::Affine3d T_c_w = T_w_c.inverse();
    for (double u = -0.6; u <= 0.6; u += 0.004)
        for (double v = -0.45; v <= 0.45; v += 0.004) {
            const Eigen::Vector3d dir = T_w_c.linear() * Eigen::Vector3d(u, v, 1).normalized();
            double best = 1e9;
            for (int a = 0; a < 3; a++) {
                const double lo[3] = {-3, -2.5, 0}, hi[3] = {3, 2.5, 3};
                for (double wall : {lo[a], hi[a]}) {
                    if (std::abs(dir(a)) < 1e-9)
                        continue;
                    const double t = (wall - o(a)) / dir(a);
                    if (t > 0.1 && t < best)
                        best = t;
                }
            }
            pts.push_back(T_c_w * (o + best * dir));
        }
    return pts;
}

} // namespace

// Real-data shape of the fused view: two submaps of a room, each integrated from three cameras (TSDF with space
// carving, as the stereo_tsdf preset) in its own anchor frame, placed and merged. The fused mesh must lie on the
// placed submap meshes (both are the same room)
TEST(AnchoredMapsTest, fusedRoomLiesOnThePlacedSubmaps) {
    VDBGPDFMapConfig c = tsdfConfig();
    c.space_carving    = true;
    std::unique_ptr<VDBGPDFMap> fused;
    try {
        fused = std::make_unique<VDBGPDFMap>(c);
    } catch (const std::runtime_error &) {
        GTEST_SKIP() << "build without VDB-GPDF";
    }
    auto cam = [](double x, double yaw) {
        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.linear() = (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                      Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX()))
                         .toRotationMatrix(); // optical axis horizontal
        T.translation() = Eigen::Vector3d(x, 0.3 * x, 1.4);
        return T;
    };
    std::vector<Eigen::Vector3d> union_pts, fused_pts;
    for (int s = 0; s < 2; s++) {
        VDBGPDFMap sub(c);
        const Eigen::Affine3d T_w_anchor = cam(-1.0 + 1.2 * s, 0.4 + 1.9 * s);
        for (int k = 0; k < 3; k++) {
            const Eigen::Affine3d T_w_c = cam(-1.0 + 1.2 * s + 0.3 * k, 0.4 + 1.9 * s + 0.25 * k);
            const Eigen::Affine3d T_a_c = T_w_anchor.inverse() * T_w_c;
            std::vector<Eigen::Vector3d> pa;
            for (const auto &p : roomView(T_w_c))
                pa.push_back(T_a_c * p);
            sub.integrate(pa, T_a_c.translation());
        }
        for (const auto &v : sub.mesh().vertices)
            union_pts.push_back(T_w_anchor * v);
        sub.accumulateInto(*fused, T_w_anchor, 1.f);
    }
    fused_pts = fused->mesh().vertices;
    ASSERT_GT(fused_pts.size(), 1000u);
    // Distance of each mesh's vertices to the true room surfaces
    auto toRoom = [](const Eigen::Vector3d &p) {
        return std::min({std::abs(p.x() + 3), std::abs(p.x() - 3), std::abs(p.y() + 2.5), std::abs(p.y() - 2.5),
                         std::abs(p.z()), std::abs(p.z() - 3)});
    };
    auto median = [&](const std::vector<Eigen::Vector3d> &pts) {
        std::vector<double> d;
        for (const auto &p : pts)
            d.push_back(toRoom(p));
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        return d[d.size() / 2];
    };
    const double mu = median(union_pts), mf = median(fused_pts);
    EXPECT_LT(mu, 0.03) << "placed submap meshes";
    EXPECT_LT(mf, 0.03) << "fused mesh (union " << mu << ")";
}

} // namespace isae