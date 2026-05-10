#ifndef MESH_H
#define MESH_H

#include <thread>
#include <vector>

#include "isaeslam/data/features/AFeature2D.h"
#include "isaeslam/data/frame.h"
#include "isaeslam/data/landmarks/ALandmark.h"
#include "utilities/geometry.h"
#include "utilities/imgProcessing.h"

namespace isae {

/*! @brief A vector of features for 2D Meshing */
typedef std::vector<std::shared_ptr<AFeature>> FeatPolygon;

/*! @brief A vector of landmarks for 3D Meshing */
typedef std::vector<std::shared_ptr<ALandmark>> LmkPolygon;

class Mesh3D;
struct Vertex;
struct Polygon;

/*!
 * @brief A class to build and update a 3D mesh from 2D features and landmarks.
 *
 * It saves all the usefull variables of the current frame used to build / update the mesh. The Mesh data structure is
 * pretty naive for now: it is a vector of polygons where each polygon is a vector of vertices. There is room for
 * improvement here. Moreover, it contains all the methods to filter the mesh and to perform raycasting on it.
 */
class Mesh3D {
  public:
    Mesh3D() = default;
    Mesh3D(double ZNCC_tsh, double max_length_tsh) : _ZNCC_tsh(ZNCC_tsh), _max_length_tsh(max_length_tsh) {}
    ~Mesh3D() = default;

    std::vector<std::shared_ptr<Polygon>> getPolygonVector() const {
        std::lock_guard<std::mutex> lock(_mesh_mtx);
        return _polygons;
    }
    std::vector<Eigen::Vector3d> getPointCloud() const {
        std::lock_guard<std::mutex> lock(_pc_mtx);
        return _point_cloud;
    }
    std::shared_ptr<Frame> getFrame() const {
        std::lock_guard<std::mutex> lock(_mesh_mtx);
        return _reference_frame;
    }

    std::unordered_map<std::shared_ptr<ALandmark>, std::shared_ptr<Vertex>> getMap() { return _map_lmk_vertex; }

    /*!
     * @brief Update the 3D mesh with a 2D Mesh of features for a given frame.
     * @param feats_polygon A vector of features to build the mesh.
     * @param frame A shared pointer to the frame containing the features.
     */
    void updateMesh(std::vector<FeatPolygon> feats_polygon, std::shared_ptr<Frame> frame);

    /*!
     * @brief Check if a triangle is valid. (i.e. no accute angles or too long edges)
     */
    bool checkTriangle(std::vector<std::shared_ptr<Vertex>> vertices);

    /*!
     * @brief Photometric check of a polygon using ZNCC on a patch on the barycenter of the polygon
     */
    bool checkPolygon(std::shared_ptr<Polygon> polygon);

    /*!
     * @brief EXPERIMENTAL Photometric check of a polygon using ZNCC on a patch on the barycenter of the polygon of a
     * given 2D area.
     */
    bool checkPolygonArea(std::shared_ptr<Polygon> polygon, double area2d);

    /*!
     * @brief EXPERIMENTAL Photometric check of a polygon using ZNCC on a patch with the full 2D polygon in it
     */
    bool checkPolygonTri(std::shared_ptr<Polygon> polygon3d, FeatPolygon polygon2d);
    void removePolygon(std::shared_ptr<Polygon> polygon) {
        std::lock_guard<std::mutex> lock(_mesh_mtx);
        for (int i = _polygons.size() - 1; i >= 0; i--) {
            if (_polygons.at(i) == polygon) {
                _polygons.erase(_polygons.begin() + i);
            }
        }
    }

    /*!
     * @brief Compute the barycenter and the normal of a polygon
     */
    void analysePolygon(std::shared_ptr<Polygon> polygon);

    /*!
     * @brief Remove polygon outliers from the 3D mesh
     */
    void filterMesh();

    /*!
     * @brief Project the mesh on camera 0 of the current frame for raycasting.
     */
    void projectMesh();

    /*!
     * @brief Generate a point cloud from the mesh with raycasting.
     */
    void generatePointCloud();

    /// Append pre-computed world-frame 3D points directly to the point cloud,
    /// bypassing ray casting. Called by MarginalDepthInjector.
    void injectDensePoints(const std::vector<Eigen::Vector3d>& pts_world);

    mutable std::mutex _mesh_mtx;
    mutable std::mutex _pc_mtx;

  private:
    std::unordered_map<std::shared_ptr<ALandmark>, std::shared_ptr<Vertex>>
        _map_lmk_vertex;                             //!< A map between landmarks and vertices
    std::vector<std::shared_ptr<Polygon>> _polygons; //!< The vector of polygons representing the 3D Mesh
    std::vector<Eigen::Vector3d> _point_cloud;       //!< The point cloud generated from the mesh
    std::unordered_map<std::shared_ptr<Polygon>, std::vector<Eigen::Vector2d>>
        _map_poly_tri2d; //!< A map between polygons and their 2D projection in the current frame

    // Storage of frame related objects
    std::shared_ptr<Frame> _reference_frame;   //!< The current frame to build / update the mesh
    std::shared_ptr<ImageSensor> _cam0, _cam1; //!< Pointers to the cameras of the current frame
    cv::Mat _img0, _img1;                      //!< The images of the cameras of the current frame
    Eigen::Affine3d _T_w_cam0;                 //!< The world to camera 0 transform of the current frame

    // Tuning parameters
    double _ZNCC_tsh       = 0.8; //!< The ZNCC threshold for photometric checks
    double _max_length_tsh = 5;   //!< The maximum length threshold for edges in the mesh
};

/*!
 * @brief A vertex in the 3D mesh, representing a landmark and its associated polygons.
 *
 * It contains the position of the vertex, its normal, and the polygons it belongs to.
 */
struct Vertex {
  public:
    Vertex() {}

    Vertex(std::shared_ptr<ALandmark> lmk) : _lmk(lmk) {}
    ~Vertex() = default;

    Eigen::Vector3d getVertexPosition() const {
        std::lock_guard<std::mutex> lock(_lmk_mtx);
        return _lmk->getPose().translation();
    }
    Eigen::Vector3d getVertexNormal() const { return _vertex_normal; }
    std::vector<std::shared_ptr<Polygon>> getPolygons() const { return _polygons; }
    std::shared_ptr<ALandmark> getLmk() const { return _lmk; }

    void addPolygon(std::shared_ptr<Polygon> polygon) { _polygons.push_back(polygon); }
    void removePolygon(std::shared_ptr<Polygon> polygon) {
        for (int i = _polygons.size() - 1; i >= 0; i--) {
            if (_polygons.at(i) == polygon) {
                _polygons.erase(_polygons.begin() + i);
            }
        }
    }

  private:
    std::shared_ptr<ALandmark> _lmk;                 //!< The landmark associated with the vertex
    Eigen::Vector3d _vertex_normal;                  //!< The normal of the vertex
    std::vector<std::shared_ptr<Polygon>> _polygons; //!< The polygons associated with the vertex

    mutable std::mutex _lmk_mtx;
};

/*!
 * @brief A polygon in the 3D mesh, representing a surface with its vertices, normal, barycenter, and covariance.
 *
 * It contains the vertices of the polygon, its normal, barycenter, covariance matrix, and a traversability score.
 */
struct Polygon : std::enable_shared_from_this<Polygon> {
  public:
    Polygon() {}

    Polygon(std::vector<std::shared_ptr<Vertex>> vertices) : _vertices(vertices) { _outlier = false; }
    ~Polygon() = default;

    void setNormal(Eigen::Vector3d normal) { _normal = normal; }
    void setBarycenter(Eigen::Vector3d barycenter) { _barycenter = barycenter; }
    void setCovariance(Eigen::Matrix2d covariance) { _covariance = covariance; }
    void setScore(double score) { _traversability_score = score; }
    void setOutlier() { _outlier = true; }

    Eigen::Vector3d getPolygonNormal() const { return _normal; }
    Eigen::Vector3d getBarycenter() const { return _barycenter; }
    Eigen::Matrix2d getCovariance() const { return _covariance; }
    std::vector<std::shared_ptr<Vertex>> getVertices() const { return _vertices; }
    double getScore() const { return _traversability_score; }
    bool isOutlier() { return _outlier; }

  private:
    Eigen::Vector3d _normal;                        //!< The normal of the polygon
    Eigen::Vector3d _barycenter;                    //!< The barycenter of the polygon
    Eigen::Matrix2d _covariance;                    //!< The covariance matrix of the polygon
    double _traversability_score;                   //!< The traversability score of the polygon i.e. the ZNCC score
    bool _outlier;                                  //!< Whether the polygon is an outlier or not
    std::vector<std::shared_ptr<Vertex>> _vertices; //!< The vertices of the polygon
};

} // namespace isae

#endif // MESH_H