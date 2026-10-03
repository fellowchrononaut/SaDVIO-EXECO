#ifndef AIMAGEPROVIDER_H
#define AIMAGEPROVIDER_H

#include <mutex>  // EXECO_QUEUE_MUTEX
#include <fstream>
#include <iostream>
#include <deque>
#include <queue>

#include <Eigen/Core>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/data/sensors/DoubleSphere.h"
#include "isaeslam/data/sensors/Fisheye.h"
#include "isaeslam/data/sensors/IMU.h"
#include "isaeslam/slamParameters.h"

namespace isae {

/*!
 * @brief ADataProvider class for managing data from various sensors.
 *
 * This class is responsible for loading sensor configurations, processing frames, and managing the queue of frames.
 */
class ADataProvider {
  public:
    ADataProvider(std::string path, Config slam_config);

    /*!
     * @brief Return the next frame from the queue.
     */
    std::shared_ptr<Frame> next();

    std::vector<std::shared_ptr<cam_config>> getCamConfigs() { return _cam_configs; };
    std::shared_ptr<imu_config> getIMUConfig() { return _imu_config; }
    int getNCam() { return _ncam; };
    double getStereoSyncTolNs() const { return _stereo_sync_tol_ns; }

    /*!
     * @brief From raw image to sensor objects.
     */
    std::vector<std::shared_ptr<ImageSensor>> createImageSensors(const std::vector<cv::Mat> &imgs,
                                                                 const std::vector<cv::Mat> &masks = {});

    /*!
     * @brief From raw IMU measurements to an IMU sensor object.
     */
    std::shared_ptr<IMU> createImuSensor(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

    /*!
     * @brief Create a frame from sensors and timestamp (ns) and add it to the queue.
     */
    void addFrameToTheQueue(std::vector<std::shared_ptr<ASensor>> sensors, unsigned long long time);

    /*!
     * @brief Number of frames waiting in the queue (non-blocking, unlike next()).
     */
    size_t queueSize() {
        std::lock_guard<std::mutex> lock(_frame_queue_mutex);
        return _frame_queue.size();
    }

    void addFrameToTheQueue(std::shared_ptr<Frame> frame);
    std::mutex _frame_queue_mutex; // EXECO_QUEUE_MUTEX

  protected:
    void loadSensorsConfiguration(const std::string &path);
    void loadCamConfig(YAML::Node cam_node);
    void loadIMUConfig(YAML::Node imu_node);

    std::shared_ptr<imu_config> _imu_config;               //!< IMU configuration
    std::vector<std::shared_ptr<cam_config>> _cam_configs; //!< Vector of camera configurations
    int _ncam;                                             //!< Number of Image Sensors
    std::queue<std::shared_ptr<Frame>> _frame_queue;       //!< Queue of frames to be processed
    Config _slam_config;                                   //!< SLAM configuration
    int _nframes;                                          //!< Frame counter
    double _stereo_sync_tol_ns = 20.0 * 1e6;             //!< Stereo pair timestamp tolerance [ns]
};

/*!
 * @brief Merges IMU measurements and images in timestamp order (shared by the offline and ROS readers).
 *
 * IMU measurements are given with stamps already in the camera clock (t_cam = t_imu - dt_imu_cam) and
 * in the body frame. When an image is emitted, all IMU measurements strictly before it are emitted
 * first as IMU-only frames; the image frame keeps the image stamp and carries an IMU measurement at
 * that time (the measurement itself if one coincides, else a linear interpolation between the
 * surrounding ones), so the preintegration ends exactly at the image. An image outside the IMU stream
 * carries no IMU.
 */
class ImuImageMerger {
  public:
    explicit ImuImageMerger(std::shared_ptr<ADataProvider> prov) : _prov(prov) {}

    /*!
     * @brief Queue an IMU measurement. Returns false (and drops it) if its stamp is not increasing.
     */
    bool addImu(long long ts, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);

    /*!
     * @brief True when an IMU measurement at or after t has been received.
     */
    bool imuReached(long long t) const;

    /*!
     * @brief Emit the IMU measurements before t_img, then the image frame. Returns false (nothing
     * emitted for the image) if t_img is not after the last emitted IMU measurement.
     */
    bool emitImageFrame(long long t_img, const std::vector<std::shared_ptr<ImageSensor>> &images);

    int droppedImu() const { return _n_dropped; }

  private:
    struct ImuSample {
        long long ts;
        Eigen::Vector3d acc, gyr;
    };

    std::shared_ptr<ADataProvider> _prov;
    std::deque<ImuSample> _pending; //!< IMU measurements not yet emitted
    ImuSample _last;                //!< Last IMU measurement emitted (for interpolation)
    bool _has_last  = false;
    int _n_dropped  = 0;
};

/*!
 * @brief EUROCGrabber class for loading and processing frames from raw data in the EUROC format
 *
 * This class is responsible for loading filenames, timestamps, and IMU data from raw data files in the EUROC format,
 * and adding frames to the data provider.
 */
class EUROCGrabber {
  public:
    EUROCGrabber(std::string folder_path, std::shared_ptr<ADataProvider> prov)
        : _folder_path(folder_path), _prov(prov), _merger(prov) {}

    /*!
     * @brief Load filenames and timestamps from .csv files.
     */
    void load_filenames();

    /*!
     * @brief Add the frame that comes next in time to the queue of the data provider.
     *
     * Emits the next image (pair) preceded by the IMU measurements before it (see ImuImageMerger). IMU
     * stamps are moved to the camera clock with the current dt_imu_cam (t_cam = t_imu - dt_imu_cam, as in
     * the ROS reader), so an online estimate applies to the data read afterwards; the reader stays at most
     * _max_queued_frames frames ahead of the SLAM. Without an IMU configuration (VO modes) the IMU file is
     * ignored. Returns false when no image is left.
     */
    bool addNextFrame();

    /*!
     * @brief Add all frames from the queue of the data provider until no more frames are available.
     */
    void addAllFrames() {
        bool not_over = true;
        while (not_over) {
            not_over = addNextFrame();
        }
    }

  private:
    std::string _folder_path; //!< Path to the folder containing the dataset
    std::queue<std::string> _cam0_filename_queue, _cam1_filename_queue; //!< Queues for camera filenames
    std::queue<long long> _cam0_timestamp_queue, _cam1_timestamp_queue; //!< Queues for camera timestamps

    std::shared_ptr<ADataProvider> _prov; //!< Pointer to the data provider
    ImuImageMerger _merger;               //!< Orders IMU measurements and images

    /*!
     * @brief Raw IMU measurement (IMU clock), moved to the merger only when needed so that the current
     * dt_imu_cam applies (it can be estimated online)
     */
    struct RawImu {
        long long ts;
        Eigen::Vector3d acc, gyr;
    };
    std::deque<RawImu> _imu_raw;
    size_t _max_queued_frames = 1000; //!< The reader waits while the SLAM has this many frames to process
};

} // namespace isae

#endif // AIMAGEPROVIDER_H
