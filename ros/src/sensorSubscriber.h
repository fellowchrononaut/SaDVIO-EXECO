#include "isaeslam/dataproviders/adataprovider.h"

#include <cmath>
#include <mutex>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <cv_bridge/cv_bridge.h>

inline cv::Mat getGrayImageFromMsg(const sensor_msgs::msg::Image &img_msg) {
    // Get and prepare images
    cv_bridge::CvImagePtr ptr;
    try {
        ptr = cv_bridge::toCvCopy(img_msg, "mono8");
    } catch (cv_bridge::Exception &e) {
        std::cout << "\n\n\ncv_bridge exeception: %s\n\n\n" << e.what() << std::endl;
    }

    return ptr->image;
}

inline cv::Mat getGrayImageFromMsg(const sensor_msgs::msg::CompressedImage &img_msg) {
    // Get and prepare images
    cv_bridge::CvImagePtr ptr;
    try {
        ptr = cv_bridge::toCvCopy(img_msg, "mono8");
    } catch (cv_bridge::Exception &e) {
        std::cout << "\n\n\ncv_bridge exeception: %s\n\n\n" << e.what() << std::endl;
    }
    cv::Mat img = ptr->image;
    cv::resize(img, img, cv::Size(1920, 1080));

    return img;
}

/*!
 * @brief ROS2 reader for images (raw or compressed) and IMU messages.
 *
 * The sync thread hands the measurements to the SLAM in timestamp order through an
 * isae::ImuImageMerger, as the offline reader does: an image (pair) is emitted once the IMU stream has
 * passed its timestamp, preceded by the IMU messages before it, and carries an IMU measurement
 * interpolated at the image time. IMU stamps are moved to the camera clock with dt_imu_cam.
 */
template <typename ImageMsg> class SensorSubscriberT : public rclcpp::Node {

  public:
    SensorSubscriberT(std::shared_ptr<isae::ADataProvider> prov, const rclcpp::QoS &img_qos, const rclcpp::QoS &imu_qos)
        : Node("sensor_subscriber"), _prov(prov), _merger(prov) {

        _subscription_left = this->create_subscription<ImageMsg>(
            _prov->getCamConfigs().at(0)->ros_topic,
            img_qos,
            std::bind(&SensorSubscriberT::subLeftImage, this, std::placeholders::_1));
        if (_prov->getNCam() == 2)
            _subscription_right = this->create_subscription<ImageMsg>(
                _prov->getCamConfigs().at(1)->ros_topic,
                img_qos,
                std::bind(&SensorSubscriberT::subRightImage, this, std::placeholders::_1));
        if (_prov->getIMUConfig())
            _subscription_imu = this->create_subscription<sensor_msgs::msg::Imu>(
                _prov->getIMUConfig()->ros_topic,
                imu_qos,
                std::bind(&SensorSubscriberT::subIMU, this, std::placeholders::_1));
    }

    void subLeftImage(const ImageMsg &img_msg) {
        std::lock_guard<std::mutex> lock(_img_mutex);
        _imgs_bufl.push(img_msg);
    }

    void subRightImage(const ImageMsg &img_msg) {
        std::lock_guard<std::mutex> lock(_img_mutex);
        _imgs_bufr.push(img_msg);
    }

    void subIMU(const sensor_msgs::msg::Imu &imu_msg) {
        std::lock_guard<std::mutex> lock(_imu_mutex);
        _imu_buf.push(imu_msg);
    }

    static long long stampNs(const builtin_interfaces::msg::Time &stamp) {
        return (long long)stamp.sec * 1000000000LL + (long long)stamp.nanosec;
    }

    void getImuInfoFromMsg(const sensor_msgs::msg::Imu &imu_msg, Eigen::Vector3d &acc, Eigen::Vector3d &gyr) {

        // Extract the acceleration and gyroscope values from the IMU message
        double ax = imu_msg.linear_acceleration.x;
        double ay = imu_msg.linear_acceleration.y;
        double az = imu_msg.linear_acceleration.z;
        double gx = imu_msg.angular_velocity.x;
        double gy = imu_msg.angular_velocity.y;
        double gz = imu_msg.angular_velocity.z;

        // Sensor frame; ADataProvider::createImuSensor rotates them into the body frame (T_BS)
        acc = Eigen::Vector3d(ax, ay, az);
        gyr = Eigen::Vector3d(gx, gy, gz);
    }

    void sync_process() {
        std::cout << "\nStarting the measurements reader thread!\n";

        const bool stereo           = (_prov->getNCam() == 2);
        const long long stereo_tol  = (long long)_prov->getStereoSyncTolNs(); // from dataset YAML
        size_t waiting_warned       = 0;

        while (true) {

            // IMU messages go to the merger, in the body frame and the camera clock
            if (_prov->getIMUConfig()) {
                std::lock_guard<std::mutex> lock(_imu_mutex);
                while (!_imu_buf.empty()) {
                    Eigen::Vector3d acc, gyr;
                    getImuInfoFromMsg(_imu_buf.front(), acc, gyr);
                    long long t = stampNs(_imu_buf.front().header.stamp) -
                                  std::llround(_prov->getIMUConfig()->dt_imu_cam.load() * 1e9);
                    if (!_merger.addImu(t, acc, gyr))
                        std::cout << "\n Throw IMU -- non-increasing stamp : " << t << "\n";
                    _imu_buf.pop();
                }
            }

            // Next image (pair), emitted once the IMU stream has passed it
            std::vector<ImageMsg> msgs;
            long long t_img = 0;
            {
                std::lock_guard<std::mutex> lock(_img_mutex);

                // Drop stereo images without a partner within the sync tolerance
                while (stereo && !_imgs_bufl.empty() && !_imgs_bufr.empty()) {
                    long long d = stampNs(_imgs_bufl.front().header.stamp) - stampNs(_imgs_bufr.front().header.stamp);
                    if (d < -stereo_tol) {
                        _imgs_bufl.pop();
                        std::cout << "\n Throw img0 -- Sync error : " << d << "\n";
                    } else if (d > stereo_tol) {
                        _imgs_bufr.pop();
                        std::cout << "\n Throw img1 -- Sync error : " << d << "\n";
                    } else
                        break;
                }

                bool have_image = !_imgs_bufl.empty() && (!stereo || !_imgs_bufr.empty());
                if (have_image) {
                    t_img = stampNs(_imgs_bufl.front().header.stamp);
                    if (!_prov->getIMUConfig() || _merger.imuReached(t_img)) {
                        msgs.push_back(_imgs_bufl.front());
                        _imgs_bufl.pop();
                        if (stereo) {
                            msgs.push_back(_imgs_bufr.front());
                            _imgs_bufr.pop();
                        }
                        waiting_warned = 0;
                    } else if (_imgs_bufl.size() >= 50 && _imgs_bufl.size() >= 2 * waiting_warned) {
                        std::cout << "\n Waiting for IMU: " << _imgs_bufl.size() << " images buffered\n";
                        waiting_warned = _imgs_bufl.size();
                    }
                }
            }

            if (msgs.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            std::vector<cv::Mat> imgs;
            for (const auto &m : msgs)
                imgs.push_back(getGrayImageFromMsg(m));
            if (!_merger.emitImageFrame(t_img, _prov->createImageSensors(imgs)))
                std::cout << "\n Throw image -- older than the IMU already processed : " << t_img << "\n";
        }

        std::cout << "\n Bag reader SyncProcess thread is terminating!\n";
    }

    std::shared_ptr<isae::ADataProvider> _prov;
    isae::ImuImageMerger _merger; //!< Only used by the sync thread

    std::queue<ImageMsg> _imgs_bufl, _imgs_bufr;
    std::queue<sensor_msgs::msg::Imu> _imu_buf;
    std::mutex _img_mutex, _imu_mutex;

    typename rclcpp::Subscription<ImageMsg>::SharedPtr _subscription_left;
    typename rclcpp::Subscription<ImageMsg>::SharedPtr _subscription_right;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _subscription_imu;
};

/*!
 * @brief Raw images. A deep IMU history so that bursts during bag playback are not dropped.
 */
class SensorSubscriber : public SensorSubscriberT<sensor_msgs::msg::Image> {
  public:
    SensorSubscriber(std::shared_ptr<isae::ADataProvider> prov)
        : SensorSubscriberT<sensor_msgs::msg::Image>(prov, rclcpp::QoS(10), rclcpp::QoS(1000)) {}
};

/*!
 * @brief Compressed images, best-effort QoS.
 */
class SensorSubscriberCompressed : public SensorSubscriberT<sensor_msgs::msg::CompressedImage> {
  public:
    SensorSubscriberCompressed(std::shared_ptr<isae::ADataProvider> prov)
        : SensorSubscriberT<sensor_msgs::msg::CompressedImage>(
              prov,
              rclcpp::QoS(rclcpp::KeepLast(10)).best_effort().durability_volatile(),
              rclcpp::QoS(rclcpp::KeepLast(1000)).best_effort().durability_volatile()) {}
};
