#include "rosVisualizer.h"
#include "sensorSubscriber.h"

#include "isaeslam/slamCore.h"
#include <rclcpp/rclcpp.hpp>

#include <cstdlib>

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    // Create the SLAM parameter object
    std::shared_ptr<isae::SLAMParameters> slam_param = std::make_shared<isae::SLAMParameters>(argv[1]);

    // Create the ROS Visualizer
    // The viewer topics (enable_visu) and the map interface for a map server (ros_map_interface) share one node
    std::shared_ptr<RosVisualizer> prosviz;
    if (slam_param->_config.enable_visu || slam_param->_config.ros_map_interface)
        prosviz = std::make_shared<RosVisualizer>(slam_param->_config.enable_visu,
                                                  slam_param->_config.ros_map_interface == 1);
    else
        prosviz = nullptr;

    std::shared_ptr<isae::SLAMCore> SLAM;

    if (slam_param->_config.slam_mode == "bimono")
        SLAM = std::make_shared<isae::SLAMBiMono>(slam_param);
    else if (slam_param->_config.slam_mode == "mono")
        SLAM = std::make_shared<isae::SLAMMono>(slam_param);
    else if (slam_param->_config.slam_mode == "nofov")
        SLAM = std::make_shared<isae::SLAMNonOverlappingFov>(slam_param);
    else if (slam_param->_config.slam_mode == "bimonovio")
        SLAM = std::make_shared<isae::SLAMBiMonoVIO>(slam_param);
    else if (slam_param->_config.slam_mode == "monovio")
        SLAM = std::make_shared<isae::SLAMMonoVIO>(slam_param);

    if (slam_param->_config.multithreading) {

        // Launch front end thread
        std::thread frontend_thread(&isae::SLAMCore::runFrontEnd, SLAM);
        frontend_thread.detach();

        // Launch back end thread
        std::thread backend_thread(&isae::SLAMCore::runBackEnd, SLAM);
        backend_thread.detach();

    } else {

        // Launch full odom thread
        std::thread odom_thread(&isae::SLAMCore::runFullOdom, SLAM);
        odom_thread.detach();
    }

    // Start the vizu thread
    if (prosviz) {
        std::thread vizu_thread(&RosVisualizer::runVisualizer, prosviz, SLAM);
        vizu_thread.detach();
    }

    // Start the sensor subscriber
    std::shared_ptr<SensorSubscriber> sensor_subscriber =
        std::make_shared<SensorSubscriber>(slam_param->getDataProvider());

    // Start a thread for providing new measurements to the SLAM
    std::thread sync_thread(&SensorSubscriber::sync_process, sensor_subscriber);

    rclcpp::spin(sensor_subscriber);

    // Shutdown (Ctrl-C): the dense map finishes (queued keyframes, final files); the SLAM threads never return
    if (SLAM->_depth_injector)
        SLAM->_depth_injector->finish();
    std::_Exit(0);
}
