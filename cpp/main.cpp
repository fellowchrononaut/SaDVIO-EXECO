#include "isaeslam/slamCore.h"
#include "isaeslam/stereo/MarginalDepthInjector.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <thread>

#include <execinfo.h>
#include <unistd.h>

namespace {
std::atomic<bool> g_stop{false};
extern "C" void onStop(int) { g_stop = true; }
// A crash prints the call stack (batch runs: stdout.log) before the process dies as it would have
extern "C" void onCrash(int sig) {
    void *frames[64];
    const int n         = backtrace(frames, 64);
    const char header[] = "\n*** crash (signal), backtrace:\n";
    if (write(STDERR_FILENO, header, sizeof(header) - 1) < 0) {
    }
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}
} // namespace

int main(int argc, char **argv) {

    // Create the SLAM parameter object
    std::string config_folder_path                   = argv[1];
    std::shared_ptr<isae::SLAMParameters> slam_param = std::make_shared<isae::SLAMParameters>(config_folder_path);

    // Create SLAM object
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
    else if (slam_param->_config.slam_mode == "nofovvio")
        SLAM = std::make_shared<isae::SLAMNonOverlappingFovVIO>(slam_param);

    // Create the data grabber
    std::string data_folder_path = argv[2];
    // The SLAM threads run until the process is stopped (batch runs: pkill at the end of the data). On SIGTERM /
    // SIGINT the dense map finishes (queued keyframes, final files), then the process exits
    std::signal(SIGTERM, onStop);
    std::signal(SIGINT, onStop);
    std::signal(SIGSEGV, onCrash);
    std::signal(SIGABRT, onCrash);
    std::thread([SLAM] {
        while (!g_stop)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        SLAM->finish(); // the window's keyframes to the loop closure (a revisit at the very end)
        if (SLAM->_depth_injector)
            SLAM->_depth_injector->finish();
        std::_Exit(0);
    }).detach();

    isae::EUROCGrabber grabber(data_folder_path, slam_param->getDataProvider());

    // Load images
    grabber.load_filenames();

    // Load image thread
    std::thread datagrabber(&isae::EUROCGrabber::addAllFrames, grabber);
    datagrabber.detach();

    if (slam_param->_config.multithreading) {

        // Launch front end thread
        std::thread frontend_thread(&isae::SLAMCore::runFrontEnd, SLAM);
        frontend_thread.detach();

        // Launch back end thread
        std::thread backend_thread(&isae::SLAMCore::runBackEnd, SLAM);
        backend_thread.join();

    } else {

        // Launch full odom thread
        std::thread odom_thread(&isae::SLAMCore::runFullOdom, SLAM);
        odom_thread.join();
    }

    return 0;
}