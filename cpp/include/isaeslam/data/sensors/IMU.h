#ifndef IMU_H
#define IMU_H

#include "isaeslam/data/sensors/ASensor.h"

#include <algorithm>
#include <atomic>
#include <Eigen/Cholesky>

namespace isae {

static Eigen::Vector3d g(0, 0, -9.81);

/*!
 * @brief Config strucuture for an IMU.
 */
struct imu_config : sensor_config {
    double gyr_noise;
    double bgyr_noise;
    double acc_noise;
    double bacc_noise;
    double rate_hz;
    //! IMU clock minus camera clock [s]: IMU stamps are moved by -dt_imu_cam to the camera clock. Atomic: the back
    //! end updates it (estimate_td) while the data thread reads it
    std::atomic<double> dt_imu_cam{0};
};

/*!
 * @brief An Inertial Measurement Unit (IMU) sensor class.
 *
 * This class represents an IMU sensor that provides acceleration and gyroscope measurements.
 * It handles the preintegration of IMU data for use in SLAM systems.
 */
class IMU : public ASensor {
  public:
    IMU(Eigen::Vector3d acc, Eigen::Vector3d gyr) : ASensor("imu"), _acc(acc), _gyr(gyr) {
        _v     = Eigen::Vector3d::Zero();
        _ba    = Eigen::Vector3d::Zero();
        _bg    = Eigen::Vector3d::Zero();
        _Sigma = Eigen::Matrix<double, 9, 9>::Zero();
    }
    IMU(std::shared_ptr<imu_config> config, Eigen::Vector3d acc, Eigen::Vector3d gyr)
        : ASensor("imu"), _acc(acc), _gyr(gyr) {
        _v          = Eigen::Vector3d::Zero();
        _gyr_noise  = config->gyr_noise;
        _acc_noise  = config->acc_noise;
        _bgyr_noise = config->bgyr_noise;
        _bacc_noise = config->bacc_noise;
        _rate_hz    = config->rate_hz;
        _Sigma      = Eigen::Matrix<double, 9, 9>::Zero();
        _ba         = Eigen::Vector3d::Zero();
        _bg         = Eigen::Vector3d::Zero();
        _delta_R    = Eigen::Matrix3d::Identity();

        _eta << _gyr_noise, _gyr_noise, _gyr_noise, _acc_noise, _acc_noise, _acc_noise;
        _eta = _eta.cwiseAbs2();
        _eta = _eta * _rate_hz;
    }
    ~IMU() {}

    Eigen::Vector3d getAcc() const { return _acc; }
    Eigen::Vector3d getGyr() const { return _gyr; }

    void setBa(Eigen::Vector3d ba) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _ba = ba;
    }
    void setBg(Eigen::Vector3d bg) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _bg = bg;
    }
    void setDeltaP(const Eigen::Vector3d delta_p) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _delta_p = delta_p;
    }
    void setDeltaV(const Eigen::Vector3d delta_v) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _delta_v = delta_v;
    }
    void setDeltaR(const Eigen::Matrix3d delta_R) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _delta_R = delta_R;
    }
    void setVelocity(const Eigen::Vector3d v) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _v = v;
    }
    Eigen::Vector3d getBa() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _ba;
    }
    Eigen::Vector3d getBg() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _bg;
    }
    Eigen::Vector3d getDeltaP() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _delta_p;
    }
    Eigen::Vector3d getDeltaV() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _delta_v;
    }
    Eigen::Matrix3d getDeltaR() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _delta_R;
    }
    Eigen::Vector3d getVelocity() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _v;
    }
    Eigen::MatrixXd getCov() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _Sigma;
    }
    double getIntegratedDt() const { return _integrated_dt; }
    int getGapSteps() const { return _n_gap_steps; }

    /*!
     * @brief True if the preintegration covariance is finite and positive definite (a factor can use it)
     */
    bool hasValidCovariance() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _Sigma.rows() == 9 && _Sigma.allFinite() && Eigen::LLT<Eigen::MatrixXd>(_Sigma).info() == Eigen::Success;
    }

    /*!
     * @brief Longest integration step considered as continuous IMU data (10 nominal periods, >= 50 ms)
     */
    static double maxStepDtForRate(double rate_hz) { return std::max(0.05, 10.0 / rate_hz); }
    double maxStepDt() const { return maxStepDtForRate(_rate_hz); }

    /*!
     * @brief Mark a measurement interpolated across missing raw IMU data (set by the reader): every integration
     * step that starts or ends at it counts as a gap, however short it is
     */
    void markRawGap() { _raw_gap = true; }
    bool rawGap() const { return _raw_gap; }
    double getGyrNoise() const { return _gyr_noise; }
    double getAccNoise() const { return _acc_noise; }
    double getbGyrNoise() const { return _bgyr_noise; }
    double getbAccNoise() const { return _bacc_noise; }

    void setLastKF(std::shared_ptr<Frame> frame) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _last_kf = frame;
    }
    std::shared_ptr<Frame> getLastKF() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        if (_last_kf.lock())
            return _last_kf.lock();
        else
            return nullptr;
    }
    void setLastIMU(std::shared_ptr<IMU> imu) {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _last_IMU = imu;
    }
    std::shared_ptr<IMU> getLastIMU() {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _last_IMU;
    }

    /*!
     * @brief Process IMU data to compute pre integration deltas and covariances.
     *
     * This function processes the IMU data to compute the deltas in position, velocity, and orientation on SO(3)
     * It is based on "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry" by Forster et al.
     * Source: https://arxiv.org/abs/1512.02363
     */
    bool processIMU();

    /*!
     * @brief Dead reckoning over one step of dt: the measurement of the step start (acc, gyr, in the frame) is
     * held over the step and corrected with the biases. Moves the pose T_w_f and the velocity v (world frame).
     */
    static void deadReckon(Eigen::Affine3d &T_w_f,
                           Eigen::Vector3d &v,
                           const Eigen::Vector3d &acc,
                           const Eigen::Vector3d &gyr,
                           const Eigen::Vector3d &ba,
                           const Eigen::Vector3d &bg,
                           double dt);

    /*!
     * @brief Estimate the transformation between the Last KF and the current frame with pre integration deltas.
     */
    void estimateTransformIMU(Eigen::Affine3d &dT);

    /*!
     * @brief Integrate again from the last KF with new linearization biases.
     *
     * The raw measurements are those of the IMU chain (_last_IMU links) back to the IMU of the last KF.
     * Used when the KF biases moved too far for the first-order bias correction.
     *
     * @return false if the chain does not reach the last KF (nothing changed)
     */
    bool repropagate(const Eigen::Vector3d &ba_lin, const Eigen::Vector3d &bg_lin);

    /*!
     * @brief Biases the preintegration was computed with (those of the last KF when it started). Factors
     * correct the deltas to first order with (bias of the last KF) - (linearization bias).
     */
    Eigen::Vector3d getBaLin() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _ba_lin;
    }
    Eigen::Vector3d getBgLin() const {
        std::lock_guard<std::mutex> lock(_imu_mtx);
        return _bg_lin;
    }

    Eigen::Matrix3d _J_dR_bg; //!< Jacobian of the delta rotation w.r.t the gyro bias
    Eigen::Matrix3d _J_dv_ba; //!< Jacobian of the delta velocity w.r.t the accel bias
    Eigen::Matrix3d _J_dv_bg; //!< Jacobian of the delta velocity w.r.t the gyro bias
    Eigen::Matrix3d _J_dp_ba; //!< Jacobian of the delta position w.r.t the accel bias
    Eigen::Matrix3d _J_dp_bg; //!< Jacobian of the delta position w.r.t the gyro bias

    unsigned long long _timestamp_imu; //!< Timestamp stored to avoid dependency on the frame
    Eigen::Affine3d _T_w_f_imu;        //!< Transform from the world to the frame to avoid dependency on the frame

  private:
    /*!
     * @brief Preintegrated quantities from the last KF, with their covariance and bias Jacobians
     */
    struct Preint {
        Eigen::Matrix3d dR               = Eigen::Matrix3d::Identity();
        Eigen::Vector3d dv               = Eigen::Vector3d::Zero();
        Eigen::Vector3d dp               = Eigen::Vector3d::Zero();
        Eigen::Matrix<double, 9, 9> Sigma = Eigen::Matrix<double, 9, 9>::Zero();
        Eigen::Matrix3d J_dR_bg          = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d J_dv_ba          = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d J_dv_bg          = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d J_dp_ba          = Eigen::Matrix3d::Zero();
        Eigen::Matrix3d J_dp_bg          = Eigen::Matrix3d::Zero();
        double integrated_dt             = 0;
        int gap_steps                    = 0;
    };

    /*!
     * @brief One preintegration step (Forster et al.): the measurements acc, gyr are held over dt and
     * corrected with the linearization biases. Starting from Preint() is starting at the KF.
     */
    static void preintegrate(Preint &s,
                             const Eigen::Vector3d &acc,
                             const Eigen::Vector3d &gyr,
                             double dt,
                             const Eigen::Vector3d &ba_lin,
                             const Eigen::Vector3d &bg_lin,
                             const Vector6d &eta,
                             double max_step_dt);

    Preint getPreint() const;
    void setPreint(const Preint &s);

    // Measurements
    Eigen::Vector3d _acc; //!< Acceleration measurement
    Eigen::Vector3d _gyr; //!< Gyroscope measurement

    // IMU noise
    double _gyr_noise;  //!< Gyroscope noise
    double _bgyr_noise; //!< Gyroscope bias random walk noise
    double _acc_noise;  //!< Accelerometer noise
    double _bacc_noise; //! Accelerometer bias random walk noise
    double _rate_hz = 200; //!< IMU rate in Hz
    Vector6d _eta;      //!< Noise vector for the IMU measurements

    // States computed by processIMU()
    Eigen::Vector3d _delta_p; //!< Pre integration delta in position
    Eigen::Vector3d _delta_v; //!< Pre integration delta in velocity
    Eigen::Matrix3d _delta_R; //!< Pre integration delta in orientation (SO(3))
    Eigen::Vector3d _ba_lin = Eigen::Vector3d::Zero(); //!< Accelerometer bias of the preintegration
    Eigen::Vector3d _bg_lin = Eigen::Vector3d::Zero(); //!< Gyroscope bias of the preintegration
    Eigen::Vector3d _ba;      //!< Accelerometer bias
    Eigen::Vector3d _bg;      //! Gyroscope bias
    Eigen::Vector3d _v;       //!< Velocity of the IMU in the world frame

    // Covariance computation
    Eigen::MatrixXd _Sigma; //!< Covariance of the pre integration deltas

    // Diagnostics
    double _integrated_dt = 0; //!< Time integrated since the last KF (sum of the integration steps)
    int _n_gap_steps      = 0; //!< Steps since the last KF longer than maxStepDt() (missing IMU data)
    bool _raw_gap         = false; //!< Interpolated across missing raw IMU data (see markRawGap)

    /*!
     * @brief Longest step from prev to this measurement counted as continuous data (0: always a gap)
     */
    double stepLimit(const IMU *prev) const { return (_raw_gap || prev->_raw_gap) ? 0.0 : maxStepDt(); }

    std::shared_ptr<IMU> _last_IMU; //!< Last IMU measurement used for pre integration
    std::weak_ptr<Frame> _last_kf;  //!< Last keyframe used for pre integration

    // Mutex
    mutable std::mutex _imu_mtx;
};

/*!
 * @brief Inertial state carried across a re-initialization.
 *
 * After a visual dropout the IMU has carried the pose, so the next initialization can start from the last state
 * (pose, velocity, biases) instead of the origin with a static bias guess. The state is dead-reckoned with every
 * IMU measurement that arrives until the initialization uses it, and dropped when it is not plausible, when IMU
 * data is missing, or when the last visual pose is older than max_age (config reinit_carry_max_age_vio; dead
 * reckoning drifts quadratically).
 */
struct CarriedImuState {
    bool valid            = false;
    Eigen::Affine3d T_w_f = Eigen::Affine3d::Identity(); //!< Pose of the frame
    Eigen::Vector3d v     = Eigen::Vector3d::Zero();     //!< Velocity in the world frame
    Eigen::Vector3d ba    = Eigen::Vector3d::Zero();     //!< Accelerometer bias
    Eigen::Vector3d bg    = Eigen::Vector3d::Zero();     //!< Gyroscope bias
    Eigen::Vector3d acc   = Eigen::Vector3d::Zero();     //!< Last measurement, held over the next step
    Eigen::Vector3d gyr   = Eigen::Vector3d::Zero();
    unsigned long long ts        = 0; //!< Time of the state (ns)
    unsigned long long ts_visual = 0; //!< Time of the last visual pose (ns)
    double max_step              = 0; //!< Longest step counted as continuous IMU data (s)
    double max_age               = 0; //!< Longest time since the last visual pose (s)

    /*!
     * @brief Start from a state; false (and invalid) if it is not finite, its biases are implausible or the last
     * visual pose is too old
     */
    bool capture(const Eigen::Affine3d &T_w_f,
                 const Eigen::Vector3d &v,
                 const Eigen::Vector3d &ba,
                 const Eigen::Vector3d &bg,
                 const Eigen::Vector3d &acc,
                 const Eigen::Vector3d &gyr,
                 unsigned long long ts,
                 unsigned long long ts_visual,
                 double max_step,
                 double max_age);

    /*!
     * @brief Move the state to a new IMU measurement at ts (raw_gap: interpolated across missing raw data)
     */
    void propagate(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr, unsigned long long ts, bool raw_gap);

    /*!
     * @brief The state at ts (up to one step after the last measurement); false if there is none
     */
    bool at(unsigned long long ts, CarriedImuState &out) const;

    double age(unsigned long long t) const { return t > ts_visual ? (t - ts_visual) * 1e-9 : 0.0; }
};

/*!
 * @brief Gravity alignment and bias guess from the first IMU measurements.
 *
 * R_w_f aligns the mean specific force with the world up axis (any yaw). The window is considered static
 * when the mean angular rate stays below 0.1 rad/s, no gyroscope sample departs from the mean by more than
 * 0.05 rad/s and the accelerometer spread stays below 0.2 m/s^2 (rms); only then are the biases guessed
 * (gyroscope bias = mean angular rate, accelerometer bias = magnitude error along gravity), otherwise
 * they are left at zero for the inertial initialization to estimate.
 *
 * @return true if the window is static
 */
bool staticImuInitialization(const std::vector<Eigen::Vector3d> &accs,
                             const std::vector<Eigen::Vector3d> &gyrs,
                             Eigen::Matrix3d &R_w_f,
                             Eigen::Vector3d &ba,
                             Eigen::Vector3d &bg);

} // namespace isae

#endif