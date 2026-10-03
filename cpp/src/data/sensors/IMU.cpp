#include "isaeslam/data/sensors/IMU.h"

namespace isae {

void IMU::preintegrate(Preint &s,
                       const Eigen::Vector3d &acc,
                       const Eigen::Vector3d &gyr,
                       double dt,
                       const Eigen::Vector3d &ba_lin,
                       const Eigen::Vector3d &bg_lin,
                       const Vector6d &eta,
                       double max_step_dt) {
    const double dt22           = 0.5 * dt * dt;
    const Eigen::Vector3d acc_c = acc - ba_lin;
    const Eigen::Matrix3d dR    = geometry::exp_so3((gyr - bg_lin) * dt);
    const Eigen::Matrix3d Jr    = geometry::so3_rightJacobian((gyr - bg_lin) * dt);
    const Eigen::Matrix3d dR_dA = s.dR * geometry::skewMatrix(acc_c);

    // Covariance propagation, state order (rotation, velocity, position)
    Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();
    A.block(0, 0, 3, 3)           = dR.transpose();
    A.block(3, 0, 3, 3)           = -dR_dA * dt;
    A.block(6, 0, 3, 3)           = -dR_dA * dt22;
    A.block(6, 3, 3, 3)           = Eigen::Matrix3d::Identity() * dt;
    Eigen::Matrix<double, 9, 6> B = Eigen::Matrix<double, 9, 6>::Zero();
    B.block(0, 0, 3, 3)           = Jr * dt;
    B.block(3, 3, 3, 3)           = s.dR * dt;
    B.block(6, 3, 3, 3)           = s.dR * dt22;
    s.Sigma = A * s.Sigma * A.transpose() + B * eta.asDiagonal() * B.transpose();
    s.Sigma.block(6, 6, 3, 3) += 0.0001 * Eigen::Matrix3d::Identity() * dt; // integration cov

    // Bias Jacobians (with the previous deltas)
    s.J_dp_ba = s.J_dp_ba + s.J_dv_ba * dt - dt22 * s.dR;
    s.J_dp_bg = s.J_dp_bg + s.J_dv_bg * dt - dt22 * dR_dA * s.J_dR_bg;
    s.J_dv_ba = s.J_dv_ba - s.dR * dt;
    s.J_dv_bg = s.J_dv_bg - dR_dA * s.J_dR_bg * dt;
    s.J_dR_bg = dR.transpose() * s.J_dR_bg - Jr * dt;

    // Deltas
    s.dp = s.dp + s.dv * dt + s.dR * acc_c * dt22;
    s.dv = s.dv + s.dR * acc_c * dt;
    s.dR = s.dR * dR;

    s.integrated_dt += dt;
    s.gap_steps += (dt > max_step_dt);
}

IMU::Preint IMU::getPreint() const {
    std::lock_guard<std::mutex> lock(_imu_mtx);
    Preint s;
    s.dR            = _delta_R;
    s.dv            = _delta_v;
    s.dp            = _delta_p;
    s.Sigma         = _Sigma;
    s.J_dR_bg       = _J_dR_bg;
    s.J_dv_ba       = _J_dv_ba;
    s.J_dv_bg       = _J_dv_bg;
    s.J_dp_ba       = _J_dp_ba;
    s.J_dp_bg       = _J_dp_bg;
    s.integrated_dt = _integrated_dt;
    s.gap_steps     = _n_gap_steps;
    return s;
}

void IMU::setPreint(const Preint &s) {
    std::lock_guard<std::mutex> lock(_imu_mtx);
    _delta_R       = s.dR;
    _delta_v       = s.dv;
    _delta_p       = s.dp;
    _Sigma         = s.Sigma;
    _J_dR_bg       = s.J_dR_bg;
    _J_dv_ba       = s.J_dv_ba;
    _J_dv_bg       = s.J_dv_bg;
    _J_dp_ba       = s.J_dp_ba;
    _J_dp_bg       = s.J_dp_bg;
    _integrated_dt = s.integrated_dt;
    _n_gap_steps   = s.gap_steps;
}

bool IMU::processIMU() {

    if (!_last_kf.lock() || !_last_IMU || !_frame.lock()) {
        return false;
    }

    // Case of wrong sync, return false
    if (_frame.lock()->getTimestamp() < _last_IMU->_timestamp_imu) {
        return false;
    }

    // Update last IMU pose if available
    if (_last_IMU->getFrame()) {
        _last_IMU->_T_w_f_imu = _last_IMU->getFrame()->getFrame2WorldTransform();
    }

    // The preintegration restarts if the last measurement is in a KF
    const bool restart = _last_IMU->getFrame() && _last_IMU->getFrame()->isKeyFrame();

    // Linearization biases: those of the KF when the preintegration starts, then kept along the chain (the
    // back end may update the KF bias meanwhile; the factors correct for it, see IMUFactor)
    {
        Eigen::Vector3d ba_lin = restart ? _last_IMU->getBa() : _last_IMU->getBaLin();
        Eigen::Vector3d bg_lin = restart ? _last_IMU->getBg() : _last_IMU->getBgLin();
        std::lock_guard<std::mutex> lock(_imu_mtx);
        _ba_lin = ba_lin;
        _bg_lin = bg_lin;
        _ba     = ba_lin;
        _bg     = bg_lin;
    }

    _timestamp_imu = _frame.lock()->getTimestamp();

    // Compute increments over the true step duration, so that the preintegration always covers the same
    // time span as the factor built on it. A step much longer than the nominal period means missing IMU
    // data: it is counted, and no preintegration factor is built across it (see AOptimizer).
    const double dt       = (_timestamp_imu - _last_IMU->_timestamp_imu) * 1e-9;
    const Eigen::Vector3d acc = _last_IMU->getAcc(), gyr = _last_IMU->getGyr();

    // Dead reckoning of the frame pose and velocity (prediction)
    const double dt22      = 0.5 * dt * dt;
    Eigen::Vector3d dv     = (acc - _ba_lin) * dt;
    Eigen::Vector3d dp     = (acc - _ba_lin) * dt22;
    Eigen::Matrix3d dR     = geometry::exp_so3((gyr - _bg_lin) * dt);
    Eigen::Matrix3d R_w_fp = _last_IMU->_T_w_f_imu.rotation();
    _v                     = _last_IMU->getVelocity() + R_w_fp * dv + g * dt;

    Eigen::Affine3d T_w_f            = _last_IMU->_T_w_f_imu;
    T_w_f.affine().block(0, 0, 3, 3) = R_w_fp * dR;
    T_w_f.affine().block(0, 3, 3, 1) += _last_IMU->getVelocity() * dt + R_w_fp * dp + g * dt22;
    _frame.lock()->setWorld2FrameTransform(T_w_f.inverse());
    _T_w_f_imu = _frame.lock()->getFrame2WorldTransform();

    // Preintegration
    Preint s = restart ? Preint() : _last_IMU->getPreint();
    preintegrate(s, acc, gyr, dt, _ba_lin, _bg_lin, _eta, maxStepDt());
    setPreint(s);

    return true;
}

bool IMU::repropagate(const Eigen::Vector3d &ba_lin, const Eigen::Vector3d &bg_lin) {
    std::shared_ptr<Frame> kf = getLastKF();
    if (!kf || !kf->getIMU())
        return false;
    const IMU *kf_imu = kf->getIMU().get();

    // Measurements after the KF, back to front
    std::vector<IMU *> chain;
    IMU *cur = this;
    while (cur && cur != kf_imu) {
        chain.push_back(cur);
        cur = cur->_last_IMU.get();
    }
    if (!cur)
        return false;

    Preint s;
    const IMU *prev = kf_imu;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        double dt = ((*it)->_timestamp_imu - prev->_timestamp_imu) * 1e-9;
        preintegrate(s, prev->getAcc(), prev->getGyr(), dt, ba_lin, bg_lin, _eta, maxStepDt());
        prev = *it;
    }
    setPreint(s);
    std::lock_guard<std::mutex> lock(_imu_mtx);
    _ba_lin = ba_lin;
    _bg_lin = bg_lin;
    return true;
}

void IMU::estimateTransformIMU(Eigen::Affine3d &dT) {

    double dt                     = (_frame.lock()->getTimestamp() - _last_kf.lock()->getTimestamp()) * 1e-9;
    dT                            = Eigen::Affine3d::Identity();
    Eigen::Matrix3d R1            = _last_kf.lock()->getWorld2FrameTransform().rotation();
    dT.translation()              = _delta_p + R1 * _last_kf.lock()->getIMU()->getVelocity() * dt + 0.5 * R1 * g * dt * dt;
    dT.affine().block(0, 0, 3, 3) = _delta_R;
}

bool staticImuInitialization(const std::vector<Eigen::Vector3d> &accs,
                             const std::vector<Eigen::Vector3d> &gyrs,
                             Eigen::Matrix3d &R_w_f,
                             Eigen::Vector3d &ba,
                             Eigen::Vector3d &bg) {
    Eigen::Vector3d acc_mean = Eigen::Vector3d::Zero(), gyr_mean = Eigen::Vector3d::Zero();
    for (const auto &a : accs)
        acc_mean += a;
    for (const auto &w : gyrs)
        gyr_mean += w;
    acc_mean /= accs.size();
    gyr_mean /= gyrs.size();

    // Rotation bringing the measured specific force onto the world up axis (robust to parallel and
    // opposite vectors, unlike the Rodrigues formula built on their cross product)
    R_w_f = Eigen::Quaterniond::FromTwoVectors(acc_mean, -g).toRotationMatrix();

    double acc_var = 0, gyr_dev = 0;
    for (const auto &a : accs)
        acc_var += (a - acc_mean).squaredNorm();
    for (const auto &w : gyrs)
        gyr_dev = std::max(gyr_dev, (w - gyr_mean).norm());
    const bool is_static = gyr_mean.norm() < 0.1 && gyr_dev < 0.05 && std::sqrt(acc_var / accs.size()) < 0.2;

    ba = Eigen::Vector3d::Zero();
    bg = Eigen::Vector3d::Zero();
    if (is_static) {
        ba = acc_mean + R_w_f.transpose() * g;
        bg = gyr_mean;
    }
    return is_static;
}

} // namespace isae
