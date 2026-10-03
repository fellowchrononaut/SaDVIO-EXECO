#include "isaeslam/data/maps/localmap.h"
#include <algorithm>
#include <iostream>

namespace isae {

LocalMap::LocalMap(size_t min_kf_number, size_t max_kf_number, size_t fixed_frames_number)
    : _min_kf_number(min_kf_number), _max_kf_number(max_kf_number), _fixed_frames_number(fixed_frames_number) {}

void LocalMap::addFrame(std::shared_ptr<isae::Frame> &frame) {

    // A KF has been voted, the frame is added to the local map
    // The frames are ordered from the oldest to the newest
    _localmap_mtx.lock();
    if (std::find(_frames.begin(), _frames.end(), frame) != _frames.end()) {
        _localmap_mtx.unlock();
        return;
    }
    _frames.push_back(frame);
    _localmap_mtx.unlock();

    // Add landmarks to the map
    this->pushLandmarks(frame);

    // If we have too much frames, raise the marginalization flag
    if (_frames.size() > _max_kf_number) {
        _margin_flag = true;
    } else {
        _margin_flag = false;
    }
}

void LocalMap::removeFrame(std::shared_ptr<isae::Frame> &frame) {
    // Remove the frame from the local map
    _localmap_mtx.lock();
    for (auto it = _frames.begin(); it != _frames.end(); ++it) {
        if (*it == frame) {
            it->get()->cleanLandmarks();
            it->get()->cleanSensors();
            _frames.erase(it);
            break;
        }
    }
    _localmap_mtx.unlock();
    _margin_flag = false;
}

void LocalMap::discardLastFrame() {

    // Discard features from the marginalized frame
    _frames.at(0)->cleanLandmarks();
    _frames.at(0)->cleanSensors();

    _localmap_mtx.lock();
    _removed_frame_poses.push_back(_frames.at(0)->getFrame2WorldTransform());
    _frames.pop_front();
    _localmap_mtx.unlock();

    // remove landmarks in the map without any feature
    this->removeEmptyLandmarks();
    _margin_flag = false;
    
}

void LocalMap::removeEmptyLandmarks() {
    // Remove map empty landmarks
    for (auto &tlmks : _landmarks) {
        for (std::vector<std::shared_ptr<isae::ALandmark>>::iterator it = tlmks.second.begin();
             it != tlmks.second.end();) {
            if (it->get()->getFeatures().empty()) {
                it->get()->setMarg();

                _localmap_mtx.lock();
                it = tlmks.second.erase(it);
                _localmap_mtx.unlock();
            } else {
                it++;
            }
        }
    }
}

void LocalMap::reset() {
    _localmap_mtx.lock();
    for (auto &frame : _frames) {
        frame->cleanLandmarks();
        frame->cleanSensors();
    }
    _frames.clear();
    for (auto &tlmks : _landmarks) {
        tlmks.second.clear();
    }
    _localmap_mtx.unlock();
}

bool LocalMap::computeRelativePose(std::shared_ptr<isae::Frame> &frame1,
                                   std::shared_ptr<isae::Frame> &frame2,
                                   Eigen::Affine3d &T_f1_f2,
                                   Eigen::MatrixXd &cov) {
    // Check if the local map is not empty
    if (_frames.size() < 2) {
        return false;
    }

    // Compute the relative pose between two frames
    T_f1_f2 = frame1->getFrame2WorldTransform().inverse() * frame2->getFrame2WorldTransform();

    // Select all the frames included between the two frames
    std::vector<std::shared_ptr<isae::Frame>> frames_to_add;
    for (auto &frame : _frames) {
        if (frame->getTimestamp() >= frame1->getTimestamp() && frame->getTimestamp() <= frame2->getTimestamp()) {
            frames_to_add.push_back(frame);
        }
    }

    // If we haven't found at least 2 KF, return false
    if (frames_to_add.size() < 2) {
        return false;
    }

    // Propagate the covariance
    cov = frames_to_add.at(1)->getdTCov();
    Eigen::Affine3d T_f1_fim1 =
        frame1->getFrame2WorldTransform().inverse() * frames_to_add.at(1)->getFrame2WorldTransform();

    for (uint i = 2; i < frames_to_add.size(); i++) {
        Eigen::Affine3d T_fim1_fi = frames_to_add.at(i - 1)->getFrame2WorldTransform().inverse() *
                                    frames_to_add.at(i)->getFrame2WorldTransform();

        // Jacobian formulas come from "A micro Lie Theory for state estimation in robotics" by Solà et al.
        Eigen::MatrixXd J_f1   = Eigen::MatrixXd::Identity(6, 6);
        J_f1.block<3, 3>(0, 0) = T_fim1_fi.rotation().transpose();
        J_f1.block<3, 3>(0, 3) = -T_fim1_fi.rotation() * geometry::skewMatrix(T_fim1_fi.translation());
        J_f1.block<3, 3>(3, 3) = T_fim1_fi.rotation().transpose();

        Eigen::MatrixXd J_dt = Eigen::MatrixXd::Identity(6, 6);

        T_f1_fim1 = frame1->getFrame2WorldTransform().inverse() * frames_to_add.at(i)->getFrame2WorldTransform();
        if (cov.rows() != 6 && cov.cols() != 6) {
            cov = Eigen::MatrixXd::Identity(6, 6);
            return false;
        }
        cov = J_f1 * cov * J_f1.transpose() + J_dt * frames_to_add.at(i)->getdTCov() * J_dt.transpose();
    }

    if (cov.rows() != 6 && cov.cols() != 6) {
        cov = Eigen::MatrixXd::Identity(6, 6);
        return false;
    }

    return true;
}

} // namespace isae
