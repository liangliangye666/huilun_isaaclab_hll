#include "joint_vel_est.h"

JointVelEstimator::JointVelEstimator(int dof, double dt, double tau)
    : dt_(dt),
      tau_(tau),
      alpha_(dt / (tau + dt)),
      prev_pos_(Eigen::VectorXd::Zero(dof)),
      vel_filtered_(Eigen::VectorXd::Zero(dof)),
      initialized_(false)
{}

const Eigen::VectorXd& JointVelEstimator::update(const Eigen::VectorXd& pos) {
    if (!initialized_) {
        prev_pos_ = pos;
        initialized_ = true;
        vel_filtered_.setZero();
        return vel_filtered_;
    }

    Eigen::VectorXd vel_raw = (pos - prev_pos_) / dt_;
    prev_pos_ = pos;

    vel_filtered_ += alpha_ * (vel_raw - vel_filtered_);

    return vel_filtered_;
}
