#ifndef JOINT_VEL_EST
#define JOINT_VEL_EST

#include <Eigen/Dense>

class JointVelEstimator {
public:
    JointVelEstimator(int dof, double dt, double tau = 0.02);

    const Eigen::VectorXd& update(const Eigen::VectorXd& pos);

private:
    double dt_, tau_, alpha_;
    Eigen::VectorXd prev_pos_;
    Eigen::VectorXd vel_filtered_;
    bool initialized_;
};

#endif