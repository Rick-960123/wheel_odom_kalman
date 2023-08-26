//
// Created by meng on 2021/2/19.
//
#include "eskf.h"
#include "../3rd/sophus/se3.hpp"

constexpr double kDegree2Radian = M_PI / 180.0;

Eigen::Matrix3d BuildSkewMatrix(const Eigen::Vector3d& vec){
    Eigen::Matrix3d matrix;
    matrix << 0.0,     -vec[2],   vec[1],
              vec[2],    0.0,     -vec[0],
              -vec[1],   vec[0],    0.0;

    return matrix;
}

// TODO include IMU's orientation into this EKF (as if it was a sun sensor.

void init_eskf()
{
  n.param<std::string>("/robot_id", robot_id, "ZR1001");
  n.param<std::string>(robot_id + "/imu0", imu_name, "imu0");
  n.param<std::vector<double>>(robot_id + "/" + imu_name + "/T_i_b", V_T_i_b, std::vector<double>());
  T_i_b = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(V_T_i_b.data());
  n.param<std::vector<double>>(robot_id + "/" + imu_name + "/acc_sm", V_R_acc_mea, std::vector<double>());
  R_acc_mea = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(V_R_acc_mea.data());
  n.param<std::vector<double>>(robot_id + "/" + imu_name + "/gyr_sm", V_R_gyr_mea, std::vector<double>());
  R_gyr_mea = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(V_R_gyr_mea.data());

  n.param<std::vector<double>>(robot_id + "/" + imu_name + "/acc_bias", acc_bias, std::vector<double>());
  n.param<std::vector<double>>(robot_id + "/" + imu_name + "/gyr_bias", gyr_bias, std::vector<double>());

  n.param<double>(robot_id + "/" + imu_name + "/sigma_acc_noise", sigma_nua, 0.1);
  n.param<double>(robot_id + "/" + imu_name + "/sigma_acc_bias", sigma_xa, 0.1);

  n.param<double>(robot_id + "/" + imu_name + "/sigma_gyr_noise", sigma_nug, 0.01);
  n.param<double>(robot_id + "/" + imu_name + "/sigma_gyr_bias", sigma_xg, 0.01);

  n.param<double>(robot_id + "/" + imu_name + "update_rate", hz, 10.0);

  n.param<int>("num_data", filter.num_data, 10);  // num init pts
  // gravity vector
  n.param<double>("gravity", filter.g, 9.81);

  // encoder slip model, %slip
  n.param<double>("slip_k", filter.k, 0.05);

  R_i_enc = T_enc_to_imu.block<3, 3>(0, 0);
  t_i_enc = T_enc_to_imu.block<3, 1>(0, 3);

  filter.dt = 1.0 / hz;

  for (int i = 0; i < 3; i++)
  {
    filter.Q(i, i) = sigma_nua * sigma_nua;
    filter.Q(3 + i, 3 + i) = sigma_nug * sigma_nug;
    filter.Q(6 + i, 6 + i) = sigma_xa * sigma_xa;
    filter.Q(9 + i, 9 + i) = sigma_xg * sigma_xg;
  }

  filter.Ra = Eigen::Matrix<double, 3, 3>::Identity(3, 3) * (sigma_nua * sigma_nua);

  ROS_INFO("filter.Q:\n %s", matrixToString(filter.Q).c_str());
  ROS_INFO("filter.R:\n %s", matrixToString(filter.Ra).c_str());
}

void relocalization(double& x, double& y, double& z, double& roll, double& pitch, double& yaw)
{
  geometry_msgs::Quaternion q;
  q = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
  state << x, y, z, 0, 0, 0, q.w, q.x, q.y, q.z, state[9], state[10], state[11], state[12], state[13], state[14];
  enc_meas << x, y, z, yaw, q.w, q.x, q.y, q.z;
}

void init_state()
{
  // compute initial orientation
  Eigen::Vector3d g_b = sum_accel / filter.num_data;

  // initial roll (roll) and pitch (pitch)
  double roll = atan2(-g_b[1], -g_b[2]);
  double pitch = atan2(g_b[0], sqrt(g_b[1] * g_b[1] + g_b[2] * g_b[2]));

  // set initial yaw to zero
  double yaw = 0;

  // q is navigation to body transformation: R_bi
  // YPR: R_ib = R(yaw)R(pitch)R(Roll)
  // RPY: R_bi = R(-Roll)R(-Pitch)R(-yaw)
  Eigen::Quaternion<double> quat = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                                   Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                                   Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
  quat = quat.inverse();

  // compute gyroscope biases
  Eigen::Vector3d gyr_biases = sum_gyro / filter.num_data;

  std::vector<double> p{ 0.0, 0.0, 0.0 };
  std::vector<double> q{ quat.w(), quat.x(), quat.y(), quat.z() };

  // initialize state: [p, v, b, a_bias, w_bias] = [position, velocity, quaternion, accel bias, gyro bias],  size 16
  state << p[0], p[1], p[2], 0, 0, 0, q[0], q[1], q[2], q[3], acc_bias[0], acc_bias[1], acc_bias[2], gyr_biases[0],
      gyr_biases[1], gyr_biases[2];

  int T = filter.num_data / hz;  // number of measurements over rate of IMU

  // initialize covariance
  cov.block<2, 2>(6, 6) = (sigma_nua / filter.g) * (sigma_nua / filter.g) / T *
                          Eigen::Matrix<double, 2, 2>::Identity();  // orientation uncertainty
  cov.block<3, 3>(12, 12) =
      (sigma_nug) * (sigma_nug) / T * Eigen::Matrix<double, 3, 3>::Identity();  // gyro bias uncertainty
  // TODO accelerometer initial uncertainty (after initializing accelerometer bias)
  enc_meas << p[0], p[1], p[2], 0, q[0], q[1], q[2], q[3];

  ROS_INFO("filter.P:\n %s", matrixToString(cov).c_str());

  ROS_INFO("filter.z:\n %s", matrixToString(enc_meas).c_str());
  initialized = true;
}

ESKF::ESKF(ros::NodeHandle& n) {
    n.param<std::string>("/robot_id", robot_id, "ZR1001");
    n.param<std::string>(robot_id + "/imu0", imu_name, "imu0");
    n.param<std::vector<double>>(robot_id + "/" + imu_name + "/T_i_b", V_T_i_b, std::vector<double>());
    T_i_b = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(V_T_i_b.data());
    n.param<std::vector<double>>(robot_id + "/" + imu_name + "/acc_sm", V_R_acc_mea, std::vector<double>());
    R_acc_mea = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(V_R_acc_mea.data());
    n.param<std::vector<double>>(robot_id + "/" + imu_name + "/gyr_sm", V_R_gyr_mea, std::vector<double>());
    R_gyr_mea = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(V_R_gyr_mea.data());

    n.param<std::vector<double>>(robot_id + "/" + imu_name + "/acc_bias", acc_bias, std::vector<double>());
    n.param<std::vector<double>>(robot_id + "/" + imu_name + "/gyr_bias", gyr_bias, std::vector<double>());

    n.param<double>(robot_id + "/" + imu_name + "/sigma_acc_noise", sigma_nua, 0.1);
    n.param<double>(robot_id + "/" + imu_name + "/sigma_acc_bias", sigma_xa, 0.1);

    n.param<double>(robot_id + "/" + imu_name + "/sigma_gyr_noise", sigma_nug, 0.01);
    n.param<double>(robot_id + "/" + imu_name + "/sigma_gyr_bias", sigma_xg, 0.01);

    n.param<double>(robot_id + "/" + imu_name + "update_rate", hz, 10.0);

    n.param<double>("gravity", gravity, 9.81);

    n.param<double>("slip_k", filter.k, 0.05);

    R_i_enc = T_enc_to_imu.block<3, 3>(0, 0);
    t_i_enc = T_enc_to_imu.block<3, 1>(0, 3);

    double cov_prior_posi = node["covariance"]["prior"]["posi"].as<double>();
    double cov_prior_vel = node["covariance"]["prior"]["vel"].as<double>();
    double cov_prior_ori = node["covariance"]["prior"]["ori"].as<double>();
    double cov_prior_epsilon = node["covariance"]["prior"]["epsilon"].as<double>();
    double cov_prior_delta = node["covariance"]["prior"]["delta"].as<double>();
    double cov_measurement_posi = node["covariance"]["measurement"]["posi"].as<double>();
    double cov_process_gyro = node["covariance"]["process"]["gyro"].as<double>();
    double cov_process_accel = node["covariance"]["process"]["accel"].as<double>();

    g_ = Eigen::Vector3d(0.0, 0.0, gravity);

    SetCovarianceP(cov_prior_posi, cov_prior_vel, cov_prior_ori,
                   cov_prior_epsilon, cov_prior_delta);
    SetCovarianceR(cov_measurement_posi);
    SetCovarianceQ(cov_process_gyro, cov_process_accel);

    X_.setZero();
    F_.setZero();
    C_.setIdentity();
    G_.block<3,3>(INDEX_MEASUREMENT_POSI,INDEX_MEASUREMENT_POSI) = Eigen::Matrix3d::Identity();

    F_.block<3,3>(INDEX_STATE_POSI, INDEX_STATE_VEL) = Eigen::Matrix3d::Identity();
    F_.block<3,3>(INDEX_STATE_ORI, INDEX_STATE_ORI) = BuildSkewMatrix(-w_);
}

void ESKF::SetCovarianceQ(double gyro_noise, double accel_noise) {
    Q_.setZero();
    Q_.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * gyro_noise * gyro_noise;
    Q_.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * accel_noise * accel_noise;
}

void ESKF::SetCovarianceR(double posi_noise) {
    R_.setZero();
    R_ = Eigen::Matrix3d::Identity() * posi_noise * posi_noise;
}

void ESKF::SetCovarianceP(double posi_noise, double velo_noise, double ori_noise,
                          double gyro_noise, double accel_noise) {
    P_.setZero();
    P_.block<3,3>(INDEX_STATE_POSI, INDEX_STATE_POSI) = Eigen::Matrix3d::Identity() * posi_noise;
    P_.block<3,3>(INDEX_STATE_VEL, INDEX_STATE_VEL) = Eigen::Matrix3d::Identity() * velo_noise;
    P_.block<3,3>(INDEX_STATE_ORI, INDEX_STATE_ORI) = Eigen::Matrix3d::Identity() * ori_noise;
    P_.block<3,3>(INDEX_STATE_GYRO_BIAS, INDEX_STATE_GYRO_BIAS) = Eigen::Matrix3d::Identity() * gyro_noise;
    P_.block<3,3>(INDEX_STATE_ACC_BIAS, INDEX_STATE_ACC_BIAS) = Eigen::Matrix3d::Identity() * accel_noise;
}

bool ESKF::Init(const GPSData &curr_gps_data, const IMUData &curr_imu_data) {
    init_velocity_ = curr_gps_data.true_velocity;
    velocity_ = init_velocity_;

    Eigen::Quaterniond Q = Eigen::AngleAxisd(90 * kDegree2Radian, Eigen::Vector3d::UnitZ()) *
                           Eigen::AngleAxisd(0 * kDegree2Radian, Eigen::Vector3d::UnitY()) *
                           Eigen::AngleAxisd(180 * kDegree2Radian, Eigen::Vector3d::UnitX());
    init_pose_.block<3,3>(0,0) = Q.toRotationMatrix();
    pose_ = init_pose_;

    imu_data_buff_.clear();
    imu_data_buff_.push_back(curr_imu_data);

    curr_gps_data_ = curr_gps_data;

    return true;
}

void ESKF::GetFGY(TypeMatrixF &F, TypeMatrixG &G, TypeVectorY &Y) {
    F = Ft_;
    G = G_;
    Y = Y_;
}

bool ESKF::Correct(const GPSData &curr_gps_data) {
    curr_gps_data_ = curr_gps_data;

    Y_ = pose_.block<3,1>(0,3) - curr_gps_data.position_ned;

    K_ = P_ * G_.transpose() * (G_ * P_ * G_.transpose() + C_ * R_ * C_.transpose()).inverse();

    P_ = (TypeMatrixP::Identity() - K_ * G_) * P_;
    X_ = X_ + K_ * (Y_ - G_ * X_);

    EliminateError();

    ResetState();

    return true;
}

bool ESKF::Predict(const IMUData &curr_imu_data) {
    imu_data_buff_.push_back(curr_imu_data);

    UpdateOdomEstimation();

    double delta_t = curr_imu_data.time - imu_data_buff_.front().time;

    Eigen::Vector3d curr_accel = pose_.block<3, 3>(0, 0)
                                 * curr_imu_data.linear_accel;

    UpdateErrorState(delta_t, curr_accel);

    imu_data_buff_.pop_front();
}

bool ESKF::UpdateErrorState(double t, const Eigen::Vector3d &accel) {
    Eigen::Matrix3d F_23 = BuildSkewMatrix(accel);

    F_.block<3,3>(INDEX_STATE_VEL, INDEX_STATE_ORI) = F_23;
    F_.block<3,3>(INDEX_STATE_VEL, INDEX_STATE_ACC_BIAS) = pose_.block<3,3>(0,0);
    F_.block<3,3>(INDEX_STATE_ORI, INDEX_STATE_GYRO_BIAS) = -pose_.block<3,3>(0,0);
    B_.block<3,3>(INDEX_STATE_VEL, 3) = pose_.block<3,3>(0,0);
    B_.block<3,3>(INDEX_STATE_ORI, 0) = -pose_.block<3,3>(0,0);

    TypeMatrixF Fk = TypeMatrixF::Identity() + F_ * t;
    TypeMatrixB Bk = B_ * t;

    Ft_ = F_ * t;

    X_ = Fk * X_;
    P_ = Fk * P_ * Fk.transpose() + Bk * Q_ * Bk.transpose();

    return true;
}

bool ESKF::UpdateOdomEstimation() {
    Eigen::Vector3d angular_delta;
    ComputeAngularDelta(angular_delta);

    Eigen::Matrix3d R_nm_nm_1;
    ComputeEarthTranform(R_nm_nm_1);

    Eigen::Matrix3d curr_R, last_R;
    ComputeOrientation(angular_delta, R_nm_nm_1, curr_R, last_R);

    Eigen::Vector3d curr_vel, last_vel;
    ComputeVelocity(curr_vel, last_vel, curr_R, last_R);

    ComputePosition(curr_vel, last_vel);

    return true;
}

bool ESKF::ComputeAngularDelta(Eigen::Vector3d &angular_delta) {
    IMUData curr_imu_data = imu_data_buff_.at(1);
    IMUData last_imu_data = imu_data_buff_.at(0);

    double delta_t = curr_imu_data.time - last_imu_data.time;

    if (delta_t <= 0){
        return false;
    }

    Eigen::Vector3d curr_angular_vel = curr_imu_data.angle_velocity;

    Eigen::Vector3d last_angular_vel = last_imu_data.angle_velocity;

    Eigen::Vector3d curr_unbias_angular_vel = curr_angular_vel;
    Eigen::Vector3d last_unbias_angular_vel = last_angular_vel;

    angular_delta = 0.5 * (curr_unbias_angular_vel + last_unbias_angular_vel) * delta_t;

    return true;
}

bool ESKF::ComputeEarthTranform(Eigen::Matrix3d &R_nm_nm_1) {
    IMUData curr_imu_data = imu_data_buff_.at(1);
    IMUData last_imu_data = imu_data_buff_.at(0);

    double delta_t = curr_imu_data.time - last_imu_data.time;

    constexpr double rm = 6353346.18315;
    constexpr double rn = 6384140.52699;
    Eigen::Vector3d w_en_n(-velocity_[1] / (rm + curr_gps_data_.position_lla[2]),
                           velocity_[0] / (rn + curr_gps_data_.position_lla[2]),
                           velocity_[0] / (rn + curr_gps_data_.position_lla[2])
                           * std::tan(curr_gps_data_.position_lla[0] * kDegree2Radian));

    Eigen::Vector3d w_in_n = w_en_n + w_;

    auto angular = delta_t * w_in_n;

    Eigen::AngleAxisd angle_axisd(angular.norm(), angular.normalized());

    R_nm_nm_1 = angle_axisd.toRotationMatrix().transpose();
}

bool ESKF::ComputeOrientation(const Eigen::Vector3d &angular_delta,
                              const Eigen::Matrix3d R_nm_nm_1,
                              Eigen::Matrix3d &curr_R,
                              Eigen::Matrix3d &last_R) {
    Eigen::AngleAxisd angle_axisd(angular_delta.norm(), angular_delta.normalized());
    last_R = pose_.block<3, 3>(0, 0);

    curr_R = R_nm_nm_1 * pose_.block<3, 3>(0, 0) * angle_axisd.toRotationMatrix();

    pose_.block<3, 3>(0, 0) = curr_R;

    return true;
}

bool ESKF::ComputeVelocity(Eigen::Vector3d &curr_vel, Eigen::Vector3d& last_vel,
                                             const Eigen::Matrix3d &curr_R,
                                             const Eigen::Matrix3d last_R) {
    IMUData curr_imu_data = imu_data_buff_.at(1);
    IMUData last_imu_data = imu_data_buff_.at(0);
    double delta_t = curr_imu_data.time - last_imu_data.time;
    if (delta_t <=0 ){
        return false;
    }

    Eigen::Vector3d curr_accel = curr_imu_data.linear_accel;
    Eigen::Vector3d curr_unbias_accel = GetUnbiasAccel(curr_R * curr_accel);

    Eigen::Vector3d last_accel = last_imu_data.linear_accel;
    Eigen::Vector3d last_unbias_accel = GetUnbiasAccel(last_R * last_accel);

    last_vel = velocity_;

    velocity_ += delta_t * 0.5 * (curr_unbias_accel + last_unbias_accel);
    curr_vel = velocity_;

    return true;
}

Eigen::Vector3d ESKF::GetUnbiasAccel(const Eigen::Vector3d &accel) {
//    return accel - accel_bias_ + g_;
    return accel + g_;
}

bool ESKF::ComputePosition(const Eigen::Vector3d& curr_vel, const Eigen::Vector3d& last_vel){
    double delta_t = imu_data_buff_.at(1).time - imu_data_buff_.at(0).time;

    pose_.block<3,1>(0,3) += 0.5 * delta_t * (curr_vel + last_vel);

    return true;
}

void ESKF::ResetState() {
    X_.setZero();
}

void ESKF::EliminateError() {
    pose_.block<3,1>(0,3) = pose_.block<3,1>(0,3) - X_.block<3,1>(INDEX_STATE_POSI, 0);

    velocity_ = velocity_ - X_.block<3,1>(INDEX_STATE_VEL, 0);
    Eigen::Matrix3d C_nn = Sophus::SO3d::exp(X_.block<3,1>(INDEX_STATE_ORI, 0)).matrix();
    pose_.block<3,3>(0,0) = C_nn * pose_.block<3,3>(0,0);

    gyro_bias_ = gyro_bias_ - X_.block<3,1>(INDEX_STATE_GYRO_BIAS, 0);
    accel_bias_ = accel_bias_ - X_.block<3,1>(INDEX_STATE_ACC_BIAS, 0);
}

Eigen::Matrix4d ESKF::GetPose() const {
    return pose_;
}