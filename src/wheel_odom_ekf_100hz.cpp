/* Copyright (c) 2020 ZhenRoot, Inc. All Rights Reserved
 *

 * last modified 2022/08/06 *
 * 混合型里程计，具备自我恢复功能，可以通过简单的节点间握手改变里程计的运行状态，
 *
 * 实现多种功能。
 *
 */
#include <autoware_msgs/ConfigNdt.h>
#include <autoware_msgs/ndt_stat.h>
#include <cstdlib>
#include <deque>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt8MultiArray.h>
#include <string>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <zr_msgs/motor_info.h>

namespace wheel_odom_ekf
{
static double current_pose_x = 0.0;
static double current_pose_y = 0.0;
static double current_pose_z = 0.0;
static double current_pose_roll = 0.0;
static double current_pose_pitch = 0.0;
static double current_pose_yaw = 0.0;

static double fitness_score_threshold = 0.6;
static double fitness_score = 0.0;

static double ekf_pose_x = 0.0;
static double ekf_pose_y = 0.0;
static double ekf_pose_z = 0.0;
static double ekf_pose_roll = 0.0;
static double ekf_pose_pitch = 0.0;
static double ekf_pose_yaw = 0.0;

static double previous_pose_x = 0.0;
static double previous_pose_y = 0.0;
static double previous_pose_z = 0.0;
static double previous_pose_roll = 0.0;
static double previous_pose_pitch = 0.0;
static double previous_pose_yaw = 0.0;

static double offset_x = 0.0;
static double offset_y = 0.0;
static double offset_z = 0.0;
static double offset_roll = 0.0;
static double offset_pitch = 0.0;
static double offset_yaw = 0.0;

static int encoder_iteration_num = 0;
static int PERIOD = 200;

static bool emergency_button_pressed_flag = false;
static bool initialized_flag = false;

static double wheel_radius = 0.1;
static int encoder_ticks_per_rev = 1000;
static double wheelbase = 0.5;
static double wheelratio = 0.95;

static double current_linear_x = 0.0;
static double current_linear_y = 0.0;
static double current_angular_z = 0.0;
static double velhicle_linear_x = 0.0;
static double velhicle_angular_z = 0.0;

static bool first_imu_msg = true;
static bool first_enc_msg = true;
static bool first_imu_twist_msg = true;

static double data_num = 0;
static double last_encoder_w = 0.0;

Eigen::Vector3d gravity(0.0, 0.0, 9.81);
Eigen::Vector3d sum_g;
Eigen::Vector3d sum_w;
Eigen::Vector3d b_a;
Eigen::Vector3d b_w;
Eigen::Vector3d last_w_w;
Eigen::Vector3d last_a_w;

Eigen::Matrix<double, 5, 1> state;
Eigen::Matrix<double, 5, 1> last_encoder_state;
Eigen::Matrix<double, 3, 3> R_b_w;
Eigen::Matrix<double, 5, 5> Q;
Eigen::Matrix<double, 5, 5> P;
Eigen::Matrix<double, 5, 5> R;
Eigen::Matrix<double, 5, 5> F;
Eigen::Matrix<double, 5, 5> H;
Eigen::MatrixXd K;

ros::Publisher odom_pub, encoder_twist_pub;

ros::Time last_imu_stamp;
ros::Time last_motorinfo_stamp;

void init_state(double& x, double& y, double& z, double& roll, double& pitch, double& yaw, double v_x = 0.0,
                double v_y = 0.0)
{
  current_pose_x = x;
  current_pose_y = y;
  current_pose_z = z;

  current_pose_roll = roll;
  current_pose_pitch = pitch;
  current_pose_yaw = yaw;

  state[0] = x;
  state[1] = y;
  state[2] = v_x;
  state[3] = v_y;
  state[4] = yaw;

  last_encoder_state = state;

  first_enc_msg = true;
  first_imu_msg = true;
  first_imu_twist_msg = true;
  initialized_flag = true;

  encoder_iteration_num = 0;
  ROS_INFO("wheel_odom_ekf_node::init_state x = %f y = %f z = %f v_x = %f v_y = %f yaw = %f", state[0], state[1],
           state[2], state[3], pitch, state[4]);
}

void ResetCallback(const std_msgs::Bool::ConstPtr& msg)
{
  if (msg->data)
  {
    double x, y, z, roll, pitch, yaw = 0.0;
    wheel_odom_ekf::init_state(x, y, z, roll, pitch, yaw);
  }
}

void pub_odom()
{
  current_pose_x = state[0];
  current_pose_y = state[1];
  current_linear_x = state[2];
  current_linear_y = state[3];
  current_pose_yaw = state[4];

  // 发布坐标变换
  static tf::TransformBroadcaster broadcaster;
  geometry_msgs::TransformStamped odom_trans;
  odom_trans.header.frame_id = "wheel_odom";
  odom_trans.child_frame_id = "base_link_in_wheel_odom";
  odom_trans.header.stamp = last_imu_stamp;
  odom_trans.transform.translation.x = current_pose_x;
  odom_trans.transform.translation.y = current_pose_y;
  odom_trans.transform.translation.z = current_pose_z;
  odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(current_pose_yaw);
  broadcaster.sendTransform(odom_trans);

  // 发布里程计消息
  nav_msgs::Odometry odom;
  odom.header.stamp = last_imu_stamp;
  odom.header.frame_id = "wheel_odom";
  odom.child_frame_id = "base_link_in_wheel_odom";
  odom.pose.pose.position.x = current_pose_x;
  odom.pose.pose.position.y = current_pose_y;
  odom.pose.pose.position.z = current_pose_z;
  geometry_msgs::Quaternion odom_quat;
  odom_quat = tf::createQuaternionMsgFromRollPitchYaw(current_pose_roll, current_pose_pitch, current_pose_yaw);
  odom.pose.pose.orientation = odom_quat;
  odom.twist.twist.linear.x = current_linear_x;
  odom.twist.twist.linear.y = current_linear_y;
  odom.twist.twist.linear.z = 0.0;
  odom.twist.twist.angular.x = 0.0;
  odom.twist.twist.angular.y = 0.0;
  odom.twist.twist.angular.z = current_angular_z;
  odom_pub.publish(odom);

  // 发布车辆线速度和角速度
  geometry_msgs::TwistStamped encoder_twist;
  encoder_twist.header.stamp = last_imu_stamp;
  encoder_twist.header.frame_id = "base_link";
  encoder_twist.twist.linear.x = velhicle_linear_x;
  encoder_twist.twist.linear.y = 0.0;
  encoder_twist.twist.linear.z = 0.0;
  encoder_twist.twist.angular.x = 0.0;
  encoder_twist.twist.angular.y = 0.0;
  encoder_twist.twist.angular.z = velhicle_angular_z;
  encoder_twist_pub.publish(encoder_twist);
}

void FitnessCallback(const std_msgs::Float32::ConstPtr& msg)
{
  fitness_score = msg->data;
}

void emergency_switch_Callback(const std_msgs::Bool::ConstPtr& msg)
{
  emergency_button_pressed_flag = msg->data;
}

void CurrentPoseCallback(const geometry_msgs::PoseStampedPtr& pose)
{
  ekf_pose_x = pose->pose.position.x;
  ekf_pose_y = pose->pose.position.y;
  ekf_pose_z = pose->pose.position.z;
  tf::Quaternion RQ2;
  tf::quaternionMsgToTF(pose->pose.orientation, RQ2);
  tf::Matrix3x3(RQ2).getRPY(ekf_pose_roll, ekf_pose_pitch, ekf_pose_yaw);

  if (!initialized_flag)
  {
    wheel_odom_ekf::init_state(ekf_pose_x, ekf_pose_y, ekf_pose_z, ekf_pose_roll, ekf_pose_pitch, ekf_pose_yaw);
    initialized_flag = true;
  }

  offset_x = ekf_pose_x - previous_pose_x;
  offset_y = ekf_pose_y - previous_pose_y;
  offset_z = ekf_pose_z - previous_pose_z;
  offset_yaw = ekf_pose_yaw - previous_pose_yaw;

  double pose_distance = sqrt(offset_x * offset_x + offset_y * offset_y);

  if ((fitness_score < (fitness_score_threshold)) && (pose_distance < 10.0) && encoder_iteration_num >= PERIOD)
  {
    wheel_odom_ekf::init_state(ekf_pose_x, ekf_pose_y, ekf_pose_z, ekf_pose_roll, ekf_pose_pitch, ekf_pose_yaw);
  }

  previous_pose_x = ekf_pose_x;
  previous_pose_y = ekf_pose_y;
  previous_pose_z = ekf_pose_z;
  previous_pose_yaw = ekf_pose_yaw;
}

void ekf_predict(const Eigen::Vector3d& a_body, const Eigen::Vector3d& w_body, const double& dt)
{
  Eigen::Vector3d w_w = w_body;
  Eigen::Vector3d mid_w = (w_w + last_w_w) / 2;
  double delta_yaw = mid_w[2] * dt;

  Eigen::Vector3d a_w = Eigen::Vector3d::Zero();
  a_w[0] = a_body[0] * cos(state[4] + delta_yaw);
  a_w[1] = a_body[0] * sin(state[4] + delta_yaw);

  Eigen::Vector3d mid_a = (a_w + last_a_w) / 2;

  F << 1.0, 0.0, dt, 0.0, -pow(dt, 2) * mid_a[1] / 2, 0.0, 1.0, 0.0, dt, pow(dt, 2) * mid_a[0] / 2, 0.0, 0.0, 1.0, 0.0,
      -dt * mid_a[1], 0.0, 0.0, 0.0, 1.0, dt * mid_a[0], 0.0, 0.0, 0.0, 0.0, 1.0;

  P = F * P * F.transpose() + Q;

  state[0] += state[2] * dt + pow(dt, 2) * mid_a[0] / 2.0;
  state[1] += state[3] * dt + pow(dt, 2) * mid_a[1] / 2.0;
  state[2] += mid_a[0] * dt;
  state[3] += mid_a[1] * dt;
  state[4] += delta_yaw;

  last_a_w = a_w;
  last_w_w = w_w;

  if (encoder_iteration_num < PERIOD)
  {
    encoder_iteration_num++;
  }
}

void ekf_meaturement(const double& v, const double& w, const double& dt)
{
  double delta_yaw = (w + last_encoder_w) / 2 * dt;

  double v_x = v * cos(delta_yaw + last_encoder_state[4]);
  double v_y = v * sin(delta_yaw + last_encoder_state[4]);

  double p_x = v_x * dt + last_encoder_state[0];
  double p_y = v_y * dt + last_encoder_state[1];
  Eigen::Matrix<double, 5, 1> z_m;
  z_m << p_x, p_y, v_x, v_y, delta_yaw + last_encoder_state[4];

  Eigen::MatrixXd _R = R * abs(v) * 5;
  R(4, 4) = R(4, 4) * abs(w) * 100;

  K = P * H * (H.transpose() * (P * H) + R).inverse();

  Eigen::MatrixXd dz = z_m - state;

  state = state + K * dz;
  Eigen::MatrixXd E = Eigen::Matrix<double, 5, 5>::Identity();
  P = (E - K * H.transpose()) * P;

  current_angular_z = (state[4] - last_encoder_state[4]) / dt;
  last_encoder_state = state;
  last_encoder_w = w;
}

void encoders_callback(const zr_msgs::motor_info::ConstPtr& msg)
{
  if (!initialized_flag || first_imu_msg)
  {
    return;
  }
  if (!emergency_button_pressed_flag)
  {
    ros::Time cur = msg->header.stamp;
    velhicle_linear_x = (msg->left_vel + msg->right_vel) / 2.0;
    velhicle_angular_z = (msg->right_vel - msg->left_vel) / wheelbase * wheelratio;

    if (first_enc_msg)
    {
      last_motorinfo_stamp = cur;
      first_enc_msg = false;
      return;
    }

    double dt = (last_imu_stamp - last_motorinfo_stamp).toSec();
    last_motorinfo_stamp = cur;

    ekf_meaturement(velhicle_linear_x, velhicle_angular_z, dt);
  }
  pub_odom();
}

void get_rotation(Eigen::Vector3d origin_vector, Eigen::Vector3d location_vector, Eigen::Matrix3d& R_b_w)
{
  origin_vector = origin_vector / origin_vector.norm();
  location_vector = location_vector / location_vector.norm();
  double c = origin_vector.transpose() * location_vector;
  Eigen::Vector3d n_vector = origin_vector.cross(location_vector);

  Eigen::Matrix3d n_vector_invert;
  n_vector_invert << 0.0, -n_vector[2], n_vector[1], n_vector[2], 0.0, -n_vector[0], -n_vector[1], n_vector[0], 0.0;

  R_b_w = Eigen::Matrix3d::Identity() + n_vector_invert + n_vector_invert * n_vector_invert.transpose() / (1 + c);
}

void imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg)
{
  Eigen::Vector3d a_m(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
  Eigen::Vector3d w_m(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);

  if (abs(a_m.norm() - gravity[2]) < 0.02)
  {
    sum_g = a_m + sum_g;
    sum_w = w_m + sum_w;
    data_num += 1;
  }

  if (data_num == 100)
  {
    Eigen::Vector3d g = sum_g / data_num;
    b_w = sum_w / data_num;
    get_rotation(gravity, g, R_b_w);
    b_a = R_b_w.transpose() * g - gravity;
    data_num = 0;
  }

  if (!initialized_flag)
  {
    return;
  }

  ros::Time cur = imu_msg->header.stamp;
  if (first_imu_msg)
  {
    first_imu_msg = false;
    last_imu_stamp = cur;
    return;
  }

  Eigen::Vector3d a_body = R_b_w * a_m - b_a;
  Eigen::Vector3d w_body = R_b_w * w_m - b_w;

  a_body[2] = 0.0;
  a_body[0] = a_body[2] > 0.0 ? a_body.norm() : -a_body.norm();
  a_body[1] = 0.0;

  double w = imu_msg->angular_velocity.z;
  double dt = (cur - last_imu_stamp).toSec();
  last_imu_stamp = cur;
  ekf_predict(a_body, w_body, dt);
}

}  // namespace wheel_odom_ekf
int main(int argc, char** argv)
{
  ros::init(argc, argv, "wheel_odom_ekf");
  ros::NodeHandle nh("~");

  // Load parameters from YAML file
  std::string yaml_file;
  std::string imu_topic;
  nh.param<std::string>("file_path", yaml_file, "encoder.yaml");
  nh.param<std::string>("imu_topic", imu_topic, "/imu_data");
  YAML::Node config = YAML::LoadFile(yaml_file);

  std::string robot_id;
  nh.param<std::string>("/robot_id", robot_id, "ZR1001");

  // // Get wheel parameters for specific robot from YAML file
  wheel_odom_ekf::wheel_radius = config["wheel_radius"].as<double>();
  wheel_odom_ekf::encoder_ticks_per_rev = config["encoder_ticks_per_rev"].as<int>();
  wheel_odom_ekf::wheelbase = config["wheelbase"].as<double>();
  wheel_odom_ekf::wheelratio = config["wheelratio"].as<double>();

  ROS_INFO(
      "wheel_odom_ekf_node::main robot_id = %s wheel_radius = %f "
      "encoder_ticks_per_rev = %d "
      "wheelbase = %f wheelratio = %f",
      robot_id.c_str(), wheel_odom_ekf::wheel_radius, wheel_odom_ekf::encoder_ticks_per_rev, wheel_odom_ekf::wheelbase,
      wheel_odom_ekf::wheelratio);

  wheel_odom_ekf::odom_pub = nh.advertise<nav_msgs::Odometry>("/wheel_odom", 1000);
  wheel_odom_ekf::encoder_twist_pub = nh.advertise<geometry_msgs::TwistStamped>("/encoder_twist", 10);

  ros::Subscriber sub_imu = nh.subscribe(imu_topic, 10, wheel_odom_ekf::imu_callback);
  ros::Subscriber sub = nh.subscribe("/motor_info", 10, wheel_odom_ekf::encoders_callback);

  ros::Subscriber sub_pose1 = nh.subscribe("/current_pose", 10, wheel_odom_ekf::CurrentPoseCallback);
  ros::Subscriber sub_reset = nh.subscribe("/reset_wheel_odom", 10, wheel_odom_ekf::ResetCallback);
  ros::Subscriber sub_ndt_stat = nh.subscribe("/fitness_score", 10, wheel_odom_ekf::FitnessCallback);

  wheel_odom_ekf::R_b_w = Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::MatrixXd E = Eigen::Matrix<double, 5, 5>::Identity();
  wheel_odom_ekf::Q = E;
  wheel_odom_ekf::H = E;
  wheel_odom_ekf::P = E;
  wheel_odom_ekf::R = E;

  double x, y, z, roll, pitch, yaw = 0.0;
  wheel_odom_ekf::init_state(x, y, z, roll, pitch, yaw);
  ros::spin();
  return 0;
}