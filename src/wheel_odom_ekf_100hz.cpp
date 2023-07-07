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

static double fitness_score_median = 0.0;

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
static int TransNum = 200;                // note-justin 10 second for period
static int ENCODER_TransNum = 500000000;  // note-justin 5 second for period

static int fitness_cnt = 0;
static float fitness_score = 0.0;
static double fitness_score_sum = 0.0;
static double fitness_score_mean = 0.0;

static float fitness_above_threshold = 0.0;  // threshold above 0.5 indoor 3.0 outdoor
static int fitness_score_cnt = 0;

static bool start_flag = false;
static bool emergency_button_pressed_flag = false;

static int waypoint_change_flag = 0;
static bool lift_change_flag = true;
static bool use_encoder_initialized = true;
static bool use_wheel_odom_lift_flag = false;
static bool initialized_flag = false;

#define fitness_above_threshold_indoor 0.5
#define fitness_above_threshold_outdoor 20.0

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

ros::Publisher odom_pub;
ros::Publisher odom_pose_pub;
ros::Publisher ndt_pose_to_odom_pub;
ros::Publisher encoder_twist_pub;
ros::Publisher fitness_score_median_pub;
ros::Publisher fitness_score_mean_pub;
ros::Publisher post_distance_pub;
ros::Publisher odom_yaw_pub;
ros::Publisher odom_twist_angular_pub;
ros::Publisher odom_twist_linear_pub;

ros::Subscriber sub_imu;
ros::Subscriber sub;
ros::Subscriber sub_initalpose;
ros::Subscriber sub_pose1;
ros::Subscriber sub_liftpose;
ros::Subscriber sub_start;
ros::Subscriber sub_reset;
ros::Subscriber emergency_switch_reset;
ros::Subscriber sub_ndt_stat;
ros::Subscriber pose_param_sub;
ros::Subscriber lift_flag_sub;
ros::Subscriber use_wheel_odom_su;
ros::Subscriber imu_odom_sub;

ros::Time last_imu_stamp;
ros::Time last_motorinfo_stamp;
ros::Time last_imu_twist_stamp;

class MovingAverage
{
public:
  /** Initialize your data structure here. */
  MovingAverage(int sz) : size(sz)
  {
  }

  float next(float val)
  {
    if (q.size() < size)
    {
      q.push_back(val);
      sum += val;
      return val;
    }
    else
    {
      sum -= q.front();
      sum += val;
      q.pop_front();
      q.push_back(val);
      return sum / static_cast<float>(q.size());
    }
  }

private:
  int size = 0;
  int sum = 0;
  std::deque<float> q;
};

MovingAverage avg(6);

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

  ROS_INFO("wheel_odom_ekf_node::init_state x = %f y = %f z = %f v_x = %f v_y = %f yaw = %f", state[0], state[1],
           state[2], state[3], pitch, state[4]);
}

void use_wheel_odom_callback(const std_msgs::BoolConstPtr& msg)
{
  use_wheel_odom_lift_flag = msg->data;
  TransNum = 500;
  ROS_INFO("use_wheel_odom_lift_flag = %d ", use_wheel_odom_lift_flag);
}

void ResetCallback(const std_msgs::Bool::ConstPtr& msg)
{
  if (msg->data)
  {
    encoder_iteration_num = TransNum + 10;
  }
}

void waypointflagCallback(const std_msgs::Int32::ConstPtr& msg)
{
  waypoint_change_flag = msg->data;

  if (waypoint_change_flag == 4)
  {
    fitness_above_threshold = fitness_above_threshold_indoor;
  }
  else
  {
    fitness_above_threshold = fitness_above_threshold_outdoor;
  }

  if ((waypoint_change_flag == 5 || use_wheel_odom_lift_flag) && use_encoder_initialized)
  {
    PERIOD = ENCODER_TransNum;
    if (fitness_score < 3.0)
    {
      wheel_odom_ekf::init_state(ekf_pose_x, ekf_pose_y, ekf_pose_z, ekf_pose_roll, ekf_pose_pitch, ekf_pose_yaw);
    }
    use_encoder_initialized = false;
    lift_change_flag = true;
    encoder_iteration_num = 0;
    use_wheel_odom_lift_flag = false;

    std_msgs::Int32 msg;
    msg.data = 1;
    ndt_pose_to_odom_pub.publish(msg);
  }
  else if (waypoint_change_flag != 5 && !use_encoder_initialized)
  {
    PERIOD = TransNum;
    use_encoder_initialized = true;
    lift_change_flag = true;
    encoder_iteration_num = 0;

    std_msgs::Int32 msg;
    msg.data = 0;
    ndt_pose_to_odom_pub.publish(msg);
  }
}

void pub_odom()
{
  current_pose_x = state[0];
  current_pose_y = state[1];
  current_linear_x = state[2];
  current_linear_y = state[3];
  current_pose_yaw = state[4];

  // 发布坐标变换 T_map_to_odom
  static tf::TransformBroadcaster broadcaster;
  geometry_msgs::TransformStamped odom_trans;
  odom_trans.header.frame_id = "camera_init";
  odom_trans.child_frame_id = "odom_base_link";
  odom_trans.header.stamp = last_imu_stamp;
  odom_trans.transform.translation.x = current_pose_x;
  odom_trans.transform.translation.y = current_pose_y;
  odom_trans.transform.translation.z = current_pose_z;
  odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(current_pose_yaw);
  broadcaster.sendTransform(odom_trans);

  // 发布里程计消息 T_map_to_odom
  nav_msgs::Odometry odom;
  odom.header.stamp = last_imu_stamp;
  odom.header.frame_id = "camera_init";
  odom.child_frame_id = "odom_base_link";
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

  // 发布车辆线速度和角速度
  geometry_msgs::PoseStamped odom_pose_msg;
  odom_pose_msg.header.frame_id = "map";
  odom_pose_msg.header.stamp = last_imu_stamp;
  odom_pose_msg.pose.position.x = odom.pose.pose.position.x;
  odom_pose_msg.pose.position.y = odom.pose.pose.position.y;
  odom_pose_msg.pose.position.z = odom.pose.pose.position.z;
  odom_pose_msg.pose.orientation = odom.pose.pose.orientation;
  odom_pose_pub.publish(odom_pose_msg);

  std_msgs::Float32 fitness_score_median_msg;
  fitness_score_median_msg.data = fitness_score_median;
  fitness_score_median_pub.publish(fitness_score_median_msg);

  std_msgs::Float32 store_pose_distance_msg;
  store_pose_distance_msg.data = offset_yaw;
  post_distance_pub.publish(store_pose_distance_msg);

  std_msgs::Float32 fitness_score_mean_msg;
  fitness_score_mean_msg.data = fitness_score_mean;
  fitness_score_mean_pub.publish(fitness_score_mean_msg);

  std_msgs::Float32 odom_yaw_pub_msg;
  odom_yaw_pub_msg.data = current_pose_yaw;
  odom_yaw_pub.publish(odom_yaw_pub_msg);

  std_msgs::Float32 odom_twist_angular_pub_msg;
  odom_twist_angular_pub_msg.data = odom.twist.twist.angular.z;
  odom_twist_angular_pub.publish(odom_twist_angular_pub_msg);

  std_msgs::Float32 odom_twist_linear_pub_msg;
  odom_twist_linear_pub_msg.data = odom.twist.twist.linear.x;
  odom_twist_linear_pub.publish(odom_twist_linear_pub_msg);
}

void NdtStatCallback(const autoware_msgs::ndt_stat& msg)
{
  fitness_score = msg.score;

  fitness_score_cnt++;
  if (fitness_score_cnt >= INT32_MAX)
    fitness_score_cnt = 0;

  if (fitness_score > 10.0)
  {
    fitness_score_sum = fitness_score_sum + fitness_score_mean;
  }
  else
  {
    fitness_score_sum = fitness_score_sum + fitness_score;
  }
  fitness_score_mean = fitness_score_sum / float(fitness_score_cnt);
}

void StartWheelOdomCallback(const std_msgs::Int32ConstPtr& msg)
{
  start_flag = true;
}

void initialpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& input)
{
  tf::TransformListener listener;
  tf::StampedTransform transform;
  try
  {
    ros::Time now = ros::Time(0);
    listener.waitForTransform("map", input->header.frame_id, now, ros::Duration(10.0));
    listener.lookupTransform("map", input->header.frame_id, now, transform);
  }
  catch (tf::TransformException& ex)
  {
    // ROS_ERROR("%s", ex.what());
  }

  tf::Quaternion q(input->pose.pose.orientation.x, input->pose.pose.orientation.y, input->pose.pose.orientation.z,
                   input->pose.pose.orientation.w);
  tf::Matrix3x3 m(q);

  // Justin 从rviz手动给出的pose是基于world坐标系的，所以需要将其转换到map坐标系
  double x = input->pose.pose.position.x + transform.getOrigin().x();
  double y = input->pose.pose.position.y + transform.getOrigin().y();
  double z = input->pose.pose.position.z + transform.getOrigin().z();

  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  wheel_odom_ekf::init_state(x, y, z, roll, pitch, yaw);
}

void liftpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& pose)
{
  if (lift_change_flag)
  {
    PERIOD = ENCODER_TransNum;
    encoder_iteration_num = 0;
    lift_change_flag = false;

    tf::Quaternion RQ2;
    double roll, pitch, yaw;
    tf::quaternionMsgToTF(pose->pose.pose.orientation, RQ2);
    tf::Matrix3x3(RQ2).getRPY(roll, pitch, yaw);
    double x = pose->pose.pose.position.x;
    double y = pose->pose.pose.position.y;
    double z = pose->pose.pose.position.z;
    wheel_odom_ekf::init_state(x, y, z, roll, pitch, yaw);
  }
}

void emergency_switch_Callback(const std_msgs::Bool::ConstPtr& msg)
{
  emergency_button_pressed_flag = msg->data;
}

void EKFPoseCallback(const geometry_msgs::PoseStampedPtr& pose)
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

  fitness_score_median = avg.next(fitness_score);

  if ((fitness_score_median < (fitness_score_mean + fitness_above_threshold)) && (pose_distance < 10.0) &&
      encoder_iteration_num >= PERIOD)
  {
    wheel_odom_ekf::init_state(ekf_pose_x, ekf_pose_y, ekf_pose_z, ekf_pose_roll, ekf_pose_pitch, ekf_pose_yaw);
    encoder_iteration_num = 0;
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

  current_angular_z = (state[4] - last_encoder_state[4])/dt;
  last_encoder_state = state;
  last_encoder_w = w;

  pub_odom();
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
}

void imu_twist_callback(const geometry_msgs::TwistStamped::ConstPtr& msg_ptr)
{
  if (!initialized_flag)
  {
    return;
  }
  // 急停模式下通过imu线速度和角速度代替编码器线速度和角速度
  if (emergency_button_pressed_flag && waypoint_change_flag == 5)
  {
    ros::Time cur = msg_ptr->header.stamp;
    velhicle_linear_x = msg_ptr->twist.linear.x;
    velhicle_angular_z = msg_ptr->twist.angular.z;

    if (first_imu_twist_msg)
    {
      last_imu_twist_stamp = cur;
      first_imu_twist_msg = false;
      return;
    }

    double dt = (last_imu_stamp - last_motorinfo_stamp).toSec();

    last_motorinfo_stamp = cur;
  }
  else
  {
    first_imu_twist_msg = true;
  }
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
  nh.param<std::string>("file_path", yaml_file,
                        "/home/justin/ZROS/ros/src/localization/packages/wheel_odom_kalman/config/encoder.yaml");
  nh.param<std::string>("imu_topic", imu_topic, "/imu_calibrated");
  YAML::Node config = YAML::LoadFile(yaml_file);

  std::string robot_id;
  nh.param<std::string>("/robot_id", robot_id,
                        "ZR1001");  // note-justin   如果无法获得main_socket/campus_id,
                                    // 就默认读取map11150

  // // Get wheel parameters for specific robot from YAML file
  wheel_odom_ekf::wheel_radius = config[robot_id]["wheel_radius"].as<double>();
  wheel_odom_ekf::encoder_ticks_per_rev = config[robot_id]["encoder_ticks_per_rev"].as<int>();
  wheel_odom_ekf::wheelbase = config[robot_id]["wheelbase"].as<double>();
  wheel_odom_ekf::wheelratio = config[robot_id]["wheelratio"].as<double>();

  ROS_INFO(
      "wheel_odom_ekf_node::main robot_id = %s wheel_radius = %f "
      "encoder_ticks_per_rev = %d "
      "wheelbase = %f wheelratio = %f",
      robot_id.c_str(), wheel_odom_ekf::wheel_radius, wheel_odom_ekf::encoder_ticks_per_rev, wheel_odom_ekf::wheelbase,
      wheel_odom_ekf::wheelratio);

  wheel_odom_ekf::odom_pub = nh.advertise<nav_msgs::Odometry>("/wheel_odom", 1000);
  wheel_odom_ekf::odom_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/wheel_odom_pose", 100);

  wheel_odom_ekf::ndt_pose_to_odom_pub = nh.advertise<std_msgs::Int32>("/ndt_pose_to_odom", 10);
  wheel_odom_ekf::encoder_twist_pub = nh.advertise<geometry_msgs::TwistStamped>("/encoder_twist", 10);
  wheel_odom_ekf::fitness_score_median_pub = nh.advertise<std_msgs::Float32>("/fitness_score_median", 10);
  wheel_odom_ekf::fitness_score_mean_pub = nh.advertise<std_msgs::Float32>("/fitness_score_mean", 10);
  wheel_odom_ekf::post_distance_pub = nh.advertise<std_msgs::Float32>("/store_pose_distance", 10);

  wheel_odom_ekf::odom_yaw_pub = nh.advertise<std_msgs::Float32>("/odom_yaw", 10);
  wheel_odom_ekf::odom_twist_angular_pub = nh.advertise<std_msgs::Float32>("/twist_angular", 10);
  wheel_odom_ekf::odom_twist_linear_pub = nh.advertise<std_msgs::Float32>("/twist_linear", 10);

  ros::Subscriber sub_imu = nh.subscribe(imu_topic, 10, wheel_odom_ekf::imu_callback);
  ros::Subscriber sub = nh.subscribe("/motor_info", 10, wheel_odom_ekf::encoders_callback);

  ros::Subscriber sub_initalpose = nh.subscribe("/initialpose", 10, wheel_odom_ekf::initialpose_callback);
  ros::Subscriber sub_pose1 = nh.subscribe("/ekf_pose", 10, wheel_odom_ekf::EKFPoseCallback);
  ros::Subscriber sub_liftpose = nh.subscribe("/liftpose", 100, wheel_odom_ekf::liftpose_callback);

  ros::Subscriber sub_start = nh.subscribe("/start_flag", 10, wheel_odom_ekf::StartWheelOdomCallback);
  ros::Subscriber sub_reset = nh.subscribe("/reset_wheel_odom", 10, wheel_odom_ekf::ResetCallback);
  ros::Subscriber emergency_switch_reset =
      nh.subscribe("/emergency_switch", 10, wheel_odom_ekf::emergency_switch_Callback);
  ros::Subscriber sub_ndt_stat = nh.subscribe("/ndt_stat", 10, wheel_odom_ekf::NdtStatCallback);
  ros::Subscriber lift_flag_sub =
      nh.subscribe("/global_waypoint_change_flag", 10, wheel_odom_ekf::waypointflagCallback);
  ros::Subscriber use_wheel_odom_sub = nh.subscribe("/use_wheel_odom", 10, wheel_odom_ekf::use_wheel_odom_callback);
  ros::Subscriber imu_odom_sub = nh.subscribe("/imu_twist", 10, wheel_odom_ekf::imu_twist_callback);

  wheel_odom_ekf::R_b_w = Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::MatrixXd E = Eigen::Matrix<double, 5, 5>::Identity();
  wheel_odom_ekf::Q = E;
  wheel_odom_ekf::H = E;
  wheel_odom_ekf::P = E;
  wheel_odom_ekf::R = E;
  ros::spin();
  return 0;
}