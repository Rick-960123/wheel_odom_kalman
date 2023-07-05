/* Copyright (c) 2020 ZhenRoot, Inc. All Rights Reserved
 *

 * last modified 2022/08/06 *
 * 混合型里程计，具备自我恢复功能，可以通过简单的节点间握手改变里程计的运行状态，
 *
 * 实现多种功能。
 *
 */
#include "eskf.h"
#include <autoware_msgs/ConfigNdt.h>
#include <autoware_msgs/ndt_stat.h>
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

namespace wheel_odom_eskf
{
static bool initialized_flag = false;

static double current_pose_x = 0.0;
static double current_pose_y = 0.0;
static double current_pose_z = 0.0;
static double current_pose_roll = 0.0;
static double current_pose_pitch = 0.0;
static double current_pose_yaw = 0.0;

static double fitness_score_median = 0.0;

static double ndt_pose_x = 0.0;
static double ndt_pose_y = 0.0;
static double ndt_pose_z = 0.0;

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

#define PI (3.1415926)
double clock_x;
double clock_y;
double clock_th;

static int encoder_iteration_num = 0;
static int TransNum = 200;                // note-justin 10 second for period
static int ENCODER_TransNum = 500000000;  // note-justin 5 second for period
static int PERIOD = 2000;
static int FitnessLoop = 0;  // the loop for keep on encoder
static int fitness_cnt = 0;
static int fitness_cnt_divide;
static float fitness_score = 0.0;
static double fitness_score_sum = 0.0;
static double fitness_score_mean = 0.0;

static bool emergency_button_pressed_flag = false;

static float fitness_above_threshold = 0.0;  // threshold above 0.5 indoor 3.0 outdoor

static int fitness_score_cnt = 0;

bool start_flag = false;
static double ndt_pose_roll = 0.0;
static double ndt_pose_pitch = 0.0;
static double ndt_pose_yaw = 0.0;
static int waypoint_change_flag = 0;
static bool lift_change_flag = true;
static bool use_encoder_initialized = true;
static bool use_wheel_odom_lift_flag = false;

#define fitness_above_threshold_indoor 0.5
#define fitness_above_threshold_outdoor 20.0

static double wheel_radius = 0.1;
static int encoder_ticks_per_rev = 1000;
static double wheelbase = 0.5;
static double wheelratio = 0.95;

static double odom_linear_x = 0.0;
static double odom_angular_z = 0.0;

static double last_v;
static double last_w;

ros::Time last_imu_stamp;
ros::Time last_motorinfo_stamp;
ros::Time last_imu_twist_stamp;
bool first_imu_msg = true;
bool first_enc_msg = true;
bool first_imu_twist_msg = true;

Eigen::Vector3d state;
Eigen::Vector3d last_state;

Eigen::Matrix<double, 3, 3> Q;
Eigen::Matrix<double, 3, 3> P;
Eigen::Matrix<double, 1, 1> R;
Eigen::Matrix<double, 3, 3> F;
Eigen::Vector3d H;
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
}

void pub_odom()
{
  current_pose_x = eskf::state[0];
  current_pose_y = eskf::state[1];
  current_pose_z = eskf::state[2];
  tf::Quaternion odom_quat(eskf::state[7], eskf::state[8], eskf::state[9], eskf::state[6]);
  tf::Matrix3x3(odom_quat).getRPY(current_pose_roll, current_pose_pitch, current_pose_yaw);

  // 发布坐标变换 T_map_to_odom
  tf::TransformBroadcaster broadcaster;
  geometry_msgs::TransformStamped odom_trans;
  odom_trans.header.frame_id = "map";
  odom_trans.child_frame_id = "odom_base_link";
  odom_trans.header.stamp = last_imu_stamp;
  odom_trans.transform.translation.x = current_pose_x;
  odom_trans.transform.translation.y = current_pose_y;
  odom_trans.transform.translation.z = current_pose_z;
  odom_trans.transform.rotation =
      tf::createQuaternionMsgFromRollPitchYaw(current_pose_roll, current_pose_pitch, current_pose_yaw);
  broadcaster.sendTransform(odom_trans);

  // 发布里程计消息 T_map_to_odom
  nav_msgs::Odometry odom;
  odom.header.stamp = last_imu_stamp;
  odom.header.frame_id = "map";
  odom.child_frame_id = "odom_base_link";
  odom.pose.pose.position.x = current_pose_x;
  odom.pose.pose.position.y = current_pose_y;
  odom.pose.pose.position.z = current_pose_z;
  odom.pose.pose.orientation =
      tf::createQuaternionMsgFromRollPitchYaw(current_pose_roll, current_pose_pitch, current_pose_yaw);
  odom.twist.twist.linear.x = odom_linear_x;
  odom.twist.twist.linear.y = 0.0;
  odom.twist.twist.linear.z = 0.0;
  odom.twist.twist.angular.x = 0.0;
  odom.twist.twist.angular.y = 0.0;
  odom.twist.twist.angular.z = odom_angular_z;
  odom_pub.publish(odom);

  // 发布车辆线速度和角速度
  geometry_msgs::TwistStamped encoder_twist;      // 里程计消息
  encoder_twist.header.stamp = ros::Time::now();  // 当前时间
  encoder_twist.header.frame_id = "base_link";
  encoder_twist.twist = odom.twist.twist;
  encoder_twist_pub.publish(encoder_twist);

  // // 发布车辆线速度和角速度
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

  eskf::relocalization(x, y, z, roll, pitch, yaw);
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
    eskf::relocalization(x, y, z, roll, pitch, yaw);
  }
}

void emergency_switch_Callback(const std_msgs::Bool::ConstPtr& msg)
{
  emergency_button_pressed_flag = msg->data;
}

void CalibratedPoseCallback(const geometry_msgs::PoseStampedPtr& pose)
{
  if (initialized_flag)
  {
    ndt_pose_x = pose->pose.position.x;
    ndt_pose_y = pose->pose.position.y;
    ndt_pose_z = pose->pose.position.z;

    tf::Quaternion RQ2;
    tf::quaternionMsgToTF(pose->pose.orientation, RQ2);
    tf::Matrix3x3(RQ2).getRPY(ndt_pose_roll, ndt_pose_pitch, ndt_pose_yaw);

    offset_x = ndt_pose_x - previous_pose_x;
    offset_y = ndt_pose_y - previous_pose_y;
    offset_z = ndt_pose_z - previous_pose_z;
    offset_yaw = ndt_pose_yaw - previous_pose_yaw;

    offset_yaw > M_PI ? offset_yaw -= 2 * M_PI : offset_yaw < -M_PI ? offset_yaw += 2 * M_PI : offset_yaw = offset_yaw;

    double pose_distance = sqrt(offset_x * offset_x + offset_y * offset_y);

    if (waypoint_change_flag == 4)
    {
      fitness_above_threshold = fitness_above_threshold_indoor;
    }
    else
    {
      fitness_above_threshold = fitness_above_threshold_outdoor;
    }

    MovingAverage avg(6);

    if ((waypoint_change_flag == 5 || use_wheel_odom_lift_flag) && use_encoder_initialized)
    {
      PERIOD = ENCODER_TransNum;
      if (fitness_score < 3.0)
      {
        eskf::relocalization(ndt_pose_x, ndt_pose_y, ndt_pose_z, ndt_pose_roll, ndt_pose_pitch, ndt_pose_yaw);
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

    fitness_score_median = avg.next(fitness_score);

    if ((fitness_score_median < (fitness_score_mean + fitness_above_threshold)) && (pose_distance < 10.0) &&
        encoder_iteration_num >= PERIOD)
    {
      eskf::relocalization(ndt_pose_x, ndt_pose_y, ndt_pose_z, ndt_pose_roll, ndt_pose_pitch, ndt_pose_yaw);
      encoder_iteration_num = 0;
    }

    previous_pose_x = ndt_pose_x;
    previous_pose_y = ndt_pose_y;
    previous_pose_z = ndt_pose_z;
    previous_pose_yaw = ndt_pose_yaw;
  }
}

void ekf_predict(double v, double w, double dt)
{
  double linear_vel = (v + last_v) / 2.0;
  double angular_vel = (w + last_w) / 2.0;

  state[0] += linear_vel * dt * cos(state[2]);
  state[1] += linear_vel * dt * sin(state[2]);
  state[2] += angular_vel * dt;

  F << 1.0, 0.0, -linear_vel * dt * sin(state[2]), 0.0, 1.0, linear_vel * dt * cos(state[2]), 0.0, 0.0, 1.0;
  P = F * P * F.transpose() + Q;

  odom_angular_z = angular_vel;
  odom_linear_x = linear_vel;
  last_v = v;
  last_w = w;

  if (encoder_iteration_num < PERIOD)
  {
    encoder_iteration_num++;
  }
}

void encoders_callback(const zr_msgs::motor_info::ConstPtr& msg)
{
  if (!emergency_button_pressed_flag)
  {
    ros::Time cur = msg->header.stamp;
    double v = (msg->left_vel + msg->right_vel) / 2.0;
    double w = (msg->right_vel - msg->left_vel) / wheelbase;

    if (first_enc_msg)
    {
      last_motorinfo_stamp = cur;
      last_v = v;
      last_w = w;
      first_enc_msg = false;
      return;
    }
    double dt = (cur - last_motorinfo_stamp).toSec();
    last_motorinfo_stamp = cur;
    eskf::meaturement(v, w, dt);
    pub_odom();
  }
}

void imu_twist_callback(const geometry_msgs::TwistStamped::ConstPtr& msg_ptr)
{
  // 急停模式下通过imu线速度和角速度代替编码器线速度和角速度
  if (emergency_button_pressed_flag && waypoint_change_flag == 5)
  {
    ros::Time cur = msg_ptr->header.stamp;
    double v = msg_ptr->twist.linear.x;
    double w = msg_ptr->twist.angular.z;

    if (first_imu_twist_msg)
    {
      last_imu_twist_stamp = cur;
      first_imu_twist_msg = false;
      last_v = v;
      last_w = w;
      return;
    }

    double dt = (cur - last_motorinfo_stamp).toSec();
    eskf::meaturement(v, w, dt);
    if (encoder_iteration_num < PERIOD)
    {
      encoder_iteration_num++;
    }
  }
  else
  {
    first_imu_twist_msg = true;
  }
}

void imu_callback(const sensor_msgs::Imu::ConstPtr& imu_msg)
{
  ros::Time cur = imu_msg->header.stamp;
  if (first_imu_msg && first_enc_msg)
  {
    first_imu_msg = false;
    last_imu_stamp = cur;
    return;
  }
  double dt = (cur - last_imu_stamp).toSec();
  eskf::predict(imu_msg, dt);
  last_imu_stamp = cur;
}

void pose_param_callback(const autoware_msgs::ConfigNdt::ConstPtr& input)
{
  if (initialized_flag == false)
  {
    double x = input->x;
    double y = input->y;
    double z = input->z;

    double roll = input->roll;
    double pitch = input->pitch;
    double yaw = input->yaw;
    eskf::relocalization(x, y, z, roll, pitch, yaw);
    initialized_flag = true;
  }
}

}  // namespace wheel_odom_eskf
int main(int argc, char** argv)
{
  ros::init(argc, argv, "wheel_odom_eskf");
  ros::NodeHandle nh("~");

  // Load parameters from YAML file
  std::string yaml_file;

  nh.param<std::string>("file_path", yaml_file,
                        "/home/justin/ZROS/ros/src/localization/packages/wheel_odom_kalman/config/encoder.yaml");

  YAML::Node config = YAML::LoadFile(yaml_file);

  std::string robot_id;
  nh.param<std::string>("/robot_id", robot_id,
                        "ZR1001");  // note-justin   如果无法获得main_socket/campus_id,
                                    // 就默认读取map11150

  // // Get wheel parameters for specific robot from YAML file
  wheel_odom_eskf::wheel_radius = config[robot_id]["wheel_radius"].as<double>();
  wheel_odom_eskf::encoder_ticks_per_rev = config[robot_id]["encoder_ticks_per_rev"].as<int>();
  wheel_odom_eskf::wheelbase = config[robot_id]["wheelbase"].as<double>();
  wheel_odom_eskf::wheelratio = config[robot_id]["wheelratio"].as<double>();
  std::vector<double> t_i_enc;
  t_i_enc = config[robot_id]["T_imu_enc"].as<std::vector<double>>();
  eskf::T_imu_enc = Eigen::Map<const Eigen::Matrix<double, 4, 4>>(t_i_enc.data()).transpose();

  ROS_INFO(
      "wheel_odom_eksf_node:: robot_id = %s wheel_radius = %f "
      "encoder_ticks_per_rev = %d "
      "wheelbase = %f wheelratio = %f",
      robot_id.c_str(), wheel_odom_eskf::wheel_radius, wheel_odom_eskf::encoder_ticks_per_rev,
      wheel_odom_eskf::wheelbase, wheel_odom_eskf::wheelratio);

  wheel_odom_eskf::Q << 0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.1;
  wheel_odom_eskf::P << 0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01;
  wheel_odom_eskf::R << 0.001;
  wheel_odom_eskf::H << 0.0, 0.0, 1.0;

  wheel_odom_eskf::odom_pub = nh.advertise<nav_msgs::Odometry>("/wheel_odom_eskf", 1000);
  wheel_odom_eskf::odom_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/wheel_odom_pose", 100);

  wheel_odom_eskf::ndt_pose_to_odom_pub = nh.advertise<std_msgs::Int32>("/ndt_pose_to_odom", 10);
  wheel_odom_eskf::encoder_twist_pub = nh.advertise<geometry_msgs::TwistStamped>("/encoder_twist", 10);
  wheel_odom_eskf::fitness_score_median_pub = nh.advertise<std_msgs::Float32>("/fitness_score_median", 10);
  wheel_odom_eskf::fitness_score_mean_pub = nh.advertise<std_msgs::Float32>("/fitness_score_mean", 10);
  wheel_odom_eskf::post_distance_pub = nh.advertise<std_msgs::Float32>("/store_pose_distance", 10);

  wheel_odom_eskf::odom_yaw_pub = nh.advertise<std_msgs::Float32>("/odom_yaw", 10);
  wheel_odom_eskf::odom_twist_angular_pub = nh.advertise<std_msgs::Float32>("/twist_angular", 10);
  wheel_odom_eskf::odom_twist_linear_pub = nh.advertise<std_msgs::Float32>("/twist_linear", 10);

  ros::Subscriber sub_imu = nh.subscribe("/imu_data", 10, wheel_odom_eskf::imu_callback);
  ros::Subscriber sub = nh.subscribe("/motor_info", 10, wheel_odom_eskf::encoders_callback);

  ros::Subscriber sub_initalpose = nh.subscribe("/initialpose", 10, wheel_odom_eskf::initialpose_callback);
  ros::Subscriber sub_pose1 = nh.subscribe("/ekf_pose", 10, wheel_odom_eskf::CalibratedPoseCallback);
  ros::Subscriber sub_liftpose = nh.subscribe("/liftpose", 100, wheel_odom_eskf::liftpose_callback);

  ros::Subscriber sub_start = nh.subscribe("/start_flag", 10, wheel_odom_eskf::StartWheelOdomCallback);
  ros::Subscriber sub_reset = nh.subscribe("/reset_wheel_odom", 10, wheel_odom_eskf::ResetCallback);
  ros::Subscriber emergency_switch_reset =
      nh.subscribe("/emergency_switch", 10, wheel_odom_eskf::emergency_switch_Callback);
  ros::Subscriber sub_ndt_stat = nh.subscribe("/ndt_stat", 10, wheel_odom_eskf::NdtStatCallback);
  ros::Subscriber pose_param_sub = nh.subscribe("/config/ndt", 10, wheel_odom_eskf::pose_param_callback);
  ros::Subscriber lift_flag_sub =
      nh.subscribe("/global_waypoint_change_flag", 10, wheel_odom_eskf::waypointflagCallback);
  ros::Subscriber use_wheel_odom_sub = nh.subscribe("/use_wheel_odom", 10, wheel_odom_eskf::use_wheel_odom_callback);
  ros::Subscriber imu_odom_sub = nh.subscribe("/imu_twist", 10, wheel_odom_eskf::imu_twist_callback);

  eskf::init_eskf(nh);
  ros::spin();
  return 0;
}