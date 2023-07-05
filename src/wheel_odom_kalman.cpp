/* Copyright (c) 2020 ZhenRoot, Inc. All Rights Reserved
 *

 * last modified 2022/08/06 *
 * 混合型里程计，具备自我恢复功能，可以通过简单的节点间握手改变里程计的运行状态，
 *
 * 实现多种功能。
 *
 */
#include <string>
#include <ros/ros.h>
// #include <sensor_msgs/JointState.h>
#include <tf/transform_broadcaster.h>  //左边变换 广播
#include <nav_msgs/Odometry.h>		   //导航下的里程计消息
#include <zr_msgs/motor_info.h>
#include <tf/tf.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <vector>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/Imu.h>
#include "std_msgs/Bool.h"
#include <std_msgs/Int32.h>
#include <std_msgs/UInt8MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <autoware_msgs/ndt_stat.h>
#include <autoware_msgs/ConfigNdt.h>
#include <math.h>
#include <yaml-cpp/yaml.h>
#include <deque>
#include <numeric>
#include <Eigen/Dense>

/*
 * 全局变量
 */
ros::Publisher imu_twist_pub;

static double last_yaw = 0.0;
static double last_vx = 0.0;
static double last_vy = 0.0;
static double last_vz = 0.0;

float V_L;
float V_R;
static double odom_linear_speed;   //= (V_L + V_R) / 2;
static double odom_angular_speed;  //= (V_R - V_L) / 0.66;
bool lift_start = false;
float origin_x;
float origin_y;
float origin_th;
static bool initialized_flag_pose = true;
static bool initialized_flag_param = true;

static bool initialized_flag = false;
static double current_pose_x = 0.0;
static double current_pose_y = 0.0;
static double current_pose_z = 0.0;

static double store_pose_x = 0.0;
static double store_pose_y = 0.0;
static double store_pose_z = 0.0;
static double store_pose_roll = 0.0;
static double store_pose_pitch = 0.0;
static double store_pose_yaw = 0.0;

static double previous_pose_x = 0.0;
static double previous_pose_y = 0.0;
static double previous_pose_z = 0.0;

static double current_pose_roll = 0.0;
static double current_pose_pitch = 0.0;
static double current_pose_yaw = 0.0;
static double odom_yaw = 0.0;
static double odom_pose_x = 0.0;
static double odom_pose_y = 0.0;
static double odom_pose_z = 0.0;
static double odom_pose_roll = 0.0;
static double odom_pose_pitch = 0.0;
static double odom_pose_yaw = 0.0;

static double fitness_score_median = 0.0;
static bool _use_local_transform = false;

static double ndt_pose_x = 0.0;
static double ndt_pose_y = 0.0;
static double ndt_pose_z = 0.0;

static double previous_new_pose_x = 0.0;
static double previous_new_pose_y = 0.0;
static double previous_new_pose_z = 0.0;

static double previous_new_pose_roll = 0.0;
static double previous_new_pose_pitch = 0.0;
static double previous_new_pose_yaw = 0.0;
static double imu_angular_z = 0.0;
static double imu_linear_acceleration = 0.0;

static double offset_x = 0.0;
static double offset_y = 0.0;
static double offset_z = 0.0;

static double offset_roll = 0.0;
static double offset_pitch = 0.0;
static double offset_yaw = 0.0;
static bool indoor_ = true;

double store_pose_distance = 0.0;

static double odom_roll = 0.0;
static double odom_pitch = 0.0;
#define PI (3.1415926)
double clock_x;
double clock_y;
double clock_th;
// ros::Time All_Clock1;
// ros::Time All_Clock2;
static int encoder_iteration_num = 0;
static int TransNum = 200;				  // note-justin 10 second for period
static int ENCODER_TransNum = 500000000;  // note-justin 5 second for period
static int PERIOD = 2000;				  // note-justin 50 代表50秒 100代表10秒 200代表20秒
static int FitnessLoop = 0;				  // the loop for keep on encoder
static int fitness_cnt = 0;
static int fitness_cnt_divide;
static float fitness_score = 0.0;
static double fitness_score_sum = 0.0;
static double fitness_score_mean = 0.0;

static bool emergency_button_pressed_flag = false;

static float fitness_above_threshold = 0.0;  // threshold above 0.5 indoor 3.0 outdoor
static int normal_ndt_cnt = 0;
static int encoder_reposition_cnt = 0;

static int fitness_score_cnt = 0;
static float fitness_score0 = 0.0;
static float fitness_score1 = 0.0;
static float fitness_score2 = 0.0;
static float fitness_score3 = 0.0;
static float fitness_score4 = 0.0;
static float fitness_score5 = 0.0;

static float close_info_yaw_score = 0.0;
static float offset_imu_y = 0.0;
static bool frist_imu_flag = true;
static double err_of_imu;
static double imu_calibrated_yaw = 0.0;
double roll, pitch, yaw;
bool start_flag = false;
static double ndt_pose_roll = 0.0;
static double ndt_pose_pitch = 0.0;
static double ndt_pose_yaw = 0.0;
static int waypoint_change_flag = 0;
static bool lift_change_flag = true;
static bool use_encoder_initialized = true;
static bool anti_encoder_initialized = true;
static bool use_wheel_odom_lift_flag = false;

#define odom_param_indoor 0.54
#define odom_param_outdoor 0.60

#define fitness_above_threshold_indoor 0.5
#define fitness_above_threshold_outdoor 20.0
bool no_auto_reset = false;

static double wheel_radius = 0.1;
static int encoder_ticks_per_rev = 1000;
static double wheelbase = 0.5;
static double wheelratio = 0.95;

const int MA_WINDOW_SIZE = 10;  // size of the moving average window
static double VL_filtered = 0.0;
static double VR_filtered = 0.0;

std::deque<double> VL_ma_buffer(MA_WINDOW_SIZE, 0.0);
std::deque<double> VR_ma_buffer(MA_WINDOW_SIZE, 0.0);

const int MEDIAN_FILTER_SIZE = 5;		   // size of the median filter window
std::vector<double> imu_angular_z_buffer;  // buffer for storing IMU angular velocity samples

using namespace Eigen;

Eigen::VectorXd x(5);	  // 状态向量
Eigen::MatrixXd P(5, 5);   // 方差矩阵
Eigen::MatrixXd Q(5, 5);   // 过程噪声方差矩阵
Eigen::MatrixXd R1(2, 2);  // 观测噪声方差矩阵1
Eigen::MatrixXd R2(9, 9);  // 观测噪声方差矩阵2

// Define system model matrices
MatrixXd A(5, 5);
MatrixXd B(5, 2);
MatrixXd C1(2, 5);
MatrixXd C2(9, 5);

ros::Time last_time;
ros::Publisher odom_encoder_pub;
ros::Publisher odom_fused_pub;

// bool switch_flag;

/* 函数名：ModeTransCallback
 * 功能：切换里程计模式，辅助定位模式，主动定位模式
 * 输入话题：Bool型，true激活主动定位模式，false激活辅助定位模式
 *
 */
void ModeTransCallback(const std_msgs::Bool::ConstPtr& msg)
{
  if (msg->data == true)
  {
	no_auto_reset = true;
	TransNum = 500;
	ROS_INFO("wheel_odom.cpp:: ModeTransCallback TransNum = %d use_wheel_odom = %d ", TransNum, msg->data);
  }
  if (msg->data == false)
  {
	no_auto_reset = false;
	TransNum = 500;
	ROS_INFO("wheel_odom.cpp:: ModeTransCallback TransNum = %d use_wheel_odom = %d ", TransNum, msg->data);
  }
}

void use_wheel_odom_callback(const std_msgs::BoolConstPtr& msg)
{
  use_wheel_odom_lift_flag = msg->data;

  ROS_INFO("ndt_matching.cpp:: use_wheel_odom_callback  use_wheel_odom_lift_flag = %d ", use_wheel_odom_lift_flag);
}

void ResetCallback(const std_msgs::Bool::ConstPtr& msg)
{
  if (msg->data == true)
  {
	encoder_iteration_num = TransNum + 10;
  }
}

void waypointflagCallback(const std_msgs::Int32::ConstPtr& msg)
{
  waypoint_change_flag = msg->data;
}

/* 函数名：ModeTransCallback
 * 功能：切换里程计模式，辅助定位模式，主动定位模式
 * 输入话题：Bool型，true激活主动定位模式，false激活辅助定位模式
 *
 */
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

  // ROS_ERROR("wheel_odom.cpp: ModeTransCallback fitness_score_cnt = %d fitness_score_sum = %f fitness_score_mean =
  // %f", fitness_score_cnt, fitness_score_sum, fitness_score_mean); std::cout << "fitness_score = " << fitness_score <<
  // std::endl;
}

void CloseYawCallback(const std_msgs::Float32ConstPtr& msg)
{
  close_info_yaw_score = msg->data;
  // std::cout << "fitness_score = " << fitness_score << std::endl;
}

void offset_imu_y_Callback(const std_msgs::Float32ConstPtr& msg)
{
  offset_imu_y = msg->data;
  // std::cout << "fitness_score = " << fitness_score << std::endl;
}

/* Function name: api of start the progress
 * Sub topic: wheel_odom_flag
 * Abstract: plz use this api to start the progress of wheel odom
 * when the data is true, the progress will start,when the data is false
 * the progress is waiting for starting.I think you should sub the topic
 * with the data is ture before the lift progress,when the lift progress is over
 * plz sub the topic with the data is false to stop the odom progress.
 */
void StartWheelOdomCallback(const std_msgs::Int32ConstPtr& msg)
{
  start_flag = true;
}

/* 函数名：initialpose_callback
 * 功能：坐标初始化，当定位丢失时可以手动恢复
 * 输入话题：rviz的inital_pose
 */
void initialpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& input)
{
  if (initialized_flag_pose == true)
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

	if (_use_local_transform == true)
	{
	  current_pose_x = input->pose.pose.position.x;
	  current_pose_y = input->pose.pose.position.y;
	  current_pose_z = input->pose.pose.position.z;
	  // ROS_INFO("wheel_odom.cpp:: initialpose_callback current_pose_x = %f current_pose_y = %f current_pose_z = %f
	  // _use_local_transform = %d", current_pose_x, current_pose_y, current_pose_z, _use_local_transform);
	}
	else
	{
	  // Justin 从rviz手动给出的pose是基于world坐标系的，所以需要将其转换到map坐标系
	  current_pose_x = input->pose.pose.position.x + transform.getOrigin().x();
	  current_pose_y = input->pose.pose.position.y + transform.getOrigin().y();
	  current_pose_z = input->pose.pose.position.z + transform.getOrigin().z();
	  // ROS_INFO("wheel_odom.cpp:: initialpose_callback current_pose_x = %f current_pose_y = %f current_pose_z = %f
	  // _use_local_transform = %d", current_pose_x, current_pose_y, current_pose_z, _use_local_transform);
	}
	m.getRPY(roll, pitch, yaw);

	ndt_pose_roll = roll;
	ndt_pose_pitch = pitch;
	ndt_pose_yaw = yaw;
	current_pose_yaw = yaw;

	odom_pose_x = current_pose_x;
	odom_pose_y = current_pose_y;
	odom_pose_z = current_pose_z;
	odom_pose_yaw = yaw;
	imu_calibrated_yaw = yaw;

	geometry_msgs::TransformStamped odom_trans;  // 坐标变换消息
	odom_trans.header.frame_id = "map";
	odom_trans.child_frame_id = "odom_base_link";

	ROS_INFO("wheel_odom.cpp:: initialpose_callback current_pose_x = %f current_pose_y = %f  current_pose_z = %f",
			 current_pose_x, current_pose_y, current_pose_z);
	ROS_INFO("wheel_odom.cpp:: initialpose_callback odom_pose_yaw = %f ndt_pose_yaw = %f", odom_pose_yaw, ndt_pose_yaw);

	odom_trans.header.stamp = ros::Time::now();		   // 当前时间
	odom_trans.transform.translation.x = odom_pose_x;  // 位置
	odom_trans.transform.translation.y = odom_pose_y;
	odom_trans.transform.translation.z = odom_pose_z;
	odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(odom_pose_yaw);  // 位姿

	// Justin 这里主要是为了确认一个z值
  }
}

static void param_callback(const autoware_msgs::ConfigNdt::ConstPtr& input)
{
  if (initialized_flag_param == true)
  {
	tf::TransformListener listener;
	tf::StampedTransform transform;
	try
	{
	  ros::Time now = ros::Time(0);
	  listener.waitForTransform("map", "map", now, ros::Duration(10.0));
	  listener.lookupTransform("map", "map", now, transform);
	}
	catch (tf::TransformException& ex)
	{
	  // ROS_ERROR("%s", ex.what());
	}

	if (_use_local_transform == true)
	{
	  current_pose_x = input->x;
	  current_pose_y = input->y;
	  current_pose_z = input->z;
	  // ROS_INFO("wheel_odom.cpp:: param_callback current_pose_x = %f current_pose_y = %f current_pose_z = %f
	  // _use_local_transform = %d", current_pose_x, current_pose_y, current_pose_z, _use_local_transform);
	}
	else
	{
	  // Justin 从rviz手动给出的pose是基于world坐标系的，所以需要将其转换到map坐标系
	  current_pose_x = input->x + transform.getOrigin().x();
	  current_pose_y = input->y + transform.getOrigin().y();
	  current_pose_z = input->z + transform.getOrigin().z();
	  // ROS_INFO("wheel_odom.cpp:: param_callback current_pose_x = %f current_pose_y = %f current_pose_z = %f
	  // _use_local_transform = %d", current_pose_x, current_pose_y, current_pose_z, _use_local_transform);
	}

	ndt_pose_roll = input->roll;
	ndt_pose_pitch = input->pitch;
	ndt_pose_yaw = input->yaw;
	current_pose_yaw = input->yaw;

	odom_pose_x = current_pose_x;
	odom_pose_y = current_pose_y;
	odom_pose_z = current_pose_z;
	odom_pose_yaw = input->yaw;
	imu_calibrated_yaw = input->yaw;

	geometry_msgs::TransformStamped odom_trans;  // 坐标变换消息
	odom_trans.header.frame_id = "map";
	odom_trans.child_frame_id = "odom_base_link";

	ROS_INFO("wheel_odom.cpp:: param_callback odom_pose_yaw = %f  ndt_pose_yaw = %f", odom_pose_yaw, ndt_pose_yaw);

	odom_trans.header.stamp = ros::Time::now();		   // 当前时间
	odom_trans.transform.translation.x = odom_pose_x;  // 位置
	odom_trans.transform.translation.y = odom_pose_y;
	odom_trans.transform.translation.z = odom_pose_z;
	odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(odom_pose_yaw);  // 位姿

	// Justin 这里主要是为了确认一个z值
  }
}

void liftpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& pose)
{
  if (lift_change_flag == true)  // change
  {
	PERIOD = ENCODER_TransNum;
	current_pose_x = pose->pose.pose.position.x;
	current_pose_y = pose->pose.pose.position.y;
	current_pose_z = pose->pose.pose.position.z;

	tf::Quaternion RQ2;
	double roll, pitch, yaw;
	tf::quaternionMsgToTF(pose->pose.pose.orientation, RQ2);
	tf::Matrix3x3(RQ2).getRPY(roll, pitch, yaw);

	current_pose_roll = 0.0;
	current_pose_pitch = 0.0;
	current_pose_yaw = yaw;

	encoder_iteration_num = 0;

	// ROS_ERROR("wheel_odom.cpp::liftpose_callback != 5 lift_change_flag = %d", lift_change_flag);
	// ROS_ERROR("wheel_odom.cpp::liftpose_callback = %f  odom_pose_y = %f odom_pose_yaw = %f", current_pose_x,
	// current_pose_y, current_pose_z);

	lift_change_flag = false;
  }
}

static void pose_param_callback(const autoware_msgs::ConfigNdt::ConstPtr& input)
{
  if (initialized_flag == false)
  {
	ndt_pose_x = input->x;
	ndt_pose_y = input->y;
	ndt_pose_z = input->z;

	ndt_pose_roll = input->roll;
	ndt_pose_pitch = input->pitch;
	ndt_pose_yaw = input->yaw;

	initialized_flag = true;

	// ROS_ERROR("wheel_odom.cpp:pose_param_callback:ndt_pose_x: [%f], wheel_odom.cpp:pose_param_callback:ndt_pose_y:
	// [%f]", ndt_pose_x, ndt_pose_y); ROS_ERROR("wheel_odom.cpp:pose_param_callback:roll: [%f],
	// wheel_odom.cpp:pose_param_callback:yaw: [%f]", ndt_pose_roll, ndt_pose_yaw);
  }
}

/* 函数名：VelCallback
 * 接收话题：来自base_driver的双轮motor_vel信息
 * 功能：引入数据给全局变量
 */

void emergency_switch_Callback(const std_msgs::Bool::ConstPtr& msg)
{
  if (msg->data == true)
  {
	emergency_button_pressed_flag = true;
	ROS_INFO("wheel_odom.cpp:: emergency_switch_Callback emergency_button_pressed_flag = %d",
			 emergency_button_pressed_flag);
  }
  else
  {
	emergency_button_pressed_flag = false;
	// ROS_INFO("wheel_odom.cpp:: emergency_switch_Callback emergency_button_pressed_flag = %d",
	// emergency_button_pressed_flag);
  }
}

void motorInfoCallback(const zr_msgs::motor_info::ConstPtr& motor_vel)
{
  V_L = motor_vel->left_vel;
  V_R = motor_vel->right_vel;

  /*
  VL_ma_buffer.pop_front();
  VL_ma_buffer.push_back(V_L);
  VR_ma_buffer.pop_front();
  VR_ma_buffer.push_back(V_R);
  double VL_ma = std::accumulate(VL_ma_buffer.begin(), VL_ma_buffer.end(), 0.0) / MA_WINDOW_SIZE;
  double VR_ma = std::accumulate(VR_ma_buffer.begin(), VR_ma_buffer.end(), 0.0) / MA_WINDOW_SIZE;
*/

  // ROS_INFO("wheel_odom.cpp:: motorInfoCallback left_vel: [%f]", V_L);
  // ROS_INFO("wheel_odom.cpp:: motorInfoCallback right_vel: [%f]", V_R);
  //  V_L = V_L + (rand() % 100) / float(1000);
  //  V_R = V_R + (rand() % 100) / float(1000); //TEST different scenario Usefull

  // ROS_ERROR("wheel_odom.cpp a = %f b = %f", (rand()%100)/float(1000), (rand()%100)/float(1000));

  if (waypoint_change_flag == 4)
  {
	odom_linear_speed = (V_L + V_R) / 2 * wheelratio;
	odom_angular_speed =
		(V_R - V_L) /
		odom_param_indoor;  // 0.54 adjust the real world and odom  0.612
							// ROS_INFO("wheel_odom.cpp::motorInfoCallback:left_vel: [%f], motorInfoCallback:right_vel:
							// [%f], odom_param_indoor =%f", V_L, V_R, odom_param_indoor);
							// ROS_INFO("wheel_odom.cpp::motorInfoCallback:odom_linear_speed: [%f],
							// motorInfoCallback:odom_angular_speed: [%f]", odom_linear_speed, odom_angular_speed);
  }
  else if (waypoint_change_flag == 0)
  {
	odom_linear_speed = (V_L + V_R) / 2 * wheelratio;
	odom_angular_speed =
		(V_R - V_L) /
		odom_param_outdoor;  // 0.54 adjust the real world and odom  0.612
							 // ROS_INFO("wheel_odom.cpp::motorInfoCallback:left_vel: [%f], motorInfoCallback:right_vel:
							 // [%f], odom_param_outdoor", V_L, V_R, odom_param_outdoor);
							 // ROS_INFO("wheel_odom.cpp::motorInfoCallback:odom_linear_speed: [%f],
							 // motorInfoCallback:odom_angular_speed: [%f]", odom_linear_speed, odom_angular_speed);
  }
  else
  {
	odom_linear_speed = (V_L + V_R) / 2 * wheelratio;
	odom_angular_speed = (V_R - V_L) / odom_param_outdoor;  // 0.54 adjust the real world and odom  0.612
  }

  // 使用电机速度测量更新状态向量
  x(3) = odom_linear_speed;
  x(4) = odom_angular_speed;

  // 使用过程噪声方差矩阵更新状态协方差矩阵
  P = A * P * A.transpose() + Q;

  // 使用Kalman滤波器更新状态向量和状态协方差矩阵
  MatrixXd K = P * C1.transpose() * (C1 * P * C1.transpose() + R1).inverse();
  VectorXd z(2);
  z << odom_linear_speed, odom_angular_speed;
  x = x + K * (z - C1 * x);
  P = (MatrixXd::Identity(5, 5) - K * C1) * P;

  // 计算机器人的里程计信息
  double dt = (motor_vel->header.stamp - last_time).toSec();

  if (dt >= 1.0)  // note-justin  初次计算的时候两次时间差会很大
  {
	dt = 0.02;
	// ROS_ERROR("nimu_dt_matching.cpp:: FIRST initialized_flag = %d imu_dt = %f", initialized_flag, imu_dt);
  }

  last_time = motor_vel->header.stamp;
  double dx = x(3) * cos(x(2)) * dt;
  double dy = x(3) * sin(x(2)) * dt;
  double dtheta = x(4) * dt;
  double vx = x(3);
  double vy = 0.0;
  double vtheta = x(4);
  nav_msgs::Odometry odom_msg;
  odom_msg.header.stamp = motor_vel->header.stamp;
  odom_msg.header.frame_id = "odom";
  odom_msg.child_frame_id = "base_link";
  odom_msg.pose.pose.position.x += dx;
  odom_msg.pose.pose.position.y += dy;
  odom_msg.pose.pose.orientation.z += sin(dtheta / 2.0);
  odom_msg.pose.pose.orientation.w += cos(dtheta / 2.0);
  odom_msg.twist.twist.linear.x = vx;
  odom_msg.twist.twist.linear.y = vy;
  odom_msg.twist.twist.angular.z = vtheta;

  // 发布里程计信息
  odom_encoder_pub.publish(odom_msg);

  ROS_INFO(
	  "wheel_odom.cpp:: motorInfoCallback:left_vel: [%f], motorInfoCallback:right_vel: [%f] left_vel: [%f], "
	  "motorInfoCallback:right_vel: [%f]",
	  V_L, V_R, V_L, V_R);
  ROS_INFO("wheel_odom.cpp:: motorInfoCallback:odom_linear_speed: [%f], motorInfoCallback:odom_angular_speed: [%f]",
		   odom_linear_speed, odom_angular_speed);

  // ROS_INFO("Thanks: [%s]", msg->data.c_str());
}

/* 函数名：PoseCallback1
 * 接收话题：PoseStamped（current_pose）
 * 功能：重置里程计的误差
 */

void CalibratedPoseCallback(const geometry_msgs::PoseStampedPtr& pose)
{
  if (initialized_flag == true)
  {
	ndt_pose_x = pose->pose.position.x;
	ndt_pose_y = pose->pose.position.y;
	ndt_pose_z = pose->pose.position.z;

	tf::Quaternion RQ2;
	double roll, pitch, yaw;
	tf::quaternionMsgToTF(pose->pose.orientation, RQ2);
	tf::Matrix3x3(RQ2).getRPY(roll, pitch, yaw);
	ndt_pose_roll = roll;
	ndt_pose_pitch = pitch;
	ndt_pose_yaw = yaw;  //-pi~pi
  }

  // ROS_INFO("wheel_odom.cpp:: initialized_flag: %d", initialized_flag);
  // ROS_INFO("wheel_odom.cpp::pose->pose.position.x: %f, pose->pose.position.y: %f, pose->pose.position.z: %f,
  // ndt_pose_yaw: %f,", pose->pose.position.x, pose->pose.position.y, pose->pose.position.z, ndt_pose_yaw);
}

void CurrentPoseCB(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
  // 更新卡尔曼滤波器融合后的里程计
  // 可以使用以下代码来更新：
  x << msg->pose.pose.position.x, msg->pose.pose.position.y, yaw, x(3), x(4);
  P << msg->pose.covariance[0], 0, 0, 0, 0, 0, msg->pose.covariance[6], 0, 0, 0, 0, 0, msg->pose.covariance[35], 0, 0,
	  0, 0, 0, P(3, 3), 0, 0, 0, 0, 0, P(4, 4);

  // 更新后可以重新发布卡尔曼滤波器融合后的里程计
}
/* 函数名：GetImuDataCB
 * 接收话题：来自Imu节点的imu_info
 * 功能：获得当前方向角（基于map坐标系）
 */

void ImuDataCB(const sensor_msgs::Imu::ConstPtr& imu_data)
{
  // Extract relevant data from IMU message
  double ax = imu_data->linear_acceleration.x;
  double ay = imu_data->linear_acceleration.y;
  double az = imu_data->linear_acceleration.z;
  double wx = imu_data->angular_velocity.x;
  double wy = imu_data->angular_velocity.y;
  double wz = imu_data->angular_velocity.z;

  // Update state vector with IMU measurements
  double dt = (imu_data->header.stamp - last_time).toSec();
  if (dt >= 1.0)  // note-justin  初次计算的时候两次时间差会很大
  {
	dt = 0.02;
	// ROS_ERROR("nimu_dt_matching.cpp:: FIRST initialized_flag = %d imu_dt = %f", initialized_flag, imu_dt);
  }
  last_time = imu_data->header.stamp;
  x(0) = x(0) + x(3) * cos(x(2)) * dt;
  x(1) = x(1) + x(3) * sin(x(2)) * dt;
  x(2) = x(2) + x(4) * dt;

  // Update state covariance matrix with process noise
  P = A * P * A.transpose() + Q;

  // Perform Kalman filter update using IMU measurements
  MatrixXd K = P * C2.transpose() * (C2 * P * C2.transpose() + R2).inverse();
  VectorXd z(9);
  z << ax, ay, az, wx, wy, wz, x(3), x(4), x(2);
  x = x + K * (z - C2 * x);
  P = (MatrixXd::Identity(5, 5) - K * C2) * P;

  // Publish robot pose as a PoseStamped message and a TF transform

  // 发布融合后的里程计信息到wheel_odom话题上
  nav_msgs::Odometry odom;
  odom.header.stamp = imu_data->header.stamp;
  odom.header.frame_id = "odom";
  odom.pose.pose.position.x = x(0);
  odom.pose.pose.position.y = x(1);
  odom.pose.pose.position.z = 0;
  odom.pose.pose.orientation.x = 0;
  odom.pose.pose.orientation.y = 0;
  odom.pose.pose.orientation.z = sin(x(2) / 2);
  odom.pose.pose.orientation.w = cos(x(2) / 2);
  odom.twist.twist.linear.x = x(3);
  odom.twist.twist.angular.z = x(4);
  odom_fused_pub.publish(odom);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "wheel_odom_kalman");
  ros::NodeHandle n;
  n.getParam("indoor", indoor_);

  // 初始化状态向量和协方差矩阵
  // 初始化状态向量和状态协方差矩阵
  x << 0, 0, 0, 0, 0;
  P << 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1;

  // 初始化过程噪声方差矩阵
  Q << 0.01, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0, 0, 0.01;

  // 初始化观测噪声方差矩阵和观测矩阵
  R1 << 1e-3, 0, 0, 1e-3;
  C1 << 1 / 2.0, -1 / 2.0,  // 计算线速度和角速度的观测矩阵
	  -1 / (2 * 0.2), -1 / (2 * 0.2), -1 / (2 * 0.2), 1 / (2 * 0.2), -1 / (2 * 0.2), -1 / (2 * 0.2), -1 / (2 * 0.2),
	  1 / (2 * 0.2);

  R2 << pow(3.14159 / 180.0, 2), 0, 0, pow(3.14159 / 180.0 * 10.0 / 360.0 * M_PI * x(3), 2),
	  pow(3.14159 / 180.0 * 10.0 / 360.0 * M_PI * x(3), 2), pow(3.14159 / 180.0 * 10.0 / 360.0 * M_PI * x(3), 2),
	  pow(3.14159 / 180.0 * x(4) / 360.0 * M_PI * x(3), 2), pow(3.14159 / 180.0 * x(4) / 360.0 * M_PI * x(3), 2),
	  pow(3.14159 / 180.0 * x(4) / 360.0 * M_PI * x(3), 2);
  C2 << MatrixXd::Identity(3, 3), MatrixXd::Zero(3, 2), MatrixXd::Zero(3, 1), MatrixXd::Zero(3, 3),
	  MatrixXd::Identity(3, 3), MatrixXd::Zero(3, 1), MatrixXd::Zero(1, 3), MatrixXd::Zero(1, 3),
	  MatrixXd::Identity(1, 1), MatrixXd::Zero(1, 3), MatrixXd::Zero(1, 3), MatrixXd::Zero(1, 1), MatrixXd::Zero(1, 3),
	  MatrixXd::Zero(1, 3), MatrixXd::Zero(1, 1);

  // Load parameters from YAML file
  std::string yaml_file;

  n.param<std::string>("file_path", yaml_file, "/home/justin/ZROS/ros/src/.config/encoder/encoder.yaml");

  YAML::Node config = YAML::LoadFile(yaml_file);

  std::string robot_id;
  n.param<std::string>("/wheel_odom/robot_id", robot_id,
					   "ZR1001");  // note-justin   如果无法获得main_socket/campus_id, 就默认读取map11150

  // Get wheel parameters for specific robot from YAML file
  wheel_radius = config[robot_id]["wheel_radius"].as<double>();
  encoder_ticks_per_rev = config[robot_id]["encoder_ticks_per_rev"].as<int>();
  wheelbase = config[robot_id]["wheelbase"].as<double>();
  wheelratio = config[robot_id]["wheelratio"].as<double>();

  ROS_INFO("wheel_odom.cpp:: wheel_radius = %f encoder_ticks_per_rev = %d wheelbase = %f wheelratio = %f", wheel_radius,
		   encoder_ticks_per_rev, wheelbase, wheelratio);

  // 发布里程计消息
  ros::Publisher odom_pub = n.advertise<nav_msgs::Odometry>("wheel_odom", 1000);
  ros::Publisher odom_pose_pub = n.advertise<geometry_msgs::PoseStamped>("wheel_odom_pose", 100);
  imu_twist_pub = n.advertise<geometry_msgs::TwistStamped>("imu_twist", 10);

  ros::Publisher ndt_pose_to_odom_pub = n.advertise<std_msgs::Int32>("ndt_pose_to_odom", 10);
  ros::Publisher encoder_twist_pub = n.advertise<geometry_msgs::TwistStamped>("encoder_twist", 10);
  ros::Publisher fitness_score_median_pub = n.advertise<std_msgs::Float32>("fitness_score_median", 10);
  ros::Publisher fitness_score_mean_pub = n.advertise<std_msgs::Float32>("fitness_score_mean", 10);
  ros::Publisher post_distance_pub = n.advertise<std_msgs::Float32>("store_pose_distance", 10);
  // 发布里程计信息
  odom_encoder_pub = n.advertise<nav_msgs::Odometry>("wheel_odom_encoder", 10);
  odom_fused_pub = n.advertise<nav_msgs::Odometry>("wheel_odom_fused", 50);

  ros::Publisher odom_yaw_pub = n.advertise<std_msgs::Float32>("odom_yaw", 10);
  ros::Publisher odom_twist_angular_pub = n.advertise<std_msgs::Float32>("twist_angular", 10);
  ros::Publisher odom_twist_linear_pub = n.advertise<std_msgs::Float32>("twist_linear", 10);
  // ros::Subscriber sub_start = n.subscribe("odom_start", 1000, StartCallback); comment by justin
  ros::Subscriber sub_start = n.subscribe("start_flag", 10, StartWheelOdomCallback);
  ros::Subscriber sub = n.subscribe("motor_info", 10, motorInfoCallback);
  ros::Subscriber sub_pose1 = n.subscribe("ekf_pose", 10, CalibratedPoseCallback);  // note-justin
																					// 之前使用natual_pose接下来使用ekf_pose,这样可以提升进电梯的精度
																					// ndt_pose ekf_pose

  ros::Subscriber current_pose_sub =
	  n.subscribe<geometry_msgs::PoseWithCovarianceStamped>("current_pose", 10, CurrentPoseCB);
  ros::Subscriber sub_imu_high_frequency = n.subscribe("imu_calibrated", 1000, ImuDataCB);
  ros::Subscriber sub_initalpose = n.subscribe("initialpose", 10, initialpose_callback);
  ros::Subscriber param_sub = n.subscribe("config/ndt", 10, param_callback);
  ros::Subscriber sub_modetrans = n.subscribe("use_wheel_odom", 10, ModeTransCallback);
  ros::Subscriber sub_reset = n.subscribe("reset_wheel_odom", 10, ResetCallback);
  ros::Subscriber emergency_switch_reset = n.subscribe("emergency_switch", 10, emergency_switch_Callback);
  ros::Subscriber sub_liftpose = n.subscribe("liftpose", 100, liftpose_callback);
  ros::Subscriber sub_ndt_stat = n.subscribe("ndt_stat", 10, NdtStatCallback);
  ros::Subscriber sub_close_yaw = n.subscribe("waypoint_yaw", 10, CloseYawCallback);
  ros::Subscriber sub_offset_imu_y = n.subscribe("offset_imu_y", 10, offset_imu_y_Callback);
  ros::Subscriber pose_param_sub = n.subscribe("config/ndt", 10, pose_param_callback);
  ros::Subscriber lift_flag_sub = n.subscribe("global_waypoint_change_flag", 10, waypointflagCallback);
  ros::Subscriber use_wheel_odom_sub = n.subscribe("/use_wheel_odom", 10, use_wheel_odom_callback);

  ros::Time current_time;
  ros::Time last_time;
  current_time = ros::Time::now();  // 当前时间
  last_time = ros::Time::now();		// 上次时间

  ros::Time current_fitness_time;
  ros::Time last_fitness_time;
  current_fitness_time = ros::Time::now();  // 当前时间
  last_fitness_time = ros::Time::now();		// 上次时间

  tf::TransformBroadcaster broadcaster;  // 位姿 广播
  ros::Rate loop_rate(20);				 // 频率

  const double degree = M_PI / 180;  // 度转 弧度

  static bool pose_distance_flag = true;

  // message declarations

  float current_wheel_accu_x = 0.0;
  float current_wheel_accu_y = 0.0;
  float current_wheel_accu_z = 0.0;
  encoder_iteration_num = TransNum;

  current_pose_x = ndt_pose_x;
  current_pose_y = ndt_pose_y;
  current_pose_z = ndt_pose_z;

  current_pose_roll = 0.0;
  current_pose_pitch = 0.0;
  current_pose_yaw = ndt_pose_yaw;

  while (ros::ok())
  {
	ros::spinOnce();

	current_time = ros::Time::now();  // 当前时间

	double dt;

	if (initialized_flag == false)  // note-justin  初次计算的时候两次时间差会很大
	{
	  dt = 0.02;
	  // ROS_ERROR("ndt_matching.cpp:: FIRST initialized_flag = %d dt = %f", initialized_flag, dt);
	}
	else
	{
	  dt = (current_time - last_time).toSec();  // 两次时间差
												// ROS_INFO("ndt_matching.cpp:: CONTINUE initialized_flag = %d dt = %f",
												// initialized_flag, dt);
	}
	double odom_delta_yaw = imu_angular_z * dt;  // note-justin  odom_angular_speed
	double odom_delta_x = 1.0 * odom_linear_speed * cos(odom_pose_yaw) *
						  dt;  //  note-justin imu_calibrated_yaw is wrong please use odom_angular_speed imu_angular_z
	double odom_delta_y = 1.0 * odom_linear_speed * sin(odom_pose_yaw) *
						  dt;  // note-justin imu_calibrated_yaw imu_angular_z
							   // ROS_ERROR("wheel_odom.cpp:: odom_delta_x = %f odom_delta_y = %f odom_delta_yaw = %f
							   // imu_angular_z = %f dt = %f odom_linear_speed = %f imu_calibrated_yaw = %f
							   // odom_angular_speed = %f", odom_delta_x, odom_delta_y, odom_delta_yaw, imu_angular_z,
							   // dt, odom_linear_speed, imu_calibrated_yaw, odom_angular_speed);

	// \vy y  /vx
	//  \  | /
	//   \ |/
	//    -------x-------
	//

	if (initialized_flag == true)
	{
	  if ((waypoint_change_flag == 5 || use_wheel_odom_lift_flag == true) && use_encoder_initialized == true)
	  {  // 切换至纯里程计
		PERIOD = ENCODER_TransNum;
		if (fitness_score < 3.0)
		{
		  current_pose_x = ndt_pose_x;
		  current_pose_y = ndt_pose_y;
		  current_pose_z = ndt_pose_z;

		  current_pose_roll = 0.0;
		  current_pose_pitch = 0.0;
		  current_pose_yaw = ndt_pose_yaw;

		  odom_pose_x = ndt_pose_x;
		  odom_pose_y = ndt_pose_y;
		  odom_pose_z = ndt_pose_z;

		  odom_pose_roll = 0.0;
		  odom_pose_pitch = 0.0;
		  odom_pose_yaw = ndt_pose_yaw;
		}
		use_encoder_initialized = false;
		anti_encoder_initialized = true;
		lift_change_flag = true;
		encoder_iteration_num = 0;
		// ROS_ERROR("waypoint_change_flag == 5 && use_encoder_initialized == true");
		ROS_ERROR("-----------------------use_wheel_odom_lift_flag = %d------------------------------------",
				  use_wheel_odom_lift_flag);

		use_wheel_odom_lift_flag = false;

		std_msgs::Int32 ndt_pose_to_odom_data;
		ndt_pose_to_odom_data.data = 1;
		ndt_pose_to_odom_pub.publish(ndt_pose_to_odom_data);  // note-justin  只根据模式来进行电梯模式的控制
		ROS_ERROR(
			"wheel_odom.cpp::use_encoder_initialized current_pose_x = %lf  current_pose_y = %lf current_pose_z = %lf "
			"ndt_pose_yaw = %lf",
			current_pose_x, current_pose_y, current_pose_z, ndt_pose_yaw);
		ROS_ERROR(
			"wheel_odom.cpp::use_encoder_initialized ndt_pose_yaw = %lf odom_pose_x = %lf odom_pose_y = %lf "
			"odom_pose_z = %lf",
			ndt_pose_yaw, odom_pose_x, odom_pose_y, odom_pose_z);
		ROS_ERROR("-----------------------------------------------------------");
	  }
	  else if (waypoint_change_flag != 5 && anti_encoder_initialized == true)
	  {  // 切换回激光雷达主导
		PERIOD = TransNum;
		// 建议去掉odom_pose_x y z 和current_pose是一样的
		current_pose_x = odom_pose_x;
		current_pose_y = odom_pose_y;
		current_pose_z = odom_pose_z;

		current_pose_roll = 0.0;
		current_pose_pitch = 0.0;
		current_pose_yaw = odom_pose_yaw;

		use_encoder_initialized = true;
		anti_encoder_initialized = false;
		lift_change_flag = true;
		encoder_iteration_num = 0;

		std_msgs::Int32 ndt_pose_to_odom_data;
		ndt_pose_to_odom_data.data = 0;
		ndt_pose_to_odom_pub.publish(ndt_pose_to_odom_data);
		// ROS_ERROR("waypoint_change_flag != 5 waypoint_change_flag = %d", waypoint_change_flag); //note-justin
		// 只根据模式来进行电梯模式的控制
		ROS_ERROR("-----------------------------------------------------------");
		ROS_ERROR(
			"wheel_odom.cpp::anti_encoder_initialized odom_pose_x = %f  odom_pose_y = %f odom_pose_yaw = %f "
			"current_pose_yaw = %f waypoint_change_flag = %d",
			odom_pose_x, odom_pose_y, odom_pose_yaw, current_pose_yaw, waypoint_change_flag);
		ROS_ERROR("-----------------------------------------------------------");
	  }
	  else
	  {
		// std_msgs::Int32 ndt_pose_to_odom_data;
		// ndt_pose_to_odom_data.data = 0;
		// ndt_pose_to_odom_pub.publish(ndt_pose_to_odom_data);
		//  ROS_ERROR("waypoint_change_flag != 5 ndt_pose_to_odom_data.data = 0 waypoint_change_flag = %d",
		//  waypoint_change_flag);
	  }

	  fitness_score_median =
		  (fitness_score0 + fitness_score1 + fitness_score2 + fitness_score3 + fitness_score4 + fitness_score5) /
		  float(6);
	  // ROS_ERROR("wheel_odom.cpp::NdtStatCallback fitness_score_median = %f fitness_score0 = %f fitness_score1 = %f
	  // fitness_score2 = %f fitness_score3 = %f, fitness_score4 = %f, fitness_score5 = %f", fitness_score_median,
	  // fitness_score0, fitness_score1, fitness_score2, fitness_score3, fitness_score4, fitness_score5);

	  offset_x = ndt_pose_x - previous_new_pose_x;
	  offset_y = ndt_pose_y - previous_new_pose_y;
	  offset_z = ndt_pose_z - previous_new_pose_z;
	  offset_roll = ndt_pose_roll - previous_new_pose_roll;
	  offset_pitch = ndt_pose_pitch - previous_new_pose_pitch;
	  offset_yaw = ndt_pose_yaw - previous_new_pose_yaw;
	  offset_yaw > M_PI ? offset_yaw -= 2 * M_PI :
						  offset_yaw < -M_PI ? offset_yaw += 2 * M_PI : offset_yaw = offset_yaw;

	  // ROS_ERROR("wheel_odom.cpp:: offset_x = %f offset_y = %f offset_z = %f", offset_x, offset_y, offset_z);

	  double pose_distance = sqrt(offset_x * offset_x + offset_y * offset_y);

	  previous_new_pose_x = ndt_pose_x;
	  previous_new_pose_y = ndt_pose_y;
	  previous_new_pose_z = ndt_pose_z;
	  previous_new_pose_yaw = ndt_pose_yaw;

	  if (pose_distance >= 10.0)
	  {
		pose_distance_flag = false;
		store_pose_distance = 2.0;
		// ROS_ERROR("wheel_odom.cpp::pose_distance_flag = %d pose_distance = %f", pose_distance_flag, pose_distance);
	  }
	  if (waypoint_change_flag == 4)
	  {
		fitness_above_threshold = fitness_above_threshold_indoor;
	  }
	  else
	  {
		fitness_above_threshold = fitness_above_threshold_outdoor;
	  }

	  if ((fitness_score_median < (fitness_score_mean + fitness_above_threshold)) &&
		  (pose_distance_flag == true))  // normal ndt_pose
	  {
		store_pose_x = ndt_pose_x;
		store_pose_y = ndt_pose_y;
		store_pose_z = ndt_pose_z;

		store_pose_roll = 0.0;
		store_pose_pitch = 0.0;
		store_pose_yaw = ndt_pose_yaw;
	  }

	  if (encoder_iteration_num == PERIOD)
	  {
		current_fitness_time = ros::Time::now();
		double time_elapsed = (current_fitness_time - last_fitness_time).toSec();

		// ROS_ERROR("wheel_odom.cpp::pose_distance_flag = %d pose_distance = %f", pose_distance_flag, pose_distance);

		if ((fitness_score_median < (fitness_score_mean + fitness_above_threshold)) &&
			(pose_distance_flag == true))  // normal ndt_pose  when pose_distance = 0.10129719 fitness_score_median =
										   // 0.1157822
		{
		  current_pose_x = ndt_pose_x;
		  current_pose_y = ndt_pose_y;
		  current_pose_z = ndt_pose_z;

		  current_pose_roll = 0.0;
		  current_pose_pitch = 0.0;
		  current_pose_yaw = ndt_pose_yaw;

		  normal_ndt_cnt++;
		  if (normal_ndt_cnt >= INT32_MAX)
			normal_ndt_cnt = 0;

		  ROS_INFO(
			  "wheel_odom.cpp -----------------------NORMAL NDT "
			  "POSE---------------%d------------------%f----------------%f-----",
			  normal_ndt_cnt, store_pose_distance, offset_yaw);
		  ROS_INFO("wheel_odom.cpp:: pose_distance = %f for normal ndt pose, pose_distance_flag =%d offset_yaw = %f",
				   pose_distance, pose_distance_flag, offset_yaw);
		  ROS_INFO(
			  "wheel_odom.cpp:  NORMAL NDT POSE current_pose_x =%f current_pose_y = %f current_pose_z = %f PERIOD: "
			  "PERIOD: %d",
			  current_pose_x, current_pose_y, current_pose_z, PERIOD);
		  ROS_INFO("wheel_odom.cpp:  REPOSITION: encoder_iteration_num: %d", encoder_iteration_num);
		  ROS_INFO(
			  "wheel_odom.cpp:  REPOSITION: NORMAL NDT POSE: %d fitness_score_median = %f fitness_score_mean = %f, "
			  "fitness_above_threshold = %f",
			  encoder_iteration_num, fitness_score_median, fitness_score_mean, fitness_above_threshold);
		  ROS_INFO("wheel_odom.cpp:: previous_pose_x = %f previous_pose_y = %f previous_pose_z = %f", previous_pose_x,
				   previous_pose_y, previous_pose_z);
		  ROS_INFO("wheel_odom.cpp:: current_pose_x = %f current_pose_y = %f current_pose_z = %f", current_pose_x,
				   current_pose_y, current_pose_z);
		  ROS_INFO("wheel_odom.cpp ---------------------------------------------------------------------");
		  ROS_INFO("wheel_odom.cpp ---------------------------------------------------------------------");
		}
		else if (true)  // if ((fitness_score_median < (fitness_score_mean + fitness_above_threshold))) //when
						// pose_distance too large  pose_distance = 10.129719  fitness_score_median = 0.031387
		{
		  current_pose_x = current_pose_x + odom_delta_x;
		  current_pose_y = current_pose_y + odom_delta_y;
		  current_pose_z = current_pose_z;
		  current_pose_roll = 0.0;
		  current_pose_pitch = 0.0;
		  current_pose_yaw = current_pose_yaw + odom_delta_yaw;
		  // current_pose_yaw = imu_calibrated_yaw; //note-justin
		  encoder_reposition_cnt++;
		  if (encoder_reposition_cnt >= INT32_MAX)
			encoder_reposition_cnt = 0;

		  // ROS_INFO("wheel_odom.cpp ---------------------ENCODER
		  // REPOSITION---------------%d------------------%f---------------%f----", encoder_reposition_cnt,
		  // store_pose_distance, offset_yaw); ROS_INFO("wheel_odom.cpp
		  // ---------------------------------------------------------------------"); ROS_INFO("wheel_odom.cpp::
		  // pose_distance = %f for encoder reposition pose_distance_flag = %d offset_yaw = %f", pose_distance,
		  // pose_distance_flag, offset_yaw); ROS_INFO("wheel_odom.cpp:  Odom Pose Used current_pose_x =%f
		  // current_pose_y = %f current_pose_z = %f PERIOD: PERIOD: %d", current_pose_x, current_pose_y,
		  // current_pose_z, PERIOD); ROS_INFO("wheel_odom.cpp:  REPOSITION PERIOD: PERIOD: %d odom_delta_x = %f
		  // odom_delta_y = %f", PERIOD, odom_delta_x, odom_delta_y); ROS_INFO("wheel_odom.cpp:  REPOSITION:
		  // encoder_iteration_num: %d fitness_score_median = %f fitness_score_mean = %f, fitness_above_threshold = %f",
		  // encoder_iteration_num, fitness_score_median, fitness_score_mean, fitness_above_threshold);
		  // ROS_INFO("wheel_odom.cpp:: previous_pose_x = %f previous_pose_y = %f previous_pose_z = %f",
		  // previous_pose_x, previous_pose_y, previous_pose_z); ROS_INFO("wheel_odom.cpp:: current_pose_x = %f
		  // current_pose_y = %f current_pose_z = %f", current_pose_x, current_pose_y, current_pose_z);
		  // ROS_INFO("wheel_odom.cpp ---------------------------------------------------------------------");
		  // ROS_INFO("wheel_odom.cpp ---------------------------------------------------------------------");
		}
		else if (false)  // when pose_distance = 0.10129719 fitness_score_median = 32.157822
		{
		  current_pose_x = store_pose_x;
		  current_pose_y = store_pose_y;
		  current_pose_z = store_pose_z;
		  current_pose_roll = 0.0;
		  current_pose_pitch = 0.0;
		  current_pose_yaw = store_pose_yaw;

		  // ROS_INFO("wheel_odom.cpp ---------------------STORE REPOSITION---------------%d---------%f----------",
		  // encoder_reposition_cnt, pose_distance); ROS_INFO("wheel_odom.cpp
		  // ---------------------------------------------------------------------"); ROS_INFO("wheel_odom.cpp::
		  // pose_distance = %f for encoder reposition pose_distance_flag = %d", pose_distance, pose_distance_flag);
		  // ROS_INFO("wheel_odom.cpp:  Odom Pose Used current_pose_x =%f current_pose_y = %f current_pose_z = %f
		  // PERIOD: PERIOD: %d", current_pose_x, current_pose_y, current_pose_z, PERIOD); ROS_INFO("wheel_odom.cpp:
		  // REPOSITION PERIOD: PERIOD: %d odom_delta_x = %f odom_delta_y = %f", PERIOD, odom_delta_x, odom_delta_y);
		  // ROS_INFO("wheel_odom.cpp:  REPOSITION: encoder_iteration_num: %d fitness_score_median = %f
		  // fitness_score_mean = %f, fitness_above_threshold = %f", encoder_iteration_num, fitness_score_median,
		  // fitness_score_mean, fitness_above_threshold); ROS_INFO("wheel_odom.cpp:: previous_pose_x = %f
		  // previous_pose_y = %f previous_pose_z = %f", previous_pose_x, previous_pose_y, previous_pose_z);
		  // ROS_INFO("wheel_odom.cpp:: current_pose_x = %f current_pose_y = %f current_pose_z = %f", current_pose_x,
		  // current_pose_y, current_pose_z); ROS_INFO("wheel_odom.cpp
		  // ---------------------------------------------------------------------"); ROS_INFO("wheel_odom.cpp
		  // ---------------------------------------------------------------------");
		}

		encoder_iteration_num = 0;
		store_pose_distance = 0.0;
		pose_distance_flag = true;
		// ROS_INFO("wheel_odom.cpp: REPOSITION PERIOD: PERIOD: %d", PERIOD);
		// ROS_INFO("wheel_odom.cpp: PERIOD: encoder_iteration_num: %d", encoder_iteration_num);
	  }
	  else if (encoder_iteration_num < PERIOD)
	  {
		current_pose_x = current_pose_x + odom_delta_x;
		current_pose_y = current_pose_y + odom_delta_y;
		current_pose_z = current_pose_z;
		current_pose_roll = 0.0;
		current_pose_pitch = 0.0;
		current_pose_yaw = current_pose_yaw + odom_delta_yaw;
		// current_pose_yaw = imu_calibrated_yaw;  // note-justin 也可用IMU yaw

		odom_pose_x = current_pose_x;
		odom_pose_y = current_pose_y;
		odom_pose_z = current_pose_z;
		odom_pose_roll = 0.0;
		odom_pose_pitch = 0.0;
		odom_pose_yaw = current_pose_yaw + odom_delta_yaw;
		// odom_pose_yaw = imu_calibrated_yaw; // note-justin 也可用IMU yaw

		// ROS_INFO("wheel_odom.cpp: PERIOD: PERIOD: %d", PERIOD);
		// ROS_INFO("wheel_odom.cpp: PERIOD: encoder_iteration_num: %d", encoder_iteration_num);
		// ROS_INFO("----------------------------------------------------");
		// ROS_INFO("wheel_odom.cpp:: odom_pose_x = %f odom_pose_y = %f odom_pose_z = %f odom_pose_yaw: = %f
		// current_pose_yaw = %f odom_delta_yaw = %f imu_angular_z = %f dt = %f", odom_pose_x, odom_pose_y, odom_pose_z,
		// odom_pose_yaw, current_pose_yaw, odom_delta_yaw, imu_angular_z, dt);
		encoder_iteration_num++;
		if (encoder_iteration_num >= INT32_MAX)
		  encoder_iteration_num = 0;
	  }
	  else if (encoder_iteration_num > PERIOD)
	  {
		encoder_iteration_num = 0;
	  }

	  // current_pose_yaw = imu_calibrated_yaw + ndt_pose_yaw;
	  // current_pose_yaw = ndt_pose_yaw;
	  // current_pose_yaw = ndt_pose_yaw;//close_info_yaw_score;

	  // ROS_INFO("wheel_odom.cpp:close_info_yaw_score: %f", close_info_yaw_score);

	  // ROS_INFO("wheel_odom.cpp:: odom_trans current_pose_yaw = %f odom_pose_yaw = %f", current_pose_yaw,
	  // odom_pose_yaw);

	  current_pose_yaw > 0 ? odom_pose_yaw = current_pose_yaw : odom_pose_yaw = 2 * M_PI + current_pose_yaw;

	  // odom_pose_yaw = imu_calibrated_yaw; // note-justin 使用IMU的数据
	  odom_pose_yaw = odom_pose_yaw;  // note-justin

	  geometry_msgs::TransformStamped odom_trans;  // 坐标变换消息
	  odom_trans.header.frame_id = "map";
	  odom_trans.child_frame_id = "odom_base_link";
	  // 更新左边变换消息，tf广播发布==================
	  odom_trans.header.stamp = current_time;			 // 当前时间
	  odom_trans.transform.translation.x = odom_pose_x;  // 位置
	  odom_trans.transform.translation.y = odom_pose_y;
	  odom_trans.transform.translation.z = odom_pose_z;
	  odom_trans.transform.rotation = tf::createQuaternionMsgFromYaw(odom_pose_yaw);  // 位姿
	  // publishing the odometry and the new tf
	  broadcaster.sendTransform(odom_trans);  // 发布坐标变换消息 =====

	  ROS_INFO("wheel_odom.cpp:: odom_trans odom_pose_x = %f, odom_pose_y = %f odom_pose_z = %f odom_pose_yaw = %f",
			   odom_pose_x, odom_pose_x, odom_pose_z, odom_pose_yaw);

	  // 更新 里程计消息
	  nav_msgs::Odometry odom;			 // 里程计消息
	  odom.header.stamp = current_time;  // 当前时间
	  odom.header.frame_id = "map";
	  odom.child_frame_id = "odom_base_link";

	  // 位置 position
	  odom.pose.pose.position.x = current_pose_x;
	  odom.pose.pose.position.y = current_pose_y;
	  odom.pose.pose.position.z = current_pose_z;

	  geometry_msgs::Quaternion odom_quat;  // 四元素位姿
	  odom_quat =
		  tf::createQuaternionMsgFromRollPitchYaw(odom_pose_roll, odom_pose_pitch, odom_pose_yaw);  // rpy转换到 四元素

	  odom.pose.pose.orientation = odom_quat;  // 位姿

	  ROS_INFO(
		  "wheel_odom.cpp:: odom.pose.pose.position.x: %f, odom.pose.pose.position.y: %f, odom.pose.pose.position.z: "
		  "%f, odom.pose.pose.orientation: %f,",
		  odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z, odom.pose.pose.orientation);

	  // 速度 velocity

	  odom.twist.twist.linear.x =
		  odom_linear_speed;			// odom_linear_speed * sin(imu_calibrated_yaw); //note-justin  线速度拆分
	  odom.twist.twist.linear.y = 0.0;  // odom_linear_speed * cos(imu_calibrated_yaw);
	  odom.twist.twist.linear.z = 0.0;
	  odom.twist.twist.angular.x = 0.0;  // 小速度
	  odom.twist.twist.angular.y = 0.0;
	  odom.twist.twist.angular.z = odom_angular_speed;  // note-justin odom_angular_speed; //note-justin

	  geometry_msgs::TwistStamped encoder_twist;	// 里程计消息
	  encoder_twist.header.stamp = current_time;	// 当前时间
	  encoder_twist.header.frame_id = "base_link";  // note-justin base_link  坐标系
	  encoder_twist.twist = odom.twist.twist;

	  last_time = current_time;					 // 迭代消息
	  last_fitness_time = current_fitness_time;  //

	  previous_pose_x = current_pose_x;
	  previous_pose_y = current_pose_y;
	  previous_pose_z = current_pose_z;

	  fitness_cnt++;
	  if (fitness_cnt >= INT32_MAX)
		fitness_cnt = 0;

	  fitness_cnt_divide = fitness_cnt % 6;
	  switch (fitness_cnt_divide)
	  {
		case 0:
		  fitness_score0 = fitness_score;
		  // ROS_ERROR("wheel_odom.cpp:: fitness_cnt_divide = %d, fitness_score0 = %f", fitness_cnt_divide,
		  // fitness_score0);
		  break;
		case 1:
		  fitness_score1 = fitness_score;
		  // ROS_ERROR("wheel_odom.cpp:: fitness_cnt_divide = %d, fitness_score1 = %f", fitness_cnt_divide,
		  // fitness_score1);
		  break;
		case 2:
		  fitness_score2 = fitness_score;
		  // ROS_ERROR("wheel_odom.cpp:: fitness_cnt_divide = %d, fitness_score2 = %f", fitness_cnt_divide,
		  // fitness_score2);
		  break;
		case 3:
		  fitness_score3 = fitness_score;
		  // ROS_ERROR("wheel_odom.cpp:: fitness_cnt_divide = %d, fitness_score3 = %f", fitness_cnt_divide,
		  // fitness_score3);
		  break;
		case 4:
		  fitness_score4 = fitness_score;
		  // ROS_ERROR("wheel_odom.cpp:: fitness_cnt_divide = %d, fitness_score4 = %f", fitness_cnt_divide,
		  // fitness_score4);
		  break;
		case 5:
		  fitness_score5 = fitness_score;
		  // ROS_ERROR("wheel_odom.cpp:: fitness_cnt_divide = %d, fitness_score5 = %f", fitness_cnt_divide,
		  // fitness_score5);
		  break;

		default:
		  break;
	  }

	  // if(start_flag == true)
	  //{
	  odom_pub.publish(odom);					 // 发布里程计消息====
	  encoder_twist_pub.publish(encoder_twist);  // 发布里程计消息====

	  geometry_msgs::PoseStamped odom_pose_msg;

	  odom_pose_msg.header.frame_id = "map";
	  odom_pose_msg.header.stamp = current_time;
	  odom_pose_msg.pose.position.x = odom.pose.pose.position.x;
	  odom_pose_msg.pose.position.y = odom.pose.pose.position.y;
	  odom_pose_msg.pose.position.z = odom.pose.pose.position.z;
	  odom_pose_msg.pose.orientation = odom.pose.pose.orientation;
	  ROS_INFO("wheel_odom.cpp:: odom_pose_msg.pose.position.x = %f odom_pose_msg.pose.position.y = %f",
			   odom_pose_msg.pose.position.x, odom_pose_msg.pose.position.y);
	  ROS_INFO("wheel_odom.cpp:: odom_pose_msg.pose.orientation.x = %f odom_pose_msg.pose.orientation.y = %f",
			   odom_pose_msg.pose.orientation.x, odom_pose_msg.pose.orientation.y);

	  odom_pose_pub.publish(odom_pose_msg);

	  std_msgs::Float32 fitness_score_median_msg;
	  fitness_score_median_msg.data = fitness_score_median;
	  fitness_score_median_pub.publish(fitness_score_median_msg);
	  ROS_ERROR(
		  "wheel_odom.cpp:: fitness_score_median publish: encoder_iteration_num = %d fitness_score_median = %f "
		  "fitness_score_mean = %f, fitness_above_threshold = %f",
		  encoder_iteration_num, fitness_score_median, fitness_score_mean, fitness_above_threshold);

	  std_msgs::Float32 store_pose_distance_msg;
	  store_pose_distance_msg.data = offset_yaw;
	  post_distance_pub.publish(store_pose_distance_msg);

	  std_msgs::Float32 fitness_score_mean_msg;
	  fitness_score_mean_msg.data = fitness_score_mean;
	  fitness_score_mean_pub.publish(fitness_score_mean_msg);

	  std_msgs::Float32 odom_yaw_pub_msg;
	  odom_yaw_pub_msg.data = odom_pose_yaw;
	  odom_yaw_pub.publish(odom_yaw_pub_msg);

	  std_msgs::Float32 odom_twist_angular_pub_msg;
	  odom_twist_angular_pub_msg.data = odom.twist.twist.angular.z;
	  odom_twist_angular_pub.publish(odom_twist_angular_pub_msg);

	  std_msgs::Float32 odom_twist_linear_pub_msg;
	  odom_twist_linear_pub_msg.data = odom.twist.twist.linear.x;
	  odom_twist_linear_pub.publish(odom_twist_linear_pub_msg);
	}
	// ros::spinOnce();

	loop_rate.sleep();
  }
  return 0;
}
