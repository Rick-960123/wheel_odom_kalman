#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sophus/so3.hpp>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <sophus/se3.hpp>
#include <fmt/core.h>
#include <thread>
#include <mutex>
#include <deque>

#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#define PI (3.1415926)
static int map_loaded = 0;
static int loss_cnt = 0;
static int waypoint_change_flag;
static double fitness_score_threshold;
static double fitness, map_voxel_size, scan_voxel_size, sub_map_size;

static bool need_initial = true;
static bool use_ndt = true;
static bool use_wheel_odom_flag = false;
static bool exit_initial_thread = false;

Eigen::Matrix4d T_odom_to_map, T_wheel_odom_to_map, T_base_to_map, T_base_to_odom, T_base_to_wheel_odom,
    T_base_to_lidar, T_imu_to_lidar;
pcl::PointCloud<pcl::PointXYZ>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZ>),
    sub_map(new pcl::PointCloud<pcl::PointXYZ>), keyframes_pcl(new pcl::PointCloud<pcl::PointXYZ>),
    cur_scan_in_odom(new pcl::PointCloud<pcl::PointXYZ>), cur_keypoints_in_odom(new pcl::PointCloud<pcl::PointXYZ>);

static pcl::CropBox<pcl::PointXYZ> crop_filter;
static pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
static pcl::VoxelGrid<pcl::PointXYZ> vox_filter;

static geometry_msgs::PoseStamped current_pose;
static geometry_msgs::PoseWithCovarianceStamped ndt_cov_msg;

static ros::Time cur_scan_stamp, points_odom_stamp, wheel_odom_stamp;
static ros::Time pre_scan_time;
static ros::Duration scan_duration;

static ros::Publisher current_cov_pose_pub, sub_map_pub, current_pose_pub, fitness_pub, reset_wheel_odom_pub,
    reset_odom_pub, sound_pub, relocal_flag_pub, estimate_twist_pub, cur_scan_pub, cur_keypoints_pub, keyframes_pcl_pub;

struct keyframe
{
  keyframe() : feature_points_ptr(new pcl::PointCloud<pcl::PointXYZ>()){};
  ros::Time timestamp;
  pcl::PointCloud<pcl::PointXYZ>::Ptr feature_points_ptr;
  Eigen::Matrix4d pose_in_odom;
  Eigen::Matrix4d pose_in_map;

  double diff_stamp(const keyframe& other)
  {
    return abs((timestamp - other.timestamp).toSec());
  }
  double diff_pose(const keyframe& other)
  {
    return pow((pow(pose_in_odom(0, 3) - other.pose_in_odom(0, 3), 2) +
                pow(pose_in_odom(1, 3) - other.pose_in_odom(1, 3), 2) +
                pow(pose_in_odom(2, 3) - other.pose_in_odom(2, 3), 2)),
               0.5);
  }
  double diff_yaw(const keyframe& other)
  {
    Eigen::Vector3d eulerAngle = pose_in_odom.block<3, 3>(0, 0).eulerAngles(2, 1, 0);
    Eigen::Vector3d other_eulerAngle = other.pose_in_odom.block<3, 3>(0, 0).eulerAngles(2, 1, 0);
    return abs(eulerAngle(2) - other_eulerAngle(2));
  }
};

struct matching_result
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_ptr;
  Eigen::Matrix4d trans;
  double fitness;
  matching_result() : pcl_ptr(new pcl::PointCloud<pcl::PointXYZ>()){};
};

static std::deque<keyframe> keyframes;
static std::mutex mutex;
static std::thread main_thread, initial_thread;
static std::chrono::time_point<std::chrono::system_clock> matching_start, matching_end;

pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_down_sample(pcl::PointCloud<pcl::PointXYZ>::Ptr& pcd,
                                                      const double& voxel_size)
{
  vox_filter.setInputCloud(pcd);
  vox_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_tmp(new pcl::PointCloud<pcl::PointXYZ>);
  vox_filter.filter(*pcd_tmp);
  return pcd_tmp;
}

Eigen::Matrix4d se3_inverse(const Eigen::Matrix4d& T)
{
  Sophus::SE3d T_tmp(T);
  Sophus::SE3d T_inverse = T_tmp.inverse();
  return T_inverse.matrix();
}

void pub_topic()
{
  auto cur_time = ros::Time::now();
  static tf::TransformBroadcaster br;
  Eigen::Quaterniond q;
  q = Eigen::Quaterniond(T_odom_to_map.block<3, 3>(0, 0));
  q.normalize();

  tf::Transform transform;
  tf::Quaternion qua;
  transform.setOrigin(tf::Vector3(T_odom_to_map(0, 3), T_odom_to_map(1, 3), T_odom_to_map(2, 3)));
  qua.setW(q.w());
  qua.setX(q.x());
  qua.setY(q.y());
  qua.setZ(q.z());
  transform.setRotation(qua);
  br.sendTransform(tf::StampedTransform(transform, cur_time, "map", "camera_init"));

  if (use_wheel_odom_flag)
  {
    T_base_to_map = T_wheel_odom_to_map * T_base_to_wheel_odom;
  }
  else
  {
    T_base_to_map = T_odom_to_map * T_base_to_odom;
  }

  Eigen::Quaterniond q_;
  q_ = Eigen::Quaterniond(T_base_to_map.block<3, 3>(0, 0));
  q_.normalize();
  current_pose.header.frame_id = "map";
  current_pose.header.stamp = cur_time;

  current_pose.pose.orientation.x = q_.x();
  current_pose.pose.orientation.y = q_.y();
  current_pose.pose.orientation.z = q_.z();
  current_pose.pose.orientation.w = q_.w();
  current_pose.pose.position.x = T_base_to_map(0, 3);
  current_pose.pose.position.y = T_base_to_map(1, 3);
  current_pose.pose.position.z = T_base_to_map(2, 3);
  current_pose_pub.publish(current_pose);

  std_msgs::Float32 fitness_msg;
  fitness_msg.data = fitness;
  fitness_pub.publish(fitness_msg);

  geometry_msgs::TwistStamped twist_msg;
  twist_msg.header.stamp = cur_time;
  twist_msg.header.frame_id = "base_link";
  twist_msg.twist.linear.x = 0.0;
  twist_msg.twist.angular.z = 0.0;
  estimate_twist_pub.publish(twist_msg);

  sensor_msgs::PointCloud2 scan_msg, keyspoints_msg, keyframes_pcl_msg;

  pcl::toROSMsg(*cur_scan_in_odom, scan_msg);
  scan_msg.header.frame_id = "camera_init";
  scan_msg.header.stamp = cur_time;
  cur_scan_pub.publish(scan_msg);

  pcl::toROSMsg(*cur_keypoints_in_odom, keyspoints_msg);
  keyspoints_msg.header = scan_msg.header;
  cur_keypoints_pub.publish(keyspoints_msg);

  pcl::toROSMsg(*keyframes_pcl, keyframes_pcl_msg);
  keyframes_pcl_msg.header = scan_msg.header;
  keyframes_pcl_pub.publish(keyframes_pcl_msg);
}

void reset_state()
{
  need_initial = true;
  loss_cnt = 0;
  T_odom_to_map = Eigen::Matrix4d::Identity();
  T_base_to_odom = Eigen::Matrix4d::Identity();
  T_base_to_map = Eigen::Matrix4d::Identity();
  fitness_score_threshold = use_ndt ? 0.6 : 0.4;
  ROS_INFO("已初始化状态变量");
}

void map_callback(const sensor_msgs::PointCloud2::ConstPtr& msg_ptr)
{
  if (msg_ptr->width > 0)
  {
    if (map_loaded)
    {
      return;
    }
    pcl::fromROSMsg(*msg_ptr, *global_map);
    global_map = voxel_down_sample(global_map, map_voxel_size);
    crop_filter.setInputCloud(global_map);
    map_loaded = 1;
    ROS_INFO("地图加载成功!");
  }
  else
  {
    ROS_ERROR("无效的地图!!!");
  }
}

matching_result registration_at_scale(pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_scan,
                                      pcl::PointCloud<pcl::PointXYZ>::Ptr pc_map, Eigen::Matrix4d& initial_guess,
                                      int scale)
{
  matching_result res;
  if (use_ndt)
  {
    ndt.setResolution(1.0 * scale);
    ndt.setMaximumIterations(30);
    ndt.setInputSource(voxel_down_sample(pc_scan, scan_voxel_size * scale));
    ndt.setInputTarget(voxel_down_sample(pc_map, map_voxel_size * scale));
    ndt.align(*res.pcl_ptr, initial_guess.cast<float>());
    res.fitness = ndt.getFitnessScore();
    res.trans = ndt.getFinalTransformation().cast<double>();
  }
  else
  {
    icp.setInputSource(voxel_down_sample(pc_scan, scan_voxel_size * scale));
    icp.setInputTarget(voxel_down_sample(pc_map, map_voxel_size * scale));
    icp.setMaximumIterations(30);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(0.001);
    icp.align(*res.pcl_ptr, initial_guess.cast<float>());
    res.fitness = icp.getFitnessScore();
    res.trans = icp.getFinalTransformation().cast<double>();
  }
  return res;
}

void crop_global_map_in_FOV(Eigen::Matrix4d& pose_estimation)
{
  Eigen::Matrix4d T_base_to_map_estimation = pose_estimation * T_base_to_odom;
  Eigen::Vector4d start, end;
  start = T_base_to_map_estimation.block<4, 1>(0, 3) -
          Eigen::Vector4d(sub_map_size / 2.0, sub_map_size / 2.0, sub_map_size / 2.0, 0.0);
  end = T_base_to_map_estimation.block<4, 1>(0, 3) +
        Eigen::Vector4d(sub_map_size / 2.0, sub_map_size / 2.0, sub_map_size / 2.0, 0.0);

  crop_filter.setMin(start.cast<float>());
  crop_filter.setMax(end.cast<float>());
  crop_filter.filter(*sub_map);

  sensor_msgs::PointCloud2 sub_map_msg;
  pcl::toROSMsg(*sub_map, sub_map_msg);
  sub_map_msg.header.stamp = ros::Time::now();
  sub_map_msg.header.frame_id = "map";
  sub_map_pub.publish(sub_map_msg);
}

bool is_loss()
{
  bool loss = false;
  if (cur_keypoints_in_odom->width == 0 || sub_map->width == 0)
  {
    loss = true;
    ROS_ERROR("无效点云或地图!!!");
  }
  if (loss_cnt > 50)
  {
    loss = true;
  }
  if (loss)
  {
    std_msgs::String sound_content;
    sound_content.data = "lost";
    sound_pub.publish(sound_content);
    reset_state();
    ROS_ERROR("定位丢失");
  }
  return loss;
}

bool global_localization(Eigen::Matrix4d& pose_estimation)
{
  crop_global_map_in_FOV(pose_estimation);
  matching_result res = registration_at_scale(cur_keypoints_in_odom, sub_map, pose_estimation, 2);
  res = registration_at_scale(cur_keypoints_in_odom, sub_map, res.trans, 1);
  fitness = res.fitness;
  if (fitness < fitness_score_threshold)
  {
    T_odom_to_map = res.trans;
    loss_cnt = 0;
    return true;
  }
  else
  {
    loss_cnt += 1;
    ROS_INFO(fmt::format("点云匹配失败, loss_cnt: {}, fitness: {}", loss_cnt, fitness).c_str());
    return false;
  }
}
pcl::PointCloud<pcl::PointXYZ>::Ptr detect_iss_keypoints(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
  pcl::search::KdTree<pcl::PointXYZ>::Ptr iss_tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;
  pcl::PointCloud<pcl::PointXYZ>::Ptr iss_keypoints(new pcl::PointCloud<pcl::PointXYZ>());

  iss_detector.setInputCloud(cloud);
  iss_detector.setSearchMethod(iss_tree);
  iss_detector.setSalientRadius(1.0f);
  iss_detector.setNonMaxRadius(1.0f);
  iss_detector.setThreshold21(0.975);
  iss_detector.setThreshold32(0.975);
  iss_detector.setMinNeighbors(5);
  iss_detector.setNumberOfThreads(4);
  iss_detector.compute(*iss_keypoints);
  return iss_keypoints;
}

void reset_odom(const std::string& odom_type)
{
  std_msgs::Bool reset_msg;
  reset_msg.data = true;
  if (odom_type == "wheel_odom")
  {
    reset_wheel_odom_pub.publish(reset_msg);
  }
  else if (odom_type == "points_odom")
  {
    reset_odom_pub.publish(reset_msg);
  }
  else if (odom_type == "all")
  {
    reset_odom_pub.publish(reset_msg);
    reset_wheel_odom_pub.publish(reset_msg);
  }
}

void init_function(const Eigen::Matrix4d& T_base_to_map_estimation, const std::string& init_odom)
{
  ROS_INFO("初始化子线程已开启");
  auto cur_stamp = ros::Time::now();

  while (ros::ok())
  {
    if (init_odom == "points_odom" && points_odom_stamp > cur_stamp && cur_scan_stamp > cur_stamp)
    {
      ROS_INFO("已收到points_odom重置后最新帧");
      break;
    }
    else if (init_odom == "wheel_odom" && wheel_odom_stamp > cur_stamp)
    {
      ROS_INFO("已收到wheel_odom重置后最新帧");
      break;
    }
    else if (init_odom == "all" && wheel_odom_stamp > cur_stamp && points_odom_stamp > cur_stamp &&
             cur_scan_stamp > cur_stamp)
    {
      ROS_INFO("已收到all_odom重置后最新帧");
      break;
    }
    if (exit_initial_thread)
    {
      ROS_INFO("初始化子线程已退出");
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  Eigen::Matrix4d T_odom_to_map_estimation = T_base_to_map_estimation * se3_inverse(T_base_to_odom);

  std::unique_lock<std::mutex> lock(mutex);
  if (global_localization(T_odom_to_map_estimation))
  {
    need_initial = false;
    T_wheel_odom_to_map = T_odom_to_map;
    ROS_INFO("初始化定位成功,fitness:%lf", fitness);
  }
  else
  {
    reset_state();
    ROS_ERROR("初始化定位失败,fitness:%lf", fitness);
  };
  lock.unlock();
  ROS_INFO("初始化子线程已退出");
}

void add_keyframe()
{
  keyframe kf;
  kf.timestamp = cur_scan_stamp;
  kf.pose_in_odom = T_base_to_odom;
  kf.feature_points_ptr = cur_keypoints_in_odom;

  if (keyframes.size() < 10)
  {
    keyframes.push_back(kf);
  }
  else
  {
    auto pre_keyframe = keyframes.back();
    if (kf.diff_yaw(pre_keyframe) > 0.8)
    {
      keyframes.pop_front();
      keyframes.push_back(kf);
    }
    else if (kf.diff_pose(pre_keyframe) > 1)
    {
      keyframes.pop_front();
      keyframes.push_back(kf);
    }
    else if (kf.diff_stamp(pre_keyframe) > 30)
    {
      keyframes.pop_front();
      keyframes.push_back(kf);
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr temp_map(new pcl::PointCloud<pcl::PointXYZ>);
  for (auto& kf : keyframes)
  {
    Eigen::Matrix4d trans = T_base_to_odom * se3_inverse(kf.pose_in_odom);
    pcl::PointCloud<pcl::PointXYZ> trans_cloud;
    pcl::transformPointCloud(*kf.feature_points_ptr, trans_cloud, trans);
    *temp_map += trans_cloud;
  }
  keyframes_pcl = temp_map;
}

void initialpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg_ptr)
{
  if (map_loaded == 1)
  {
    reset_odom("all");

    Eigen::Quaterniond quaternion(msg_ptr->pose.pose.orientation.w, msg_ptr->pose.pose.orientation.x,
                                  msg_ptr->pose.pose.orientation.y, msg_ptr->pose.pose.orientation.z);
    Eigen::Vector3d translation(msg_ptr->pose.pose.position.x, msg_ptr->pose.pose.position.y,
                                msg_ptr->pose.pose.position.z);

    Eigen::Matrix4d T_base_to_map_estimation = Eigen::Matrix4d::Identity();
    T_base_to_map_estimation.block<3, 3>(0, 0) = quaternion.matrix();
    T_base_to_map_estimation.block<3, 1>(0, 3) = translation;

    if (initial_thread.joinable())
    {
      exit_initial_thread = true;
      initial_thread.join();
      exit_initial_thread = false;
    }
    initial_thread = std::thread(init_function, T_base_to_map_estimation, "points_odom");
  }
  else
  {
    ROS_ERROR("地图未加载!!!");
  }
}

void points_odom_callback(const nav_msgs::Odometry::ConstPtr& msg_ptr)
{
  Eigen::Quaterniond quaternion(msg_ptr->pose.pose.orientation.w, msg_ptr->pose.pose.orientation.x,
                                msg_ptr->pose.pose.orientation.y, msg_ptr->pose.pose.orientation.z);
  Eigen::Vector3d translation(msg_ptr->pose.pose.position.x, msg_ptr->pose.pose.position.y,
                              msg_ptr->pose.pose.position.z);

  std::unique_lock<std::mutex> lock(mutex);
  points_odom_stamp = ros::Time::now();
  T_base_to_odom.block<3, 3>(0, 0) = quaternion.matrix();
  T_base_to_odom.block<3, 1>(0, 3) = translation;
  lock.unlock();
}

void wheel_odom_callback(const nav_msgs::Odometry::ConstPtr& msg_ptr)
{
  Eigen::Quaterniond quaternion(msg_ptr->pose.pose.orientation.w, msg_ptr->pose.pose.orientation.x,
                                msg_ptr->pose.pose.orientation.y, msg_ptr->pose.pose.orientation.z);

  Eigen::Vector3d translation(msg_ptr->pose.pose.position.x, msg_ptr->pose.pose.position.y,
                              msg_ptr->pose.pose.position.z);

  std::unique_lock<std::mutex> lock(mutex);
  wheel_odom_stamp = ros::Time::now();
  T_base_to_wheel_odom.block<3, 3>(0, 0) = quaternion.matrix();
  T_base_to_wheel_odom.block<3, 1>(0, 3) = translation;
  lock.unlock();
}

void points_callback(const sensor_msgs::PointCloud2::ConstPtr& msg_ptr)
{
  pcl::fromROSMsg(*msg_ptr, *cur_scan_in_odom);

  std::unique_lock<std::mutex> lock(mutex);
  cur_scan_stamp = ros::Time::now();
  cur_keypoints_in_odom = detect_iss_keypoints(cur_scan_in_odom);
  lock.unlock();
}

void gnss_callback(const geometry_msgs::PoseStamped::ConstPtr& msg_ptr)
{
}

void waypointchangeflagCallback(const std_msgs::Int32::ConstPtr& msg_ptr)
{
  waypoint_change_flag = msg_ptr->data;
}

void use_wheel_odom_callback(const std_msgs::BoolConstPtr& msg_ptr)
{
  use_wheel_odom_flag = msg_ptr->data;
  if (use_wheel_odom_flag)
  {
    reset_odom("wheel_odom");
    T_wheel_odom_to_map = T_base_to_map;
  }
  else
  {
    reset_odom("points_odom");
    if (initial_thread.joinable())
    {
      exit_initial_thread = true;
      initial_thread.join();
      exit_initial_thread = false;
    }
    initial_thread = std::thread(init_function, T_base_to_map, "points_odom");
  }
}

void thread_fuc()
{
  ROS_INFO("定位子线程已开启");
  ros::Rate rate(20);
  while (ros::ok())
  {
    // std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    double diff = (cur_scan_stamp - points_odom_stamp).toSec();

    if (0.05 > diff && diff > -0.05)
    {
      std::unique_lock<std::mutex> lock(mutex);
      add_keyframe();
      if (map_loaded == 1 && !need_initial)
      {
        global_localization(T_odom_to_map);
        is_loss();
      }
      pub_topic();
      lock.unlock();
    }
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::chrono::duration<double, std::milli> duration_ms = end - start;
    // std::cout << "匹配消耗时间（毫秒）: " << duration_ms.count() << "ms" << std::endl;

    rate.sleep();
  }
  ROS_INFO("定位子线程已退出");
}

int main(int argc, char** argv)
{
  setlocale(LC_CTYPE, "zh_CN.utf8");
  ros::init(argc, argv, "pointcloud_odom");

  ros::NodeHandle private_nh("~");

  std::string robot_id;
  private_nh.param<std::string>("/robot_id", robot_id, "ZR1001");

  std::string prefix = "/" + robot_id + "/lidar";
  std::vector<double> T_base_to_lidar_vector, T_imu_to_lidar_vector;

  private_nh.param<std::vector<double>>(prefix + "/T_base_to_lidar", T_base_to_lidar_vector, std::vector<double>());
  private_nh.param<std::vector<double>>(prefix + "/T_imu_to_lidar", T_imu_to_lidar_vector, std::vector<double>());

  T_base_to_lidar = Eigen::Map<const Eigen::Matrix<double, 4, 4>>(T_base_to_lidar_vector.data());
  T_imu_to_lidar = Eigen::Map<const Eigen::Matrix<double, 4, 4>>(T_imu_to_lidar_vector.data());

  private_nh.param<double>("/map_voxel_size", map_voxel_size, 0.3);
  private_nh.param<double>("/sub_map_size", sub_map_size, 100);
  private_nh.param<double>("/scan_voxel_size", scan_voxel_size, 0.3);
  private_nh.param<bool>("/use_ndt", use_ndt, false);

  // Publishers
  sub_map_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/sub_map", 1);
  keyframes_pcl_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/keyframes_pcl", 1);
  cur_scan_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/cur_scan_in_map", 1);
  cur_keypoints_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/cur_keypoints_in_map", 1);
  current_pose_pub = private_nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 10);
  current_cov_pose_pub = private_nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/current_cov_pose", 10);

  estimate_twist_pub = private_nh.advertise<geometry_msgs::TwistStamped>("/estimate_twist", 1000);
  fitness_pub = private_nh.advertise<std_msgs::Float32>("/fitness_score", 100);
  sound_pub = private_nh.advertise<std_msgs::String>("/sound_player", 10);
  reset_odom_pub = private_nh.advertise<std_msgs::Bool>("/reset_odom", 10);
  reset_wheel_odom_pub = private_nh.advertise<std_msgs::Bool>("/reset_wheel_odom", 10);

  // Subscribers
  ros::Subscriber initialpose_sub = private_nh.subscribe("/initialpose", 1000, initialpose_callback);
  ros::Subscriber map_sub = private_nh.subscribe("/points_map", 10, map_callback);
  ros::Subscriber points_sub = private_nh.subscribe("/cloud_registered", 10, points_callback);

  ros::Subscriber gnss_sub = private_nh.subscribe("/gnss_pose", 10, gnss_callback);
  ros::Subscriber pcl_odom_sub = private_nh.subscribe("/Odometry", 1000, points_odom_callback);
  ros::Subscriber wheel_odom_sub = private_nh.subscribe("/wheel_odom", 1000, wheel_odom_callback);

  ros::Subscriber use_wheel_odom_sub = private_nh.subscribe("/use_wheel_odom", 10, use_wheel_odom_callback);
  ros::Subscriber waypoint_change_flag_subscriber =
      private_nh.subscribe("global_waypoint_change_flag", 10, waypointchangeflagCallback);

  reset_state();
  main_thread = std::thread(thread_fuc);
  ros::spin();
  main_thread.join();
  return 0;
}
