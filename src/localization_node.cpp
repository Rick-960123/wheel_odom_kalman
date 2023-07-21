#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sophus/so3.hpp>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <pthread.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <sophus/se3.hpp>
#include <fmt/core.h>
#include <thread>

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
static double fitness_score_threshold;
static double fitness, map_voxel_size, scan_voxel_size, sub_map_size;

static bool need_initial = true;
static bool use_ndt = true;

Eigen::Matrix4d T_odom_to_map, T_base_to_map, T_base_to_odom, T_base_to_lidar, T_imu_to_lidar;
pcl::PointCloud<pcl::PointXYZ>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZ>),
    sub_map(new pcl::PointCloud<pcl::PointXYZ>), cur_scan_in_odom(new pcl::PointCloud<pcl::PointXYZ>),
    cur_keypoints_in_odom(new pcl::PointCloud<pcl::PointXYZ>);

static pcl::CropBox<pcl::PointXYZ> crop_filter;
static pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
static pcl::VoxelGrid<pcl::PointXYZ> vox_filter;

static nav_msgs::Odometry cur_pcl_odom;
static geometry_msgs::PoseStamped current_pose;
static geometry_msgs::PoseWithCovarianceStamped ndt_cov_msg;

static ros::Time current_scan_time;
static ros::Time previous_scan_time;
static ros::Duration scan_duration;

static std::chrono::time_point<std::chrono::system_clock> matching_start, matching_end;

static ros::Publisher current_cov_pose_pub, sub_map_pub, current_pose_pub, fitness_pub, sound_pub, relocal_flag_pub,
    estimate_twist_pub, cur_scan_pub, cur_keypoints_pub;

pthread_mutex_t mutex;

struct matching_result
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_ptr;
  Eigen::Matrix4d trans;
  double fitness;
  matching_result() : pcl_ptr(new pcl::PointCloud<pcl::PointXYZ>()){};
};

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

void pub_pose()
{
  static tf::TransformBroadcaster br;
  Eigen::Quaterniond q;
  q = Eigen::Quaterniond(T_odom_to_map.block<3, 3>(0, 0));

  tf::Transform transform;
  tf::Quaternion qua;
  transform.setOrigin(tf::Vector3(T_odom_to_map(0, 3), T_odom_to_map(1, 3), T_odom_to_map(2, 3)));
  qua.setW(q.w());
  qua.setX(q.x());
  qua.setY(q.y());
  qua.setZ(q.z());
  transform.setRotation(qua);
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "camera_init"));

  T_base_to_map = T_odom_to_map * T_base_to_odom;
  Eigen::Quaterniond q_;
  q_ = Eigen::Quaterniond(T_base_to_map.block<3, 3>(0, 0));
  current_pose.header.frame_id = "map";
  current_pose.header.stamp = ros::Time::now();

  current_pose.pose.orientation.x = q_.x();
  current_pose.pose.orientation.y = q_.y();
  current_pose.pose.orientation.z = q_.z();
  current_pose.pose.orientation.w = q_.w();
  current_pose.pose.position.x = T_base_to_map(0, 3);
  current_pose.pose.position.y = T_base_to_map(1, 3);
  current_pose.pose.position.z = T_base_to_map(2, 3);
  current_pose_pub.publish(current_pose);
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

void map_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
  if (input->width > 0)
  {
    if (map_loaded)
    {
      return;
    }
    pcl::fromROSMsg(*input, *global_map);
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
    ndt.setMaximumIterations(50);
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
    icp.setMaximumIterations(50);
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
  ;
  sub_map_msg.header.frame_id = "map";
  sub_map_pub.publish(sub_map_msg);
}

bool is_loss()
{
  bool loss = false;
  if (cur_scan_in_odom->width == 0 || sub_map->width == 0)
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

void initialpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& input)
{
  if (map_loaded == 1)
  {
    Eigen::Quaterniond quaternion(input->pose.pose.orientation.w, input->pose.pose.orientation.x,
                                  input->pose.pose.orientation.y, input->pose.pose.orientation.z);
    Eigen::Vector3d translation(input->pose.pose.position.x, input->pose.pose.position.y, input->pose.pose.position.z);

    Eigen::Matrix4d T_base_to_map_estimation = Eigen::Matrix4d::Identity();
    T_base_to_map_estimation.block<3, 3>(0, 0) = quaternion.matrix();
    T_base_to_map_estimation.block<3, 1>(0, 3) = translation;

    Eigen::Matrix4d T_odom_to_map_estimation = T_base_to_map_estimation * se3_inverse(T_base_to_odom);
    std::cout << T_odom_to_map_estimation << std::endl;
    if (global_localization(T_odom_to_map_estimation))
    {
      need_initial = false;
      ROS_INFO("初始化定位成功,fitness:%lf", fitness);
    }
    else
    {
      reset_state();
      ROS_ERROR("初始化定位失败,fitness:%lf", fitness);
    };
  }
  else
  {
    ROS_ERROR("地图未加载!!!");
  }
}

void points_odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
  cur_pcl_odom = *msg;
  Eigen::Quaterniond quaternion(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                                msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);

  Eigen::Vector3d translation(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);

  T_base_to_odom.block<3, 3>(0, 0) = quaternion.matrix();
  T_base_to_odom.block<3, 1>(0, 3) = translation;
}

void gnss_callback(const geometry_msgs::PoseStamped::ConstPtr& input)
{
}

void wheel_odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
}

void points_callback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  pcl::PointCloud<pcl::PointXYZ> pcl;
  pcl::fromROSMsg(*msg, *cur_scan_in_odom);

  cur_keypoints_in_odom = detect_iss_keypoints(cur_scan_in_odom);

  sensor_msgs::PointCloud2 scan_msg, keyspoints_msg;
  pcl::toROSMsg(*cur_scan_in_odom, scan_msg);
  pcl::toROSMsg(*cur_keypoints_in_odom, keyspoints_msg);
  scan_msg.header = msg->header;
  scan_msg.header.frame_id = "camera_init";
  scan_msg.header.stamp = ros::Time::now();
  keyspoints_msg.header = scan_msg.header;
  cur_scan_pub.publish(scan_msg);
  cur_keypoints_pub.publish(keyspoints_msg);
}

static std::string matrixToString(const Eigen::MatrixXd& mat)
{
  std::stringstream ss;
  ss << mat;
  return ss.str();
}

void thread_fuc()
{
  ROS_INFO("子线程已开启");
  ros::Rate rate(20);
  while (ros::ok())
  {
    if (map_loaded == 1 && !need_initial)
    {
      std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      global_localization(T_odom_to_map);
      is_loss();
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::chrono::duration<double, std::milli> duration_ms = end - start;
      std::cout << "匹配消耗时间（毫秒）: " << duration_ms.count() << "ms" << std::endl;
    }
    pub_pose();
    rate.sleep();
  }
}

int main(int argc, char** argv)
{
  setlocale(LC_CTYPE, "zh_CN.utf8");
  ros::init(argc, argv, "pointcloud_odom");
  pthread_mutex_init(&mutex, NULL);

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
  cur_scan_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/cur_scan_in_map", 1);
  cur_keypoints_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/cur_keypoints_in_map", 1);
  current_pose_pub = private_nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 10);
  current_cov_pose_pub = private_nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/current_cov_pose", 10);

  estimate_twist_pub = private_nh.advertise<geometry_msgs::TwistStamped>("/estimate_twist", 1000);
  fitness_pub = private_nh.advertise<std_msgs::Float32>("/fitness_score", 100);
  sound_pub = private_nh.advertise<std_msgs::String>("/sound_player", 10, true);

  // Subscribers
  ros::Subscriber initialpose_sub = private_nh.subscribe("/initialpose", 1000, initialpose_callback);
  ros::Subscriber map_sub = private_nh.subscribe("/points_map", 10, map_callback);
  ros::Subscriber points_sub = private_nh.subscribe("/cloud_registered", 10, points_callback);

  ros::Subscriber gnss_sub = private_nh.subscribe("/gnss_pose", 10, gnss_callback);
  ros::Subscriber pcl_odom_sub = private_nh.subscribe("/Odometry", 1000, points_odom_callback);
  ros::Subscriber wheel_odom_sub = private_nh.subscribe("/wheel_odom", 1000, wheel_odom_callback);

  reset_state();

  std::thread thread1(thread_fuc);
  ros::spin();
  thread1.join();
  return 0;
}
