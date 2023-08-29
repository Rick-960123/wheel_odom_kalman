#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <deque>
#include <unistd.h>
#include <stdlib.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <fmt/core.h>

#include <ros/ros.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

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
#include <XmlRpcValue.h>

static pcl::CropBox<pcl::PointXYZI> crop_filter;
static pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
static pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
static pcl::VoxelGrid<pcl::PointXYZI> vox_filter;

typedef std::vector<double> std_vec1d;

class ZRParameters
{
public:
  std::string robot_id;
  XmlRpc::XmlRpcValue T_lidar_to_base_vector, T_imu_to_base_vector, T_enc_to_base_vector;
  Eigen::Matrix4d T_lidar_to_base, T_imu_to_base, T_enc_to_base;

  Eigen::MatrixXd vector_to_eigen(std_vec1d& vec)
  {
    int rows = vec.size();
    Eigen::VectorXd mat(rows);
    for (int i = 0; i < rows; ++i)
    {
      mat(i) = vec[i];
    }
    return mat;
  }

  Eigen::MatrixXd vector_to_eigen(XmlRpc::XmlRpcValue& vec)
  {
    int rows = vec.size();
    int cols = vec[0].size();
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
      for (int j = 0; j < cols; ++j)
      {
        mat(i, j) = vec[i][j];
      }
    }
    return mat;
  }

  void get_paramters(ros::NodeHandle& nh)
  {
    nh.param<std::string>("/robot_id", robot_id, "ZR1001");
    nh.getParam("/lidar/T_lidar_to_base", T_lidar_to_base_vector);
    nh.getParam("/imu/T_imu_to_base", T_imu_to_base_vector);
    nh.getParam("/T_enc_to_base", T_enc_to_base_vector);

    T_lidar_to_base = vector_to_eigen(T_lidar_to_base_vector);
    T_imu_to_base = vector_to_eigen(T_imu_to_base_vector);
    T_enc_to_base = vector_to_eigen(T_enc_to_base_vector);
  }
  ZRParameters()
  {
  }
  ~ZRParameters()
  {
  }
};

class timer
{
public:
  timer()
  {
    start = std::chrono::steady_clock::now();
  }
  ~timer()
  {
  }
  void print(std::string label)
  {
    end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end - start;
    std::cout << label + ":消耗时间" << duration_ms.count() << "ms" << std::endl;
  }

private:
  std::chrono::steady_clock::time_point start, end;
};

pcl::PointCloud<pcl::PointXYZI>::Ptr voxel_down_sample(pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd,
                                                       const double& voxel_size)
{
  vox_filter.setInputCloud(pcd);
  vox_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_tmp(new pcl::PointCloud<pcl::PointXYZI>);
  vox_filter.filter(*pcd_tmp);
  return pcd_tmp;
}

Eigen::Matrix4d se3_inverse(const Eigen::Matrix4d& T)
{
  Sophus::SE3d T_tmp(T);
  Sophus::SE3d T_inverse = T_tmp.inverse();
  return T_inverse.matrix();
}

tf::Transform eigenMatrix4dToTfTransform(const Eigen::Matrix4d& eigen_mat)
{
  Eigen::Matrix3d eigen_rot = eigen_mat.block<3, 3>(0, 0);
  Eigen::Vector3d eigen_trans = eigen_mat.block<3, 1>(0, 3);
  Eigen::Quaterniond eigen_quat(eigen_rot);
  tf::Transform tf_transform;
  tf_transform.setOrigin(tf::Vector3(eigen_trans(0), eigen_trans(1), eigen_trans(2)));
  tf_transform.setRotation(tf::Quaternion(eigen_quat.x(), eigen_quat.y(), eigen_quat.z(), eigen_quat.w()));
  return tf_transform;
}