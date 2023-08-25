#include <iostream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/io/pcd_io.h>
#include <chrono>
#include <ros/ros.h>

void filter_points(pcl::PointCloud<pcl::PointXYZI>::Ptr& input, pcl::PointCloud<pcl::PointXYZI>::Ptr& output,
                   float scale = 0.2)
{
  // 创建Voxel Grid滤波器对象
  pcl::VoxelGrid<pcl::PointXYZI> pcl_filter;
  pcl_filter.setInputCloud(input);
  // pcl_filter.setLeafSize(0.1f, 0.1, 0.1f);
  pcl_filter.setLeafSize(scale, scale, scale);
  // pcl_filter.setLeafSize(0.4f, 0.4f, 0.4f);
  pcl_filter.filter(*output);
}

int main(int argc, char** argv)
{

  setlocale(LC_CTYPE, "zh_CN.utf8");
  ros::init(argc, argv, "merge_map");
  ros::NodeHandle nh;

  std::string ws;
  std::string source_pcd_name;
  std::string target_pcd_name;
  Eigen::Matrix4f initial_guess;
  Eigen::Vector3f croped_center;
  float vovel_size;
  std::vector<float> initial_guess_vec, croped_center_vec;


  nh.param<std::string>("workspace", ws, "");
  nh.param<std::string>("source_pcd_name", source_pcd_name, "");
  nh.param<std::string>("target_pcd_name", target_pcd_name, "");
  nh.param<float>("vovel_size", vovel_size, 0.2);
  nh.param<std::vector<float>>("initial_guess", initial_guess_vec, std::vector<float>());
  nh.param<std::vector<float>>("croped_center", croped_center_vec, std::vector<float>());

  initial_guess = Eigen::Map<const Eigen::Matrix<float, 4, 4>>(initial_guess_vec.data());
  croped_center = Eigen::Map<const Eigen::Matrix<float, 3, 1>>(croped_center_vec.data());

  pcl::PointCloud<pcl::PointXYZI>::Ptr source(new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(ws + source_pcd_name, *source) == -1)
  {
    PCL_ERROR("Couldn't read PCD file.\n");
    return -1;
  }
  pcl::PointCloud<pcl::PointXYZI>::Ptr target(new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(ws + target_pcd_name, *target) == -1)
  {
    PCL_ERROR("Couldn't read PCD file.\n");
    return -1;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr voxeled_target(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr voxeled_source(new pcl::PointCloud<pcl::PointXYZI>);
  filter_points(target, voxeled_target, vovel_size);
  filter_points(source, voxeled_source, vovel_size);
  pcl::io::savePCDFile<pcl::PointXYZI>(ws + "voxeled_source.pcd", *voxeled_source);
  pcl::io::savePCDFile<pcl::PointXYZI>(ws + "voxeled_target.pcd", *voxeled_target);

  pcl::PointCloud<pcl::PointXYZI>::Ptr trans_source(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::transformPointCloud(*voxeled_source, *trans_source, initial_guess);
  pcl::io::savePCDFile<pcl::PointXYZI>(ws + "trans_source.pcd", *trans_source);

  pcl::PointCloud<pcl::PointXYZI>::Ptr croped_target(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr croped_source(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::CropBox<pcl::PointXYZI> crop_filter;
  crop_filter.setInputCloud(target);
  crop_filter.setMin(Eigen::Vector4f(croped_center(0) - 50, croped_center(1) - 50, croped_center(2) - 50,
                                     1.0));  // 设置框体的最小点坐标
  crop_filter.setMax(Eigen::Vector4f(croped_center(0) + 50, croped_center(1) + 50.0, croped_center(2) + 50.0,
                                     1.0));  // 设置框体的最大点坐标
  crop_filter.filter(*croped_target);
  pcl::io::savePCDFile<pcl::PointXYZI>(ws + "croped_target.pcd", *croped_target);

  crop_filter.setInputCloud(trans_source);
  crop_filter.setMin(Eigen::Vector4f(croped_center(0) - 50, croped_center(1) - 50, croped_center(2) - 50,
                                     1.0));  // 设置框体的最小点坐标
  crop_filter.setMax(Eigen::Vector4f(croped_center(0) + 50, croped_center(1) + 50.0, croped_center(2) + 50.0,
                                     1.0));  // 设置框体的最大点坐标
  crop_filter.filter(*croped_source);
  pcl::io::savePCDFile<pcl::PointXYZI>(ws + "croped_source.pcd", *croped_source);

  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> gicp;
  gicp.setInputSource(croped_source);
  gicp.setInputTarget(croped_target);
  gicp.setMaximumIterations(500);
  gicp.setTransformationEpsilon(1e-8);
  gicp.setEuclideanFitnessEpsilon(0.001);
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  pcl::PointCloud<pcl::PointXYZI> aligned_cloud;
  Eigen::Quaternionf quaternion(0.52532199, 0., 0., 0.85090352);
  Eigen::Vector3f translation(-3.0, 112.0, 0.0);
  Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
  gicp.align(aligned_cloud, identity);

  std::cout << "是否收敛: " << gicp.hasConverged() << std::endl;
  std::cout << "迭代次数: " << gicp.getMaximumIterations() << std::endl;
  std::cout << "配准误差: " << gicp.getFitnessScore() << std::endl;
  std::cout << "变换矩阵:\n" << gicp.getFinalTransformation() << std::endl;

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> duration_ms = end - start;
  std::cout << "时间差（毫秒）: " << duration_ms.count() << "ms" << std::endl;

  pcl::io::savePCDFileBinary(ws + "aligned_cloud.pcd", aligned_cloud);

  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> gicp2;
  pcl::PointCloud<pcl::PointXYZI> trans_cloud;
  gicp2.setInputSource(trans_source);
  gicp2.setInputTarget(target);
  gicp2.setMaximumIterations(500);
  gicp2.setTransformationEpsilon(1e-8);
  gicp2.setEuclideanFitnessEpsilon(0.001);
  Eigen::Matrix4f trans = gicp.getFinalTransformation();
  gicp2.align(trans_cloud, trans);

  std::cout << "trans_cloud: " << trans_cloud.size() << std::endl;
  std::cout << "source: " << source->size() << std::endl;
  std::cout << "是否收敛: " << gicp2.hasConverged() << std::endl;
  std::cout << "迭代次数: " << gicp2.getMaximumIterations() << std::endl;
  std::cout << "配准误差: " << gicp2.getFitnessScore() << std::endl;
  std::cout << "变换矩阵:\n" << gicp2.getFinalTransformation() << std::endl;

  *target += trans_cloud;
  pcl::io::savePCDFileBinary(ws + "merged_map.pcd", *target);
  std::cout << "保存路径: " << ws + "merged_map.pcd" << std::endl;
  return 0;
}
