#include <iostream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/io/pcd_io.h>
#include <chrono>

void filter_points(pcl::PointCloud<pcl::PointXYZI>::Ptr& input, pcl::PointCloud<pcl::PointXYZI>::Ptr& output,
                   float scale = 0.2)
{
  // 创建Voxel Grid滤波器对象
  pcl::VoxelGrid<pcl::PointXYZI> pcl_filter;
  pcl_filter.setInputCloud(input);
  //   pcl_filter.setLeafSize(0.1f, 0.1, 0.1f);
  pcl_filter.setLeafSize(scale, scale, scale);
  // pcl_filter.setLeafSize(0.4f, 0.4f, 0.4f);
  pcl_filter.filter(*output);
}

int main()
{
  std::string ws = "/home/justin/zhenrobot/map/korea/";
  Eigen::Matrix4f initial_guess;
  Eigen::Vector3f croped_center;
  initial_guess << -0.96, 0.249, 0.0, 9.5, -0.2, -0.96, 0.000, 1.096, 0.0, 0.00, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  croped_center << 24, 0, 0;
  float vovel_size = 0.2;

  pcl::PointCloud<pcl::PointXYZI>::Ptr source(new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(ws + "GlobalMap.pcd", *source) == -1)
  {
    PCL_ERROR("Couldn't read PCD file.\n");
    return -1;
  }
  pcl::PointCloud<pcl::PointXYZI>::Ptr target(new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(ws + "alleyway.pcd", *target) == -1)
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

  // // 定义截取的原点和半径
  // pcl::PointXYZI center;  // 截取原点
  // center.x = -28;
  // center.y = 10;
  // center.z = -1;
  // float radius = 1.0;  // 截取半径

  // // 创建RadiusOutlierRemoval滤波器对象
  // pcl::RadiusOutlierRemoval<pcl::PointXYZI> radius_filter;
  // radius_filter.setInputCloud(target);
  // radius_filter.setRadiusSearch(radius);
  // radius_filter.setNegative(true);  // 设置为true以保留指定半径范围内的点云

  // // 执行点云截取操作
  // radius_filter.filter(*filtered_cloud);

  // // 保存截取后的点云
  // pcl::io::savePCDFile<pcl::PointXYZI>("/home/zhen/Rosbag/test_pcd/filtered_cloud.pcd", *filtered_cloud);

  pcl::PointCloud<pcl::PointXYZI>::Ptr croped_target(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr croped_source(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::CropBox<pcl::PointXYZI> crop_filter;
  crop_filter.setInputCloud(target);
  crop_filter.setMin(Eigen::Vector4f(croped_center(0) - 50, croped_center(1) - 50, croped_center(2) - 50,
                                     1.0));  // 设置框体的最小点坐标
  crop_filter.setMax(Eigen::Vector4f(croped_center(0) + 50, croped_center(1) + 50.0, croped_center(2) + 50.0,
                                     1.0));  // 设置框体的最大点坐标
  // // 执行点云截取操作
  crop_filter.filter(*croped_target);
  // 保存截取后的点云
  pcl::io::savePCDFile<pcl::PointXYZI>(ws + "croped_target.pcd", *croped_target);

  crop_filter.setInputCloud(trans_source);
  crop_filter.setMin(Eigen::Vector4f(croped_center(0) - 50, croped_center(1) - 50, croped_center(2) - 50,
                                     1.0));  // 设置框体的最小点坐标
  crop_filter.setMax(Eigen::Vector4f(croped_center(0) + 50, croped_center(1) + 50.0, croped_center(2) + 50.0,
                                     1.0));  // 设置框体的最大点坐标
  // 执行点云截取操作
  crop_filter.filter(*croped_source);
  // 保存截取后的点云
  pcl::io::savePCDFile<pcl::PointXYZI>(ws + "croped_source.pcd", *croped_source);

  // 创建配准对象
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> gicp;
  gicp.setInputSource(croped_source);
  gicp.setInputTarget(croped_target);

  // 设置配准参数
  gicp.setMaximumIterations(500);          // 设置最大迭代次数
  gicp.setTransformationEpsilon(1e-8);     // 设置收敛条件
  gicp.setEuclideanFitnessEpsilon(0.001);  // 设置配准误差
  // 获取当前时间点
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  // 执行配准
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
  gicp2.setMaximumIterations(500);          // 设置最大迭代次数
  gicp2.setTransformationEpsilon(1e-8);     // 设置收敛条件
  gicp2.setEuclideanFitnessEpsilon(0.001);  // 设置配准误差
  Eigen::Matrix4f trans = gicp.getFinalTransformation();
  gicp2.align(trans_cloud, trans);

  // 输出配准结果
  std::cout << "trans_cloud: " << trans_cloud.size() << std::endl;
  std::cout << "source: " << source->size() << std::endl;
  std::cout << "是否收敛: " << gicp2.hasConverged() << std::endl;
  std::cout << "迭代次数: " << gicp2.getMaximumIterations() << std::endl;
  std::cout << "配准误差: " << gicp2.getFitnessScore() << std::endl;
  std::cout << "变换矩阵:\n" << gicp2.getFinalTransformation() << std::endl;

  *target += trans_cloud;
  pcl::io::savePCDFileBinary(ws + "merged_map.pcd", *target);

  // // // 创建NDT对象并设置参数
  // pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
  // ndt.setResolution(1.0);            // 设置NDT的体素分辨率
  // ndt.setMaximumIterations(100000);  // 设置最大迭代次数

  // // // 设置目标点云作为参考点云
  // ndt.setInputTarget(target);

  // // // 设置源点云作为输入点云，并执行配准
  // ndt.setInputSource(source);
  // // // 获取当前时间点
  // std::chrono::steady_clock::time_point start_ndt = std::chrono::steady_clock::now();

  // ndt.align(aligned_cloud, initial_guess);

  // // // 获取当前时间点
  // std::chrono::steady_clock::time_point end_ndt = std::chrono::steady_clock::now();

  // // // 计算时间差（以毫秒为单位）
  // std::chrono::duration<double, std::milli> duration_ms_ndt = end_ndt - start_ndt;

  // // // 打印时间差
  // std::cout << "时间差（毫秒）: " << duration_ms_ndt.count() << "ms" << std::endl;
  // // 输出配准结果及变换矩阵
  // std::cout << "配准是否成功：" << ndt.hasConverged() << std::endl;
  // std::cout << "变换矩阵：" << std::endl << ndt.getFinalTransformation() << std::endl;
  // pcl::io::savePCDFileBinary("/home/zhen/Rosbag/test_pcd/ndt_aligned_cloud.pcd", aligned_cloud);

  // Eigen::Vector3f euler_angles = initial_guess.block<3, 3>(0, 0).eulerAngles(2, 1, 0);  // 顺序为 ZYX

  // // 输出欧拉角
  // std::cout << "Roll: " << euler_angles(2) << std::endl;
  // std::cout << "Pitch: " << euler_angles(1) << std::endl;
  // std::cout << "Yaw: " << euler_angles(0) << std::endl;

  return 0;
}
