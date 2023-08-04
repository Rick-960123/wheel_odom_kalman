#include <iostream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/io/pcd_io.h>
#include <chrono>

void filter_points(pcl::PointCloud<pcl::PointXYZ>::Ptr& input, pcl::PointCloud<pcl::PointXYZ>::Ptr& output,
                   float scale = 0.25)
{
  // 创建Voxel Grid滤波器对象
  pcl::VoxelGrid<pcl::PointXYZ> pcl_filter;
  pcl_filter.setInputCloud(input);
  //   pcl_filter.setLeafSize(0.1f, 0.1, 0.1f);
  pcl_filter.setLeafSize(scale, scale, scale);
  // pcl_filter.setLeafSize(0.4f, 0.4f, 0.4f);
  pcl_filter.filter(*output);
}

int main()
{
  // 加载雷达点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr radar_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // 加载雷达点云数数据
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/zhen/Rosbag/test_pcd/"
                                          "source.pcd",
                                          *radar_cloud) == -1)
  {
    PCL_ERROR("Couldn't read PCD file.\n");
    return -1;
  }

  // 加载PCD文件点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("/home/zhen/Rosbag/test_pcd/"
                                          "target.pcd",
                                          *pcd_cloud) == -1)
  {
    PCL_ERROR("Couldn't read PCD file.\n");
    return -1;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
  filter_points(radar_cloud, source);
  filter_points(pcd_cloud, target);

  // pcl::io::savePCDFile<pcl::PointXYZ>("/home/zhen/Rosbag/test_pcd/source.pcd", *source);
  // pcl::io::savePCDFile<pcl::PointXYZ>("/home/zhen/Rosbag/test_pcd/target.pcd", *target);

  // // 定义截取的原点和半径
  // pcl::PointXYZ center;  // 截取原点
  // center.x = -28;
  // center.y = 10;
  // center.z = -1;
  // float radius = 1.0;  // 截取半径

  // // 创建RadiusOutlierRemoval滤波器对象
  // pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius_filter;
  // radius_filter.setInputCloud(target);
  // radius_filter.setRadiusSearch(radius);
  // radius_filter.setNegative(true);  // 设置为true以保留指定半径范围内的点云

  // // 执行点云截取操作
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_target(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_source(new pcl::PointCloud<pcl::PointXYZ>);
  // radius_filter.filter(*filtered_cloud);

  // // 保存截取后的点云
  // pcl::io::savePCDFile<pcl::PointXYZ>("/home/zhen/Rosbag/test_pcd/filtered_cloud.pcd", *filtered_cloud);

  // 定义截取的框体参数
  pcl::CropBox<pcl::PointXYZ> crop_filter;
  crop_filter.setInputCloud(target);
  crop_filter.setMin(Eigen::Vector4f(20, -170, -50, 1.0));       // 设置框体的最小点坐标
  crop_filter.setMax(Eigen::Vector4f(120.0, -70.0, 50.0, 1.0));  // 设置框体的最大点坐标
  // // 执行点云截取操作
  crop_filter.filter(*filtered_target);
  // 保存截取后的点云
  pcl::io::savePCDFile<pcl::PointXYZ>("/home/zhen/Rosbag/test_pcd/output_cloud_target.pcd", *filtered_target);

  crop_filter.setInputCloud(source);
  crop_filter.setMin(Eigen::Vector4f(20.0, -60.0, -50.0, 1.0));  // 设置框体的最小点坐标
  crop_filter.setMax(Eigen::Vector4f(120.0, 40.0, 50.0, 1.0));   // 设置框体的最大点坐标
  // // 执行点云截取操作
  crop_filter.filter(*filtered_source);
  // 保存截取后的点云
  pcl::io::savePCDFile<pcl::PointXYZ>("/home/zhen/Rosbag/test_pcd/output_cloud_source.pcd", *filtered_source);

  // 创建配准对象
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.setInputSource(filtered_source);
  gicp.setInputTarget(filtered_target);

  // 设置配准参数
  gicp.setMaximumIterations(500);          // 设置最大迭代次数
  gicp.setTransformationEpsilon(1e-8);     // 设置收敛条件
  gicp.setEuclideanFitnessEpsilon(0.001);  // 设置配准误差
  // 获取当前时间点
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  // 执行配准
  pcl::PointCloud<pcl::PointXYZ> aligned_cloud;

  Eigen::Quaternionf quaternion(0.52532199, 0., 0., 0.85090352);
  Eigen::Vector3f translation(-3.0, 112.0, 0.0);

  Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
  // initial_guess.block<3, 3>(0, 0) = quaternion.matrix();
  // initial_guess.block<3, 1>(0, 3) = translation;

  initial_guess << -0.0283884, -0.9958858, -0.00114872, 66, 0.998859, -0.0283883, -0.000254234, -189, -8.23291e-05,
      -0.00117364, 0.999999, 0, 0, 0, 0, 1;

  gicp.align(aligned_cloud, initial_guess);

  // 输出配准结果
  std::cout << "是否收敛: " << gicp.hasConverged() << std::endl;
  std::cout << "迭代次数: " << gicp.getMaximumIterations() << std::endl;
  std::cout << "配准误差: " << gicp.getFitnessScore() << std::endl;
  std::cout << "变换矩阵:\n" << gicp.getFinalTransformation() << std::endl;

  // 获取当前时间点
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  // 计算时间差（以毫秒为单位）
  std::chrono::duration<double, std::milli> duration_ms = end - start;

  // 打印时间差
  std::cout << "时间差（毫秒）: " << duration_ms.count() << "ms" << std::endl;

  // 保存配准后的点云
  pcl::io::savePCDFileBinary("/home/zhen/Rosbag/test_pcd/aligned_cloud.pcd", aligned_cloud);

  // // // 创建NDT对象并设置参数
  // pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
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

  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp2;
  pcl::PointCloud<pcl::PointXYZ> trans_cloud;
  gicp2.setInputSource(source);
  gicp2.setInputTarget(target);
  gicp2.setMaximumIterations(500);           // 设置最大迭代次数
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

  // pcl::PointCloud<pcl::PointXYZ> trans_cloud;
  // trans = gicp2.getFinalTransformation();
  // pcl::transformPointCloud(*radar_cloud, trans_cloud, trans);

  *target += trans_cloud;
  pcl::io::savePCDFileBinary("/home/zhen/Rosbag/test_pcd/trans_map_rizhao_0.2.pcd", *target);

  // Eigen::Vector3f euler_angles = initial_guess.block<3, 3>(0, 0).eulerAngles(2, 1, 0);  // 顺序为 ZYX

  // // 输出欧拉角
  // std::cout << "Roll: " << euler_angles(2) << std::endl;
  // std::cout << "Pitch: " << euler_angles(1) << std::endl;
  // std::cout << "Yaw: " << euler_angles(0) << std::endl;

  return 0;
}
