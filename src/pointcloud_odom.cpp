#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>
#include <pthread.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <vector>

#ifdef USE_FAST_PCL
#include <fast_pcl/registration/ndt.h>
#else
#include <pcl/registration/ndt.h>
#endif

// End of adding

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <autoware_msgs/ConfigNdt.h>

#include <autoware_msgs/ndt_stat.h>
#include <sophus/se3.h>
// Added for testing on cpu
#include <fast_pcl/ndt_cpu/NormalDistributionsTransform.h>
// End of adding

#define PREDICT_POSE_THRESHOLD 0.5
#define PI (3.1415926)

#define Wa 0.4
#define Wb 0.2
#define Wc 0.2
#define Wd 0.2  // note-justin    权重

static double fitness_score_threshold = 0.0;

#define fitness_score_threshold_indoor 0.3    // note-justin  户内设置参数较小 0.3
#define fitness_score_threshold_outdoor 20.0  // note-justin  户外设置参数较大 5.0

struct pose
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

static bool need_initial_ = false;

static pose initial_pose, predict_pose, predict_pose_imu, predict_pose_odom, predict_pose_imu_odom, previous_pose,
    previous_odom_pose, ndt_pose, current_pose, current_pose_imu, current_pose_odom, current_pose_imu_odom,
    localizer_pose, previous_gnss_pose, current_gnss_pose;

static pose store_pose;
static pose current_natural_pose, previous_natural_pose;

static nav_msgs::Odometry current_odom_pose;

static double offset_x, offset_y, offset_z, offset_yaw;  // current_pos - previous_pose
static double odom_offset_x, odom_offset_y, odom_offset_z, odom_offset_roll, odom_offset_pitch,
    odom_offset_yaw;  // current_pos - previous_pose
static double previous_pose_roll, previous_pose_pitch, previous_pose_yaw;

static double offset_imu_x, offset_imu_y, offset_imu_z, offset_imu_roll, offset_imu_pitch, offset_imu_yaw;
static double offset_odom_x, offset_odom_y, offset_odom_z, offset_odom_roll, offset_odom_pitch, offset_odom_yaw;
static double offset_imu_odom_x, offset_imu_odom_y, offset_imu_odom_z, offset_imu_odom_roll, offset_imu_odom_pitch,
    offset_imu_odom_yaw;

static pcl::PointCloud<pcl::PointXYZ> global_map, sub_map;

// If the map is loaded, map_loaded will be 1.
static int map_loaded = 0;
static int _use_gnss = 1;
static int init_pos_set = 0;
static int init_ekf_set = 0;
static int init_ekf_relocalization_cnt = 0;
static int init_ekf_general_cnt = 0;

geometry_msgs::PoseWithCovarianceStamped desired_relocalization_pose_;

#ifdef CUDA_FOUND
static std::shared_ptr<gpu::GNormalDistributionsTransform> gpu_ndt_ptr =
    std::make_shared<gpu::GNormalDistributionsTransform>();
#endif

static cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> cpu_ndt;

static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;

// Default values
static int max_iter = 30;        // Maximum iterations
static float ndt_res = 1.0;      // Resolution
static double step_size = 0.1;   // Step size
static double trans_eps = 0.01;  // Transformation epsilon

static ros::Publisher sub_map_pub;

static ros::Publisher predict_pose_pub;
static geometry_msgs::PoseStamped predict_pose_msg;

static ros::Publisher predict_pose_imu_pub;
static geometry_msgs::PoseStamped predict_pose_imu_msg;

static ros::Publisher predict_pose_odom_pub;
static geometry_msgs::PoseStamped predict_pose_odom_msg;

static ros::Publisher predict_pose_imu_odom_pub;
static geometry_msgs::PoseStamped predict_pose_imu_odom_msg;

static ros::Publisher ndt_pose_pub;
static geometry_msgs::PoseStamped ndt_pose_msg;
static geometry_msgs::PoseStamped pre_ndt_pose_msg;

static ros::Publisher natural_pose_pub;
static geometry_msgs::PoseStamped natural_pose_msg;

static ros::Publisher ndt_cov_pub;
static geometry_msgs::PoseWithCovarianceStamped ndt_cov_msg;

// current_pose is published by vel_pose_mux
/*
static ros::Publisher current_pose_pub;
static geometry_msgs::PoseStamped current_pose_msg;
*/

static bool frist_imu_flag = true;

static ros::Publisher localizer_pose_pub;
static geometry_msgs::PoseStamped localizer_pose_msg;

static ros::Publisher estimate_twist_pub;
static geometry_msgs::TwistStamped estimate_twist_msg;

static ros::Time current_scan_time;
static ros::Time previous_scan_time;
static ros::Duration scan_duration;

pcl::CropBox<pcl::PointXYZ> crop_filter;
pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
pcl::VoxelGrid<pcl::PointXYZ> vox_filter;


static double exe_time = 0.0;
static bool has_converged;

static bool ndt_pose_used = true;

static int iteration = 0;
static double fitness_score = 0.0;
static double trans_probability = 0.0;

static double fitness_score_mean = 0.0;
static double fitness_score_median = 0.0;
static double ndt_relibility_data = 0.0;
static int ndt_relibility_data_cnt = 0;
static int ndt_cnt_divide = 0;
static int use_wheel_odom_cnt = 0;
static int triger_wheel_odom_cnt = 0;
static bool use_wheel_odom_flag = false;
static bool use_wheel_odom_lift_flag = false;
static bool use_wheel_relocalization_flag = false;

static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_yaw;

static double current_velocity = 0.0, previous_velocity = 0.0, previous_previous_velocity = 0.0;  //[m/s]
static double current_velocity_x = 0.0, previous_velocity_x = 0.0;
static double current_velocity_y = 0.0, previous_velocity_y = 0.0;
static double current_velocity_z = 0.0, previous_velocity_z = 0.0;
// static double current_velocity_yaw = 0.0, previous_velocity_yaw = 0.0;
static double current_velocity_smooth = 0.0;

static double current_velocity_imu_x = 0.0;
static double current_velocity_imu_y = 0.0;
static double current_velocity_imu_z = 0.0;

static double current_accel = 0.0, previous_accel = 0.0;  //[m/s^2]
static double current_accel_x = 0.0;
static double current_accel_y = 0.0;
static double current_accel_z = 0.0;
// static double current_accel_yaw = 0.0;

static double angular_velocity = 0.0;

static int use_predict_pose = 0;

static ros::Publisher estimated_vel_mps_pub, estimated_vel_kmph_pub, estimated_vel_pub;
static std_msgs::Float32 estimated_vel_mps, estimated_vel_kmph, previous_estimated_vel_kmph;
static ros::Publisher estimated_offset_imu_y_pub;
static std_msgs::Float32 offset_imu_y_data;

static std::chrono::time_point<std::chrono::system_clock> matching_start, matching_end;

static ros::Publisher time_ndt_matching_pub;
static std_msgs::Float32 time_ndt_matching;

static int _queue_size = 1000;

static ros::Publisher ndt_stat_pub;
static autoware_msgs::ndt_stat ndt_stat_msg;

static double predict_pose_error = 0.0;

static double _tf_x, _tf_y, _tf_z, _tf_roll, _tf_pitch, _tf_yaw;
static Eigen::Matrix4f tf_btol;

static std::string _localizer = "rslidar";
static std::string _offset = "linear";  // linear, zero, quadratic

static ros::Publisher ndt_reliability_pub;
static std_msgs::Float32 ndt_reliability;

static ros::Publisher ndt_fitness_pub;
static std_msgs::Float32 fitness;

static ros::Publisher lost_path_point_pub;
static std_msgs::Int32 lost_path_point;

static bool stay_lift_localization_flag = false;
static bool ndt_to_odom_pulse_flag = false;
static bool out_lift_localization_flag = true;
static bool jump_lift_localization_flag = false;
static bool in_lift_localization_flag = false;

static std_msgs::Int32 start_flag;

static bool _use_gpu = false;
static bool _use_openmp = false;

static bool _use_fast_pcl = false;

static bool _get_height = false;
static bool _use_local_transform = false;
static bool _use_imu = true;
static bool _use_odom = true;
static bool _imu_upside_down = false;

static double imu_roll, imu_pitch, imu_yaw;

tf::Quaternion imu_orientation;

static ros::Publisher relocal_flag_pub;
static ros::Publisher wheel_odom_trans_pub;
static ros::Publisher into_lift_trans_pub;
static ros::Publisher ndt_yaw_pub;
static ros::Publisher initialpose_pub_;
static ros::Publisher sound_pub_;
static ros::Publisher waypoint_change_flag_pub;
static std_msgs::Bool relocal_flag;

static bool _matching_up = true;

static std::string _imu_topic = "/imu_raw";

static std::ofstream ofs;
static std::string filename;

static sensor_msgs::Imu imu;
static nav_msgs::Odometry odom;

// static tf::TransformListener local_transform_listener;
static tf::StampedTransform local_transform;

static tf::Quaternion imu_q;

static uint points_map_num = 0;

static double pre_yaw;

static double current_odom_pose_roll = 0.0;
static double current_odom_pose_pitch = 0.0;
static double current_odom_pose_yaw = 0.0;
static double diff_x_new = 0.0;
static double diff_y_new = 0.0;
static double diff_z_new = 0.0;

bool inital_pose_flag = true;
static int ndt_normal_cnt = 0;
static int ndt_normal_cnt_divide = 0;
static int odom_pose_used_cnt = 0;  // note-justin    持续使用odom的计数

static int encoder_position_cnt = 0;
static int save_point_cnt = 0;
static int save_point_cnt_divide = 0;

static int ekf_pose_cnt = 0;

static int lost_point_cnt = 0;

static bool indoor_;

static int waypoint_change_flag;

static int manually_into_environment;

const int fitness_window_size = 5;
std::vector<double> fitness_filter_buf;

pthread_mutex_t mutex;

static void wheel_odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
  odom = *msg;

  current_odom_pose.pose.pose.position.x = msg->pose.pose.position.x;
  current_odom_pose.pose.pose.position.y = msg->pose.pose.position.y;
  current_odom_pose.pose.pose.position.z = msg->pose.pose.position.z;

  current_odom_pose.pose.pose.orientation.x = msg->pose.pose.orientation.x;
  current_odom_pose.pose.pose.orientation.y = msg->pose.pose.orientation.y;
  current_odom_pose.pose.pose.orientation.z = msg->pose.pose.orientation.z;
  current_odom_pose.pose.pose.orientation.w = msg->pose.pose.orientation.w;

  tf::Quaternion RQ2;
  tf::quaternionMsgToTF(msg->pose.pose.orientation, RQ2);

  tf::Matrix3x3(RQ2).getRPY(current_odom_pose_roll, current_odom_pose_pitch, current_odom_pose_yaw);

  current_odom_pose_roll = 0.0;
  current_odom_pose_pitch = 0.0;

  odom_offset_x = current_odom_pose.pose.pose.position.x - previous_odom_pose.x;
  odom_offset_x = current_odom_pose.pose.pose.position.y - previous_odom_pose.y;
  odom_offset_x = current_odom_pose.pose.pose.position.z - previous_odom_pose.z;

  odom_offset_roll = current_odom_pose_roll - previous_odom_pose.roll;
  odom_offset_pitch = current_odom_pose_pitch - previous_odom_pose.pitch;
  odom_offset_yaw = current_odom_pose_yaw - previous_odom_pose.yaw;

  previous_odom_pose.x = current_odom_pose.pose.pose.position.x;
  previous_odom_pose.y = current_odom_pose.pose.pose.position.y;
  previous_odom_pose.z = current_odom_pose.pose.pose.position.z;

  previous_odom_pose.roll = current_odom_pose_roll;
  previous_odom_pose.pitch = current_odom_pose_pitch;
  previous_odom_pose.yaw = current_odom_pose_yaw;
}

static void use_wheel_odom_callback(const std_msgs::BoolConstPtr& msg)
{
  use_wheel_odom_lift_flag = msg->data;
}

void waypointchangeflagCallback(const std_msgs::Int32::ConstPtr& msg)
{
  waypoint_change_flag = msg->data;
  use_wheel_odom_flag = waypoint_change_flag == 5 ? true : false;
}

void manuallyEnvironCallback(const std_msgs::Int32& msg)
{
  manually_into_environment = msg.data;
}

void lostPathPointCallback(const std_msgs::Int32::ConstPtr& msg)
{
  if (msg->data == 0)
  {
    lost_path_point_flag = msg->data;
    ROS_INFO("ndt_matching.cpp:: lostPathPointCallback Received lost path point: %d", msg->data);
  }
}

/**************************************************************************/

static void param_callback(const autoware_msgs::ConfigNdt::ConstPtr& input)
{
  if (_use_gnss != input->init_pos_gnss)
  {
    init_pos_set = 0;
  }
  else if (_use_gnss == 0 &&
           (initial_pose.x != input->x || initial_pose.y != input->y || initial_pose.z != input->z ||
            initial_pose.roll != input->roll || initial_pose.pitch != input->pitch || initial_pose.yaw != input->yaw))
  {
    init_pos_set = 0;
  }

  _use_gnss = input->init_pos_gnss;

  // Setting parameters
  if (input->resolution != ndt_res)
  {
    ndt_res = input->resolution;
#ifdef CUDA_FOUND
    if (_use_gpu == true)
    {
      gpu_ndt_ptr->setResolution(ndt_res);
    }
    else
    {
#endif
      if (_use_fast_pcl)
      {
        cpu_ndt.setResolution(ndt_res);
      }
      else
      {
        ndt.setResolution(ndt_res);
      }
#ifdef CUDA_FOUND
    }
#endif
  }
  if (input->step_size != step_size)
  {
    step_size = input->step_size;
#ifdef CUDA_FOUND
    if (_use_gpu == true)
    {
      gpu_ndt_ptr->setStepSize(step_size);
    }
    else
    {
#endif
      if (_use_fast_pcl)
      {
        cpu_ndt.setStepSize(step_size);
      }
      else
      {
        ndt.setStepSize(step_size);
      }
#ifdef CUDA_FOUND
    }
#endif
  }
  if (input->trans_epsilon != trans_eps)
  {
    trans_eps = input->trans_epsilon;
#ifdef CUDA_FOUND
    if (_use_gpu == true)
    {
      gpu_ndt_ptr->setTransformationEpsilon(trans_eps);
    }
    else
    {
#endif
      if (_use_fast_pcl)
      {
        cpu_ndt.setTransformationEpsilon(trans_eps);
      }
      else
      {
        ndt.setTransformationEpsilon(trans_eps);
      }
#ifdef CUDA_FOUND
    }
#endif
  }
  if (input->max_iterations != max_iter)
  {
    max_iter = input->max_iterations;
#ifdef CUDA_FOUND
    if (_use_gpu == true)
    {
      gpu_ndt_ptr->setMaximumIterations(max_iter);
    }
    else
    {
#endif
      if (_use_fast_pcl)
      {
        cpu_ndt.setMaximumIterations(max_iter);
      }
      else
      {
        ndt.setMaximumIterations(max_iter);
      }
#ifdef CUDA_FOUND
    }
#endif
  }

  if (_use_gnss == 0 && init_pos_set == 0)
  {
    initial_pose.x = input->x;
    initial_pose.y = input->y;
    initial_pose.z = input->z;
    initial_pose.roll = input->roll;
    initial_pose.pitch = input->pitch;
    initial_pose.yaw = input->yaw;

    if (_use_local_transform == true)
    {
      tf::Vector3 v(input->x, input->y, input->z);
      tf::Quaternion q;
      q.setRPY(input->roll, input->pitch, input->yaw);
      tf::Transform transform(q, v);
      initial_pose.x = (local_transform.inverse() * transform).getOrigin().getX();
      initial_pose.y = (local_transform.inverse() * transform).getOrigin().getY();
      initial_pose.z = (local_transform.inverse() * transform).getOrigin().getZ();

      tf::Matrix3x3 m(q);
      m.getRPY(initial_pose.roll, initial_pose.pitch, initial_pose.yaw);

      std::cout << "initial_pose.x: " << initial_pose.x << std::endl;
      std::cout << "initial_pose.y: " << initial_pose.y << std::endl;
      std::cout << "initial_pose.z: " << initial_pose.z << std::endl;
      std::cout << "initial_pose.roll: " << initial_pose.roll << std::endl;
      std::cout << "initial_pose.pitch: " << initial_pose.pitch << std::endl;
      std::cout << "initial_pose.yaw: " << initial_pose.yaw << std::endl;
    }

    // Setting position and posture for the first time.
    localizer_pose.x = initial_pose.x;
    localizer_pose.y = initial_pose.y;
    localizer_pose.z = initial_pose.z;
    localizer_pose.roll = initial_pose.roll;
    localizer_pose.pitch = initial_pose.pitch;
    localizer_pose.yaw = initial_pose.yaw;

    previous_pose.x = initial_pose.x;
    previous_pose.y = initial_pose.y;
    previous_pose.z = initial_pose.z;
    previous_pose.roll = initial_pose.roll;
    previous_pose.pitch = initial_pose.pitch;
    previous_pose.yaw = initial_pose.yaw;

    current_pose.x = initial_pose.x;
    current_pose.y = initial_pose.y;
    current_pose.z = initial_pose.z;
    current_pose.roll = initial_pose.roll;
    current_pose.pitch = initial_pose.pitch;
    current_pose.yaw = initial_pose.yaw;

    current_velocity = 0;
    current_velocity_x = 0;
    current_velocity_y = 0;
    current_velocity_z = 0;
    angular_velocity = 0;

    current_pose_imu.x = 0;
    current_pose_imu.y = 0;
    current_pose_imu.z = 0;
    current_pose_imu.roll = 0;
    current_pose_imu.pitch = 0;
    current_pose_imu.yaw = 0;

    current_velocity_imu_x = current_velocity_x;
    current_velocity_imu_y = current_velocity_y;
    current_velocity_imu_z = current_velocity_z;
    init_pos_set = 1;

    ROS_INFO(
        "ndt_matching.cpp:: param_callback subscribe config/ndt  initial_pose.x = %f initial_pose.y = %f "
        "initial_pose.z = %f initial_pose.roll = %f initial_pose.pitch = %f initial_pose.yaw = %f",
        initial_pose.x, initial_pose.y, initial_pose.z, initial_pose.roll, initial_pose.pitch, initial_pose.yaw);

    pre_ndt_pose_msg.header.frame_id = "map";
    pre_ndt_pose_msg.header.stamp = current_scan_time;
    pre_ndt_pose_msg.pose.position.x = initial_pose.x;
    pre_ndt_pose_msg.pose.position.y = initial_pose.y;
    pre_ndt_pose_msg.pose.position.z = initial_pose.z;
    pre_ndt_pose_msg.pose.orientation = tf::createQuaternionMsgFromYaw(initial_pose.yaw);
  }
}

static void map_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
  if (input->width > 0)
  {
    pcl::fromROSMsg(*input, global_map);
    crop_filter.setInputCloud(global_map);
    map_loaded = 1;
  }
  else
  {
    ROS_ERROR("无效的地图!")
  }
}

static void gnss_callback(const geometry_msgs::PoseStamped::ConstPtr& input)
{
  tf::Quaternion gnss_q(input->pose.orientation.x, input->pose.orientation.y, input->pose.orientation.z,
                        input->pose.orientation.w);
  tf::Matrix3x3 gnss_m(gnss_q);
  current_gnss_pose.x = input->pose.position.x;
  current_gnss_pose.y = input->pose.position.y;
  current_gnss_pose.z = input->pose.position.z;
  gnss_m.getRPY(current_gnss_pose.roll, current_gnss_pose.pitch, current_gnss_pose.yaw);

  // Justin 当pose突然丢掉时候，可以用gnss来进行纠正，作为全局定位的依据，后续可以和VSLAM结合

  if ((_use_gnss == 1 && init_pos_set == 0) || fitness_score >= 30000000.0)
  {
    previous_pose.x = previous_gnss_pose.x;
    previous_pose.y = previous_gnss_pose.y;
    previous_pose.z = previous_gnss_pose.z;
    previous_pose.roll = previous_gnss_pose.roll;
    previous_pose.pitch = previous_gnss_pose.pitch;
    previous_pose.yaw = previous_gnss_pose.yaw;

    current_pose.x = current_gnss_pose.x;
    current_pose.y = current_gnss_pose.y;
    current_pose.z = current_gnss_pose.z;
    current_pose.roll = current_gnss_pose.roll;
    current_pose.pitch = current_gnss_pose.pitch;
    current_pose.yaw = current_gnss_pose.yaw;

    current_pose_imu = current_pose_odom = current_pose_imu_odom = current_pose;

    offset_x = current_pose.x - previous_pose.x;
    offset_y = current_pose.y - previous_pose.y;
    offset_z = current_pose.z - previous_pose.z;
    offset_yaw = current_pose.yaw - previous_pose.yaw;

    init_pos_set = 1;
  }

  previous_gnss_pose.x = current_gnss_pose.x;
  previous_gnss_pose.y = current_gnss_pose.y;
  previous_gnss_pose.z = current_gnss_pose.z;
  previous_gnss_pose.roll = current_gnss_pose.roll;
  previous_gnss_pose.pitch = current_gnss_pose.pitch;
  previous_gnss_pose.yaw = current_gnss_pose.yaw;
}

static void wheel_odom_callback(const geometry_msgs::PoseStamped::ConstPtr& input)
{
  need_initial_ = false;

  tf::Quaternion gnss_q(input->pose.orientation.x, input->pose.orientation.y, input->pose.orientation.z,
                        input->pose.orientation.w);
  tf::Matrix3x3 gnss_m(gnss_q);
  current_gnss_pose.x = input->pose.position.x;
  current_gnss_pose.y = input->pose.position.y;
  current_gnss_pose.z = input->pose.position.z;
  gnss_m.getRPY(current_gnss_pose.roll, current_gnss_pose.pitch, current_gnss_pose.yaw);

  tf::Vector3 ekf_pose_vector(current_gnss_pose.x, current_gnss_pose.y, current_gnss_pose.z);
  tf::Vector3 ndt_pose_vector(current_pose.x, current_pose.y, current_pose.z);

  double distance = tf::tfDistance(ekf_pose_vector, ndt_pose_vector);

  float reposition_threshold_data = 0.0;

  if (waypoint_change_flag == 4)
  {
    reposition_threshold_data = 500000.0;
  }
  else
  {
    reposition_threshold_data = 500000.0;
  }

  init_ekf_general_cnt++;

  if (ndt_reliability_data_mean >= reposition_threshold_data)
  {
    init_ekf_relocalization_cnt++;
  }
  else
  {
  }

  if (init_ekf_relocalization_cnt >= 10 || lost_path_point_flag == 0)
  {
    use_wheel_relocalization_flag = true;
    lost_path_point_flag = 1;
    ROS_INFO(
        "ndt_matching.cpp:: wheel_odom_callback ========init_ekf_relocalization_cnt = %d "
        "=======ndt_reliability_data_mean = %f====================lost_path_point_flag = %d ",
        init_ekf_relocalization_cnt, ndt_reliability_data_mean, lost_path_point_flag);
  }

  // Justin 当pose突然丢掉时候，可以用gnss来进行纠正，作为全局定位的依据，后续可以和VSLAM结合
  if ((init_ekf_set == 0) && (use_wheel_relocalization_flag == true) &&
      (waypoint_change_flag != 5))  // note-justin indoor 50 outdoor 80
  {
    tf::TransformListener listener;
    tf::StampedTransform transform;
    try
    {
      ros::Time now = ros::Time(0);
      listener.waitForTransform("map", input->header.frame_id, now, ros::Duration(50.0));
      listener.lookupTransform("map", input->header.frame_id, now, transform);
    }
    catch (tf::TransformException& ex)
    {
      // ROS_ERROR("%s", ex.what());
    }

    if (_use_local_transform == true)
    {
      current_pose.x = current_gnss_pose.x;
      current_pose.y = current_gnss_pose.y;
      current_pose.z = current_gnss_pose.z;
      ROS_INFO(
          "ndt_matching.cpp:: wheel_odom_callback current_pose_x = %f current_pose_y = %f current_pose_z = %f "
          "_use_local_transform = %d",
          current_pose.x, current_pose.y, current_pose.z, _use_local_transform);
    }
    else
    {
      // Justin 从rviz手动给出的pose是基于world坐标系的，所以需要将其转换到map坐标系

      current_pose.x = current_gnss_pose.x + transform.getOrigin().x();
      current_pose.y = current_gnss_pose.y + transform.getOrigin().y();
      current_pose.z = current_gnss_pose.z + transform.getOrigin().z();
      ROS_INFO(
          "ndt_matching.cpp:: wheel_odom_callback current_pose_x = %f current_pose_y = %f current_pose_z = %f "
          "_use_local_transform = %d",
          current_pose.x, current_pose.y, current_pose.z, _use_local_transform);
    }

    init_ekf_general_cnt = 0;
    init_ekf_relocalization_cnt = 0;
    use_wheel_relocalization_flag = false;

    current_pose.roll = current_gnss_pose.roll;
    current_pose.pitch = current_gnss_pose.pitch;
    current_pose.yaw = current_gnss_pose.yaw;  // note-justin  使用IMU的YAW

    current_pose_imu = current_pose_odom = current_pose_imu_odom = current_pose;

    previous_pose.x = current_pose.x;
    previous_pose.y = current_pose.y;
    previous_pose.z = current_pose.z;
    previous_pose.roll = current_pose.roll;
    previous_pose.pitch = current_pose.pitch;
    previous_pose.yaw = current_pose.yaw;

    offset_x = 0.0;
    offset_y = 0.0;
    offset_z = 0.0;
    offset_yaw = 0.0;

    offset_imu_x = 0.0;
    offset_imu_y = 0.0;
    offset_imu_z = 0.0;
    offset_imu_roll = 0.0;
    offset_imu_pitch = 0.0;
    offset_imu_yaw = 0.0;

    offset_odom_x = 0.0;
    offset_odom_y = 0.0;
    offset_odom_z = 0.0;
    offset_odom_roll = 0.0;
    offset_odom_pitch = 0.0;
    offset_odom_yaw = 0.0;

    offset_imu_odom_x = 0.0;
    offset_imu_odom_y = 0.0;
    offset_imu_odom_z = 0.0;
    offset_imu_odom_roll = 0.0;
    offset_imu_odom_pitch = 0.0;
    offset_imu_odom_yaw = 0.0;

    ROS_INFO(
        "\n\nndt_matching.cpp:: wheel_odom_callback ===============================RESET NDT "
        "POSE==================%f=============================================================",
        ndt_reliability_data_mean);
    ROS_ERROR(
        "ndt_matching.cpp:: wheel_odom_callback input->pose.position.x = %f input->pose.position.y = %f "
        "input->pose.position.z = %f current_pose.roll = %f current_pose.pitch = %f current_pose.yaw = %f ekf_pose_cnt "
        "= %d distance = %f",
        input->pose.position.x, input->pose.position.y, input->pose.position.z, current_gnss_pose.roll,
        current_gnss_pose.pitch, current_gnss_pose.yaw, ekf_pose_cnt, distance);
    ROS_INFO(
        "ndt_matching.cpp:: wheel_odom_callback "
        "========================================================================================================\n\n");

    init_ekf_set = 1;
  }

  ekf_pose_cnt = ekf_pose_cnt + 1;

  if (ekf_pose_cnt % 500 == 0)
  {
    init_ekf_set = 0;
    init_ekf_general_cnt = 0;
    init_ekf_relocalization_cnt = 0;
    use_wheel_relocalization_flag = false;
    // ROS_INFO("ndt_matching.cpp:: wheel_odom_callback init_ekf_set = %d ekf_pose_cnt = %d", init_ekf_set,
    // ekf_pose_cnt);
  }

  if (ekf_pose_cnt >= 100000000)
  {
    ekf_pose_cnt = 0;
    init_ekf_set = 0;
    init_ekf_general_cnt = 0;
    init_ekf_relocalization_cnt = 0;

    // ROS_INFO("ndt_matching.cpp:: wheel_odom_callback init_ekf_set = %d  = %d", init_ekf_set, ekf_pose_cnt);
  }
}

static void stay_lift_localization_flagCallback(const std_msgs::Int32::ConstPtr& msg)
{
  waypoint_change_flag = msg->data;

  if (waypoint_change_flag == 5 && out_lift_localization_flag == true)
  {
    out_lift_localization_flag = false;
    jump_lift_localization_flag = false;
    in_lift_localization_flag = true;
    ndt_to_odom_pulse_flag == true;
  }
  else if (waypoint_change_flag != 5 && in_lift_localization_flag == true &&
           use_wheel_odom_lift_flag == false)  // note-justin 5-> 4
  {
    out_lift_localization_flag = true;
    jump_lift_localization_flag = true;
    in_lift_localization_flag = false;
    ndt_to_odom_pulse_flag = false;
  }

  if (((waypoint_change_flag == 5 && ndt_to_odom_pulse_flag == true)) ||
      use_wheel_odom_lift_flag == true)  // note-justin 4->5 同时考虑use_wheel_odom 和路径模式
  // if ((waypoint_change_flag == 5 && ndt_to_odom_pulse_flag == true)) // note-justin 4->5 不考虑use_wheel_odom
  // 和只考虑路径模式5
  {
    stay_lift_localization_flag = true;
    ROS_INFO(
        "ndt_matching.cpp::stay_lift_localization_flag:: waypoint_chang_flag = %d stay_lift_localization_flag = %d "
        "use_wheel_odom_lift_flag = %d",
        waypoint_change_flag, stay_lift_localization_flag, use_wheel_odom_lift_flag);

    ndt_stat_msg.header.stamp = current_scan_time;
    ndt_stat_msg.exe_time = time_ndt_matching.data;
    ndt_stat_msg.iteration = iteration;
    ndt_stat_msg.score = 0.1;
    ndt_stat_msg.velocity = current_velocity;
    ndt_stat_msg.acceleration = current_accel;
    ndt_stat_msg.use_predict_pose = 0;

    ndt_stat_pub.publish(ndt_stat_msg);
  }
  else
  {
    stay_lift_localization_flag = false;
    // ROS_INFO("ndt_matching.cpp::stay_lift_localization_flag:: waypoint_chang_flag = %d stay_lift_localization_flag =
    // %d",waypoint_change_flag, stay_lift_localization_flag);
  }
}

static void go_lift_localization_flagCallback(const std_msgs::Int32::ConstPtr& msg)
{
  if (msg->data == 1)  // 4->5
  {
    ndt_to_odom_pulse_flag = true;
    std::cout << "ndt_matching.cpp::ndt_to_odom_pulse_flag::" << ndt_to_odom_pulse_flag << std::endl;
  }
  else if (msg->data == 0)
  {
    ndt_to_odom_pulse_flag = false;
    std::cout << "ndt_matching.cpp::ndt_to_odom_pulse_flag::" << ndt_to_odom_pulse_flag << std::endl;
  }

  // if(locker == 2)locker = 3;
}

// start_flag callback Functions
static void StartFlagCallback(const std_msgs::Int32& msg)
{
  start_flag = msg;
}
void inverse_se3(Eigen::MatrixXd& T)
{
  Sophus::SE3 T_tmp(T);
  return T_tmp.inverse();
}

void registration_at_scale(pcl::PointCloud<pcl::PointXYZ>::Ptr& pc_scan, pcl::PointCloud<pcl::PointXYZ>::Ptr pc_map,
                           Eigen::MatrixXd& initial, double scale)
{
  icp.setInputSource(pc_scan);
  icp.setInputTarget(pc_map);     // 设置配准参数
  icp.setMaximumIterations(100);       // 设置最大迭代次数
  icp.setTransformationEpsilon(1e-8);     // 设置收敛条件
  icp.setEuclideanFitnessEpsilon(0.001);  // 设置配准误差
  pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
  icp.align(aligned_cloud, initial_guess);

  return icp.getFinalTransformation(), icp.getFitnessScore()
}

void crop_global_map_in_FOV(Eigen::MatrixXd& pose_estimation, Eigen::MatrixXd& cur_odom)
{
  Eigen::MatrixXd T_map_to_base_link = pose_estimation * cur_odom;

  Eigen::Vector4d start, end;
  start = T_map_to_base_link.transpose().block<4, 1>(0, 3) - Eigen::Vector4f(100, 100, 100, 0.0);
  end = T_map_to_base_link.transpose().block<4, 1>(0, 3) + Eigen::Vector4f(100, 100, 100, 0.0);

  crop_filter.setMin(start);
  crop_filter.setMax(end);
  crop_filter.filter(*sub_map);

  sensor_msgs::PointCloud2 sub_map_msg;
  pcl::toROSMsg(*sub_map, sub_map_msg);
  sub_map_msg.header.stamp = point_odom.stamp;
  sub_map_msg.header.frame_id = "map";
  sub_map_pub.publish(sub_map_msg);
}

void global_localization(Eigen::MatrixXd& pose_estimation)
{
  crop_global_map_in_FOV(pose_estimation, cur_odom);

}

static void initialpose_callback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& input)
{
  tf::Quaternion q(input->pose.pose.orientation.x, input->pose.pose.orientation.y, input->pose.pose.orientation.z,
                   input->pose.pose.orientation.w);
  tf::Matrix3x3 m(q);

  m.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

  if (_get_height == true && map_loaded == 1)
  {
    double min_distance = DBL_MAX;
    double nearest_z = current_pose.z;
    for (const auto& p : map)
    {
      double distance = hypot(current_pose.x - p.x, current_pose.y - p.y);
      if (distance < min_distance)
      {
        min_distance = distance;
        nearest_z = p.z;
      }
    }
    current_pose.z = nearest_z;
  }

  current_pose_imu = current_pose_odom = current_pose_imu_odom = current_pose;
  previous_pose.x = current_pose.x;
  previous_pose.y = current_pose.y;
  previous_pose.z = current_pose.z;
  previous_pose.roll = current_pose.roll;
  previous_pose.pitch = current_pose.pitch;
  previous_pose.yaw = current_pose.yaw;
}

static const double wrapToPm(double a_num, const double a_max)
{
  if (a_num >= a_max)
  {
    a_num -= 2.0 * a_max;
  }
  return a_num;
}

static const double wrapToPmPi(double a_angle_rad)
{
  return wrapToPm(a_angle_rad, M_PI);
}

static void imuUpsideDown(const sensor_msgs::Imu::Ptr input)
{
  double input_roll, input_pitch, input_yaw;

  tf::Quaternion input_orientation;
  tf::quaternionMsgToTF(input->orientation, input_orientation);
  tf::Matrix3x3(input_orientation).getRPY(input_roll, input_pitch, input_yaw);

  input->angular_velocity.x *= -1;
  input->angular_velocity.y *= -1;
  input->angular_velocity.z *= -1;

  input->linear_acceleration.x *= -1;
  input->linear_acceleration.y *= -1;
  input->linear_acceleration.z *= -1;

  input_roll *= -1;
  input_pitch *= -1;
  input_yaw *= -1;

  input->orientation = tf::createQuaternionMsgFromRollPitchYaw(input_roll, input_pitch, input_yaw);
}

static void matching_status()
{
  if (fitness_score >= 200.0)
  {
    _matching_up = false;
    relocal_flag.data = true;
  }
  else if (fitness_score < 200.0)
  {
    _matching_up = true;
    relocal_flag.data = false;
  }
  relocal_flag_pub.publish(relocal_flag);
}

static void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
{
  if (need_initial_)
  {
    ROS_ERROR("ndt_matching.cpp:: points_callback  need_initial_ = %d", need_initial_);
    return;
  }

  if (map_loaded == 1 && init_pos_set == 1)
  {
    matching_start = std::chrono::system_clock::now();

    static tf::TransformBroadcaster br, br_imu;
    tf::Transform transform;
    tf::Quaternion predict_q, ndt_q, current_q, localizer_q;

    pcl::PointXYZ p;
    pcl::PointCloud<pcl::PointXYZ> filtered_scan;

    current_scan_time = input->header.stamp;

    pcl::fromROSMsg(*input, filtered_scan);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZ>(filtered_scan));
    int scan_points_num = filtered_scan_ptr->size();

    Eigen::Matrix4f t(Eigen::Matrix4f::Identity());   // base_link
    Eigen::Matrix4f t2(Eigen::Matrix4f::Identity());  // localizer

    std::chrono::time_point<std::chrono::system_clock> align_start, align_end, getFitnessScore_start,
        getFitnessScore_end;
    static double align_time, getFitnessScore_time = 0.0;

    pthread_mutex_lock(&mutex);
#ifdef CUDA_FOUND
    if (_use_gpu == true)
    {
      gpu_ndt_ptr->setInputSource(filtered_scan_ptr);
    }
    else
    {
#endif
      if (_use_fast_pcl)
      {
        cpu_ndt.setInputSource(filtered_scan_ptr);
      }
      else
      {
        ndt.setInputSource(filtered_scan_ptr);
      }
#ifdef CUDA_FOUND
    }
#endif

    pose predict_pose_for_ndt;
    if (inital_pose_flag == false)
    {
      natural_pose_msg.header.frame_id = "map";
      natural_pose_msg.header.stamp = current_scan_time;
      natural_pose_msg.pose.position.x = previous_pose.x + offset_x;
      natural_pose_msg.pose.position.y = previous_pose.y + offset_y;
      natural_pose_msg.pose.position.z = previous_pose.z + offset_z;
      natural_pose_msg.pose.orientation = tf::createQuaternionMsgFromYaw(previous_pose.yaw + offset_yaw);

      current_natural_pose.x = natural_pose_msg.pose.position.x;
      current_natural_pose.y = natural_pose_msg.pose.position.y;
      current_natural_pose.z = natural_pose_msg.pose.position.z;
      current_natural_pose.roll = previous_pose.roll;
      current_natural_pose.pitch = previous_pose.pitch;
      current_natural_pose.yaw = previous_pose.yaw + offset_yaw;

      natural_pose_pub.publish(natural_pose_msg);

      double pose_distance = sqrt(odom_offset_x * odom_offset_x + odom_offset_y * odom_offset_y);
      ////ROS_ERROR("ndt_matching.cpp:: pose_distance = %f for reposition", pose_distance);

      if ((waypoint_change_flag == 3) || (waypoint_change_flag == 4) || (waypoint_change_flag == 6) ||
          (waypoint_change_flag == 7))  // note-justin  室内参数
      {
        fitness_score_threshold = fitness_score_threshold_indoor;
        ////ROS_ERROR("ndt_matching.cpp:: fitness_score_threshold = %f, indoor = %d", fitness_score_threshold, indoor_);
      }
      else if (waypoint_change_flag == 0)  // note-justin  室外参数
      {
        fitness_score_threshold = fitness_score_threshold_outdoor;
        ////ROS_ERROR("ndt_matching.cpp:: fitness_score_threshold = %f, indoor = %d", fitness_score_threshold, indoor_);
      }
      else if (waypoint_change_flag == 100)  // note-justin 如果无法获得路径信息的时候，特指打点的时候
      {
        if (fitness_score_mean > fitness_score_threshold_indoor)
        {
          fitness_score_threshold = fitness_score_threshold_outdoor;
        }
        else
        {
          fitness_score_threshold = fitness_score_threshold_indoor;
        }
      }

      if (jump_lift_localization_flag == true)  // 5->4  note-justin 出电梯的时候
      {
        current_pose.x = current_odom_pose.pose.pose.position.x;
        current_pose.y = current_odom_pose.pose.pose.position.y;
        current_pose.z = current_odom_pose.pose.pose.position.z;
        current_pose.roll = 0.0;
        current_pose.pitch = 0.0;
        current_pose.yaw = current_odom_pose_yaw;  // note-justin  使用IMU的YAW

        current_pose_imu = current_pose_odom = current_pose_imu_odom = current_pose;

        previous_pose.x = current_odom_pose.pose.pose.position.x;
        previous_pose.y = current_odom_pose.pose.pose.position.y;
        previous_pose.z = current_odom_pose.pose.pose.position.z;
        previous_pose.roll = 0.0;
        previous_pose.pitch = 0.0;
        previous_pose.yaw = current_odom_pose_yaw;  // note-justin  使用IMU的YAW

        offset_x = 0.0;
        offset_y = 0.0;
        offset_z = 0.0;
        offset_yaw = 0.0;

        offset_imu_x = 0.0;
        offset_imu_y = 0.0;
        offset_imu_z = 0.0;
        offset_imu_roll = 0.0;
        offset_imu_pitch = 0.0;
        offset_imu_yaw = 0.0;

        offset_odom_x = 0.0;
        offset_odom_y = 0.0;
        offset_odom_z = 0.0;
        offset_odom_roll = 0.0;
        offset_odom_pitch = 0.0;
        offset_odom_yaw = 0.0;

        offset_imu_odom_x = 0.0;
        offset_imu_odom_y = 0.0;
        offset_imu_odom_z = 0.0;
        offset_imu_odom_roll = 0.0;
        offset_imu_odom_pitch = 0.0;
        offset_imu_odom_yaw = 0.0;

        std_msgs::Bool Trans_flag;
        Trans_flag.data = true;
        wheel_odom_trans_pub.publish(Trans_flag);

        ROS_INFO("======================OUT LIFT========%d=================%f================", waypoint_change_flag,
                 ndt_reliability_data_mean);
        ROS_INFO(
            "ndt_matching.cpp: current_odom_pose.pose.pose.position.x = %f, current_odom_pose.pose.pose.position.y = "
            "%f, current_odom_pose.pose.pose.position.z = %f, imu_yaw = %f, current_odom_pose_yaw = %f",
            current_odom_pose.pose.pose.position.x, current_odom_pose.pose.pose.position.y,
            current_odom_pose.pose.pose.position.z, imu_yaw, current_odom_pose_yaw);

        ROS_INFO("======================================================\n\n");

        if (ndt_reliability_data_mean <= 50)
        {
          jump_lift_localization_flag = false;
          ROS_INFO("======================FINALLY OUT LIFT========%d=================%f================",
                   waypoint_change_flag, ndt_reliability_data_mean);
          ROS_INFO(
              "ndt_matching.cpp: current_odom_pose.pose.pose.position.x = %f, current_odom_pose.pose.pose.position.y = "
              "%f, current_odom_pose.pose.pose.position.z = %f, imu_yaw = %f, current_odom_pose_yaw = %f",
              current_odom_pose.pose.pose.position.x, current_odom_pose.pose.pose.position.y,
              current_odom_pose.pose.pose.position.z, imu_yaw, current_odom_pose_yaw);

          ROS_INFO("======================================================\n\n");
        }

        std_msgs::Bool into_lift_flag;
        into_lift_flag.data = false;
        into_lift_trans_pub.publish(into_lift_flag);
      }
    }

    /***********************************************************************/

    /**********************************************************************/

    /***********************************************************************/
    if (inital_pose_flag == true)
    {
      predict_pose.x = previous_pose.x + offset_x;
      predict_pose.y = previous_pose.y + offset_y;
      predict_pose.z = previous_pose.z + offset_z;
      predict_pose.roll = previous_pose.roll;
      predict_pose.pitch = previous_pose.pitch;
      predict_pose.yaw = previous_pose.yaw;
      inital_pose_flag = false;
      predict_pose_for_ndt = predict_pose;
    }

    /**********************************************************************/

    if (_use_imu == true && _use_odom == true)
    {
      // ROS_ERROR("ndt_matching.cpp:: _use_imu = %d _use_odom = %d", _use_imu, _use_odom);
    }
    if (_use_imu == true && _use_odom == false)
    {
      imu_calc(current_scan_time);
      // ROS_ERROR("ndt_matching.cpp:: _use_imu = %d _use_odom = %d", _use_imu, _use_odom);
    }

    if (_use_imu == true && _use_odom == true)
    {
      predict_pose_for_ndt = predict_pose_imu_odom;
      // ROS_ERROR("ndt_matching.cpp:: _use_imu = %d _use_odom = %d, predict_pose_imu_odom.x = %f
      // predict_pose_imu_odom.y = %f  predict_pose_imu_odom.z = %f", _use_imu, _use_odom, predict_pose_imu_odom.x,
      // predict_pose_imu_odom.y, predict_pose_imu_odom.z);
    }
    else if (_use_imu == true && _use_odom == false)
    {
      predict_pose_for_ndt = predict_pose_imu;
      // ROS_ERROR("ndt_matching.cpp:: _use_imu = %d _use_odom = %d, predict_pose_imu.x = %f predict_pose_imu.y = %f
      // predict_pose_imu.z = %f", _use_imu, _use_odom, predict_pose_imu.x, predict_pose_imu.y, predict_pose_imu.z);
    }
    else if (_use_imu == false && _use_odom == true)
    {
      predict_pose_for_ndt = predict_pose_odom;
      // ROS_ERROR("ndt_matching.cpp:: _use_imu = %d _use_odom = %d, predict_pose_odom.x = %f predict_pose_odom.y = %f
      // predict_pose_odom.z = %f", _use_imu, _use_odom, predict_pose_odom.x, predict_pose_odom.y, predict_pose_odom.z);
    }
    else
    {
      predict_pose_for_ndt = predict_pose;
      // ROS_ERROR("ndt_matching.cpp:: _use_imu = %d _use_odom = %d, predict_pose.x = %f predict_pose.y = %f
      // predict_pose.z = %f", _use_imu, _use_odom, predict_pose.x, predict_pose.y, predict_pose.z);
    }

    Eigen::Translation3f init_translation(predict_pose_for_ndt.x, predict_pose_for_ndt.y, predict_pose_for_ndt.z);
    Eigen::AngleAxisf init_rotation_x(predict_pose_for_ndt.roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf init_rotation_y(predict_pose_for_ndt.pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf init_rotation_z(predict_pose_for_ndt.yaw, Eigen::Vector3f::UnitZ());
    Eigen::Matrix4f init_guess = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x) * tf_btol;

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);

#ifdef CUDA_FOUND
    if (_use_gpu == true)
    {
      align_start = std::chrono::system_clock::now();
      gpu_ndt_ptr->align(init_guess);
      align_end = std::chrono::system_clock::now();

      has_converged = gpu_ndt_ptr->hasConverged();

      t = gpu_ndt_ptr->getFinalTransformation();
      iteration = gpu_ndt_ptr->getFinalNumIteration();

      getFitnessScore_start = std::chrono::system_clock::now();
      fitness_score = gpu_ndt_ptr->getFitnessScore();
      getFitnessScore_end = std::chrono::system_clock::now();

      trans_probability = gpu_ndt_ptr->getTransformationProbability();
    }
    else
#endif
        if (_use_fast_pcl)
    {
      align_start = std::chrono::system_clock::now();
      cpu_ndt.align(init_guess);
      align_end = std::chrono::system_clock::now();

      has_converged = cpu_ndt.hasConverged();

      t = cpu_ndt.getFinalTransformation();
      iteration = cpu_ndt.getFinalNumIteration();

      getFitnessScore_start = std::chrono::system_clock::now();
      fitness_score = cpu_ndt.getFitnessScore();
      getFitnessScore_end = std::chrono::system_clock::now();

      trans_probability = cpu_ndt.getTransformationProbability();
    }
    else
    {
      align_start = std::chrono::system_clock::now();
#ifdef USE_FAST_PCL
      ndt.omp_align(*output_cloud, init_guess);
#else
      ndt.align(*output_cloud, init_guess);
#endif
      align_end = std::chrono::system_clock::now();

      has_converged = ndt.hasConverged();

      t = ndt.getFinalTransformation();
      iteration = ndt.getFinalNumIteration();

      getFitnessScore_start = std::chrono::system_clock::now();
#ifdef USE_FAST_PCL
      fitness_score = ndt.omp_getFitnessScore();
#else
      fitness_score = ndt.getFitnessScore();
#endif
      getFitnessScore_end = std::chrono::system_clock::now();

      trans_probability = ndt.getTransformationProbability();

      Eigen::Matrix<double, 6, 1> score_gradient;  // note-justin
      Eigen::Matrix<double, 6, 6> hessian;         // note- justin
      Eigen::Matrix<double, 6, 1> p;               // note- justin
      bool compute_hessian = true;
      Eigen::Vector3d x_trans;
      Eigen::Vector3d c_inv;

      Eigen::Matrix<double, 4, 1> final_transformation_;

      ndt.getFitnessScore();
    }

    align_time = std::chrono::duration_cast<std::chrono::microseconds>(align_end - align_start).count() / 1000.0;

    t2 = t * tf_btol.inverse();

    getFitnessScore_time =
        std::chrono::duration_cast<std::chrono::microseconds>(getFitnessScore_end - getFitnessScore_start).count() /
        1000.0;

    pthread_mutex_unlock(&mutex);

    tf::Matrix3x3 mat_l;  // localizer
    mat_l.setValue(static_cast<double>(t(0, 0)), static_cast<double>(t(0, 1)), static_cast<double>(t(0, 2)),
                   static_cast<double>(t(1, 0)), static_cast<double>(t(1, 1)), static_cast<double>(t(1, 2)),
                   static_cast<double>(t(2, 0)), static_cast<double>(t(2, 1)), static_cast<double>(t(2, 2)));

    // Update localizer_pose
    localizer_pose.x = t(0, 3);
    localizer_pose.y = t(1, 3);
    localizer_pose.z = t(2, 3);
    mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw, 1);

    tf::Matrix3x3 mat_b;  // base_link
    mat_b.setValue(static_cast<double>(t2(0, 0)), static_cast<double>(t2(0, 1)), static_cast<double>(t2(0, 2)),
                   static_cast<double>(t2(1, 0)), static_cast<double>(t2(1, 1)), static_cast<double>(t2(1, 2)),
                   static_cast<double>(t2(2, 0)), static_cast<double>(t2(2, 1)), static_cast<double>(t2(2, 2)));

    // Update ndt_pose
    ndt_pose.x = t2(0, 3);
    ndt_pose.y = t2(1, 3);
    ndt_pose.z = t2(2, 3);
    mat_b.getRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw, 1);

    // Calculate the difference between ndt_pose and predict_pose
    // Justin 这里会计算ndt优化得出的pose和之前predict的pose之间的差值，如果差值很小，也会考虑predict的pose
    predict_pose_error = sqrt((ndt_pose.x - predict_pose_for_ndt.x) * (ndt_pose.x - predict_pose_for_ndt.x) +
                              (ndt_pose.y - predict_pose_for_ndt.y) * (ndt_pose.y - predict_pose_for_ndt.y) +
                              (ndt_pose.z - predict_pose_for_ndt.z) * (ndt_pose.z - predict_pose_for_ndt.z));

    if (predict_pose_error <= PREDICT_POSE_THRESHOLD)
    {
      use_predict_pose = 0;
    }
    else
    {
      use_predict_pose = 1;
    }
    use_predict_pose = 0;

    if (use_predict_pose == 0)
    {
      current_pose.x = ndt_pose.x;
      current_pose.y = ndt_pose.y;
      current_pose.z = ndt_pose.z;
      current_pose.roll = ndt_pose.roll;
      current_pose.pitch = ndt_pose.pitch;
      current_pose.yaw = ndt_pose.yaw;
    }
    else
    {
      current_pose.x = predict_pose_for_ndt.x;
      current_pose.y = predict_pose_for_ndt.y;
      current_pose.z = predict_pose_for_ndt.z;
      current_pose.roll = predict_pose_for_ndt.roll;
      current_pose.pitch = predict_pose_for_ndt.pitch;
      current_pose.yaw = predict_pose_for_ndt.yaw;
    }

    // Compute the velocity and acceleration
    scan_duration = current_scan_time - previous_scan_time;
    double secs = scan_duration.toSec();
    diff_x = current_pose.x - previous_pose.x;
    diff_y = current_pose.y - previous_pose.y;
    diff_z = current_pose.z - previous_pose.z;
    diff_yaw = current_pose.yaw - previous_pose.yaw;
    diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

    current_velocity = diff / secs;
    if (current_velocity > 1.0)
    {
      current_velocity = 1.0;
    }
    current_velocity_x = diff_x / secs;
    current_velocity_y = diff_y / secs;
    current_velocity_z = diff_z / secs;
    angular_velocity = diff_yaw / secs;

    current_pose_imu.x = current_pose.x;
    current_pose_imu.y = current_pose.y;
    current_pose_imu.z = current_pose.z;
    current_pose_imu.roll = current_pose.roll;
    current_pose_imu.pitch = current_pose.pitch;
    current_pose_imu.yaw = current_pose.yaw;

    current_velocity_imu_x = current_velocity_x;
    current_velocity_imu_y = current_velocity_y;
    current_velocity_imu_z = current_velocity_z;

    current_pose_odom.x = current_pose.x;
    current_pose_odom.y = current_pose.y;
    current_pose_odom.z = current_pose.z;
    current_pose_odom.roll = current_pose.roll;
    current_pose_odom.pitch = current_pose.pitch;
    current_pose_odom.yaw = current_pose.yaw;

    current_pose_imu_odom.x = current_pose.x;
    current_pose_imu_odom.y = current_pose.y;
    current_pose_imu_odom.z = current_pose.z;
    current_pose_imu_odom.roll = current_pose.roll;
    current_pose_imu_odom.pitch = current_pose.pitch;
    current_pose_imu_odom.yaw = current_pose.yaw;

    current_velocity_smooth = (current_velocity + previous_velocity + previous_previous_velocity) / 3.0;
    if (current_velocity_smooth < 0.2)
    {
      current_velocity_smooth = 0.0;
    }

    current_accel = (current_velocity - previous_velocity) / secs;
    current_accel_x = (current_velocity_x - previous_velocity_x) / secs;
    current_accel_y = (current_velocity_y - previous_velocity_y) / secs;
    current_accel_z = (current_velocity_z - previous_velocity_z) / secs;

    estimated_vel_mps.data = current_velocity;
    estimated_vel_kmph.data = current_velocity * 3.6;

    estimated_vel_mps_pub.publish(estimated_vel_mps);
    estimated_vel_kmph_pub.publish(estimated_vel_kmph);

    // Set values for publishing pose
    predict_q.setRPY(predict_pose.roll, predict_pose.pitch, predict_pose.yaw);
    if (_use_local_transform == true)
    {
      tf::Vector3 v(predict_pose.x, predict_pose.y, predict_pose.z);
      tf::Transform transform(predict_q, v);
      predict_pose_msg.header.frame_id = "map";
      predict_pose_msg.header.stamp = current_scan_time;
      predict_pose_msg.pose.position.x = (local_transform * transform).getOrigin().getX();
      predict_pose_msg.pose.position.y = (local_transform * transform).getOrigin().getY();
      predict_pose_msg.pose.position.z = (local_transform * transform).getOrigin().getZ();
      predict_pose_msg.pose.orientation.x = (local_transform * transform).getRotation().x();
      predict_pose_msg.pose.orientation.y = (local_transform * transform).getRotation().y();
      predict_pose_msg.pose.orientation.z = (local_transform * transform).getRotation().z();
      predict_pose_msg.pose.orientation.w = (local_transform * transform).getRotation().w();
    }
    else
    {
      predict_pose_msg.header.frame_id = "map";
      predict_pose_msg.header.stamp = current_scan_time;
      predict_pose_msg.pose.position.x = predict_pose.x;
      predict_pose_msg.pose.position.y = predict_pose.y;
      predict_pose_msg.pose.position.z = predict_pose.z;
      predict_pose_msg.pose.orientation.x = predict_q.x();
      predict_pose_msg.pose.orientation.y = predict_q.y();
      predict_pose_msg.pose.orientation.z = predict_q.z();
      predict_pose_msg.pose.orientation.w = predict_q.w();
    }

    ndt_q.setRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw);
    if (_use_local_transform == true)
    {
      tf::Vector3 v(ndt_pose.x, ndt_pose.y, ndt_pose.z);
      tf::Transform transform(ndt_q, v);
      ndt_pose_msg.header.frame_id = "map";
      ndt_pose_msg.header.stamp = current_scan_time;
      ndt_pose_msg.pose.position.x = (local_transform * transform).getOrigin().getX();
      ndt_pose_msg.pose.position.y = (local_transform * transform).getOrigin().getY();
      ndt_pose_msg.pose.position.z = (local_transform * transform).getOrigin().getZ();
      ndt_pose_msg.pose.orientation.x = (local_transform * transform).getRotation().x();
      ndt_pose_msg.pose.orientation.y = (local_transform * transform).getRotation().y();
      ndt_pose_msg.pose.orientation.z = (local_transform * transform).getRotation().z();
      ndt_pose_msg.pose.orientation.w = (local_transform * transform).getRotation().w();
    }
    else
    {
      ndt_pose_msg.header.frame_id = "map";
      ndt_pose_msg.header.stamp = current_scan_time;
      ndt_pose_msg.pose.position.x = ndt_pose.x;
      ndt_pose_msg.pose.position.y = ndt_pose.y;
      ndt_pose_msg.pose.position.z = ndt_pose.z;
      ndt_pose_msg.pose.orientation.x = ndt_q.x();
      ndt_pose_msg.pose.orientation.y = ndt_q.y();
      ndt_pose_msg.pose.orientation.z = ndt_q.z();
      ndt_pose_msg.pose.orientation.w = ndt_q.w();
    }

    if (stay_lift_localization_flag == true)  // 4->5 note-justin  进入电梯的时候
    {
      current_pose.x = current_odom_pose.pose.pose.position.x;
      current_pose.y = current_odom_pose.pose.pose.position.y;
      current_pose.z = current_odom_pose.pose.pose.position.z;
      current_pose.roll = 0.0;
      current_pose.pitch = 0.0;
      current_pose.yaw = current_odom_pose_yaw;  // note-justin  使用IMU的YAW

      current_pose_imu = current_pose_odom = current_pose_imu_odom = current_pose;

      previous_pose.x = current_odom_pose.pose.pose.position.x;
      previous_pose.y = current_odom_pose.pose.pose.position.y;
      previous_pose.z = current_odom_pose.pose.pose.position.z;
      previous_pose.roll = 0.0;
      previous_pose.pitch = 0.0;
      previous_pose.yaw = current_odom_pose_yaw;  // note-justin  使用IMU的YAW

      offset_x = 0.0;
      offset_y = 0.0;
      offset_z = 0.0;
      offset_yaw = 0.0;

      offset_imu_x = 0.0;
      offset_imu_y = 0.0;
      offset_imu_z = 0.0;
      offset_imu_roll = 0.0;
      offset_imu_pitch = 0.0;
      offset_imu_yaw = 0.0;

      offset_odom_x = 0.0;
      offset_odom_y = 0.0;
      offset_odom_z = 0.0;
      offset_odom_roll = 0.0;
      offset_odom_pitch = 0.0;
      offset_odom_yaw = 0.0;

      offset_imu_odom_x = 0.0;
      offset_imu_odom_y = 0.0;
      offset_imu_odom_z = 0.0;
      offset_imu_odom_roll = 0.0;
      offset_imu_odom_pitch = 0.0;
      offset_imu_odom_yaw = 0.0;

      ndt_pose_msg.header.frame_id = "map";
      ndt_pose_msg.header.stamp = current_scan_time;
      ndt_pose_msg.pose.position.x = current_odom_pose.pose.pose.position.x;
      ndt_pose_msg.pose.position.y = current_odom_pose.pose.pose.position.y;
      ndt_pose_msg.pose.position.z = current_odom_pose.pose.pose.position.z;
      ndt_pose_msg.pose.orientation = current_odom_pose.pose.pose.orientation;
      // ndt_pose_msg.pose.orientation = tf::createQuaternionMsgFromYaw(current_odom_pose_yaw); // note-justin
      // YAW使用IMU的数据
      ROS_INFO("=====================INTO LIFT=======waypoint_change_flag = %d===%f== ", waypoint_change_flag,
               ndt_reliability_data_mean);
      ROS_ERROR("ndt_matching.cpp: current_odom_pose.pose.pose.position.x = %f",
                current_odom_pose.pose.pose.position.x);
      ROS_ERROR("ndt_matching.cpp: current_odom_pose.pose.pose.position.y = %f",
                current_odom_pose.pose.pose.position.y);
      ROS_ERROR("ndt_matching.cpp: current_odom_pose.pose.pose.position.z = %f",
                current_odom_pose.pose.pose.position.z);
      ROS_ERROR("ndt_matching.cpp: imu_yaw = %f current_odom_pose_yaw = %f", imu_yaw, current_odom_pose_yaw);
      ROS_INFO("=====================INTO LIFT========global_closest_waypoint = %d===%f= \n\n", global_closest_waypoint,
               ndt_reliability_data_mean);

      std_msgs::Bool into_lift_flag;
      into_lift_flag.data = true;
      into_lift_trans_pub.publish(into_lift_flag);
    }

    if (ndt_reliability_data_mean >= 12000.0)
    {
      ROS_INFO("\n\n==================================================================================");
      ROS_INFO("ndt_matching.cpp:: START ndt_reliability_data_mean = %f", ndt_reliability_data_mean);
      ROS_INFO("==================================================================================\n\n");
    }

    if (manually_into_environment > 0)
    {
      use_wheel_odom_flag = true;
    }

    current_q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

    localizer_q.setRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw);
    if (_use_local_transform == true)
    {
      tf::Vector3 v(localizer_pose.x, localizer_pose.y, localizer_pose.z);
      tf::Transform transform(localizer_q, v);
      localizer_pose_msg.header.frame_id = "map";
      localizer_pose_msg.header.stamp = current_scan_time;
      localizer_pose_msg.pose.position.x = (local_transform * transform).getOrigin().getX();
      localizer_pose_msg.pose.position.y = (local_transform * transform).getOrigin().getY();
      localizer_pose_msg.pose.position.z = (local_transform * transform).getOrigin().getZ();
      localizer_pose_msg.pose.orientation.x = (local_transform * transform).getRotation().x();
      localizer_pose_msg.pose.orientation.y = (local_transform * transform).getRotation().y();
      localizer_pose_msg.pose.orientation.z = (local_transform * transform).getRotation().z();
      localizer_pose_msg.pose.orientation.w = (local_transform * transform).getRotation().w();
    }
    else
    {
      localizer_pose_msg.header.frame_id = "map";
      localizer_pose_msg.header.stamp = current_scan_time;
      localizer_pose_msg.pose.position.x = localizer_pose.x;
      localizer_pose_msg.pose.position.y = localizer_pose.y;
      localizer_pose_msg.pose.position.z = localizer_pose.z;
      localizer_pose_msg.pose.orientation.x = localizer_q.x();
      localizer_pose_msg.pose.orientation.y = localizer_q.y();
      localizer_pose_msg.pose.orientation.z = localizer_q.z();
      localizer_pose_msg.pose.orientation.w = localizer_q.w();
    }

    predict_pose_pub.publish(predict_pose_msg);

    float ndt_yaw = 0.0;
    ndt_yaw = current_pose.yaw;

    ndt_yaw > 0 ? ndt_yaw = ndt_yaw : ndt_yaw = 2 * PI + ndt_yaw;

    std_msgs::Float32 ndt_yaw_pub_msg;
    ndt_yaw_pub_msg.data = ndt_yaw;
    ndt_yaw_pub.publish(ndt_yaw_pub_msg);

    localizer_pose_pub.publish(localizer_pose_msg);

    // Justin current_pose通过TF的方式发布出去 Send TF "base_link" to "map"
    transform.setOrigin(tf::Vector3(current_pose.x, current_pose.y, current_pose.z));
    transform.setRotation(current_q);

    if (_use_local_transform == true)
    {
      br.sendTransform(tf::StampedTransform(local_transform * transform, current_scan_time, "map", "base_link"));
    }
    else
    {
      br.sendTransform(tf::StampedTransform(transform, current_scan_time, "map", "base_link"));
    }

    if (_use_imu)
    {
      // imu_calc(current_scan_time);
      // ROS_ERROR("ndt_matching.cpp:: current_scan_time = %f", current_scan_time);

      tf::Vector3 imu_v(current_pose.x, current_pose.y, current_pose.z);
      tf::Transform transform_imu(imu_q, imu_v);
      br_imu.sendTransform(tf::StampedTransform(transform_imu, current_scan_time, "map", "/imu_orientation"));
    }

    matching_end = std::chrono::system_clock::now();
    exe_time = std::chrono::duration_cast<std::chrono::microseconds>(matching_end - matching_start).count() / 1000.0;
    time_ndt_matching.data = exe_time;
    time_ndt_matching_pub.publish(time_ndt_matching);

    // Set values for /estimate_twist
    estimate_twist_msg.header.stamp = current_scan_time;
    estimate_twist_msg.header.frame_id = "base_link";
    estimate_twist_msg.twist.linear.x = current_velocity;
    estimate_twist_msg.twist.linear.y = 0.0;
    estimate_twist_msg.twist.linear.z = 0.0;
    estimate_twist_msg.twist.angular.x = 0.0;
    estimate_twist_msg.twist.angular.y = 0.0;
    estimate_twist_msg.twist.angular.z = angular_velocity;

    estimate_twist_pub.publish(estimate_twist_msg);

    geometry_msgs::Vector3Stamped estimate_vel_msg;
    estimate_vel_msg.header.stamp = current_scan_time;
    estimate_vel_msg.vector.x = current_velocity;
    estimated_vel_pub.publish(estimate_vel_msg);

    // Set values for /ndt_stat
    ndt_stat_msg.header.stamp = current_scan_time;
    ndt_stat_msg.exe_time = time_ndt_matching.data;

    // ROS_ERROR("ndt_matching.cpp:: exe_time = %f iteration = %f  trans_probability = %f fitness_score = %f", exe_time,
    // iteration, trans_probability, fitness_score);

    if (iteration >= 10000)
    {
      iteration = 10000;  // note-justin   防止溢出
    }
    if (exe_time >= 10000)
    {
      exe_time = 10000;  // note-justin   防止溢出
    }

    if (trans_probability >= 10000)
    {
      trans_probability = 10000;  // note-justin   防止溢出
    }

    if (fitness_score >= 10000)
    {
      fitness_score = 10000;  // note-justin   防止溢出
    }

    // 将数据加入滤波器缓存
    fitness_filter_buf.push_back(fitness_score);

    // 如果缓存中的数据个数超过窗口大小，则移除最早的数据
    if (fitness_filter_buf.size() > fitness_window_size)
    {
      fitness_filter_buf.erase(fitness_filter_buf.begin());
    }

    // 计算滤波后的结果
    double filtered_fitness_score =
        std::accumulate(fitness_filter_buf.begin(), fitness_filter_buf.end(), 0.0) / fitness_filter_buf.size();

    ndt_stat_msg.iteration = iteration;
    ndt_stat_msg.score = filtered_fitness_score;
    ndt_stat_msg.velocity = current_velocity;
    ndt_stat_msg.acceleration = current_accel;
    ndt_stat_msg.use_predict_pose = 0;

    ndt_stat_pub.publish(ndt_stat_msg);
    /* Compute NDT_Reliability */
    ndt_relibility_data = Wa * (exe_time / 100.0) * 100.0 + Wb * (iteration / 100.0) * 100.0 +
                          Wc * ((2.0 - trans_probability) / 2.0) * 100.0 + Wd * 100 * fitness_score;
    // ROS_INFO("ndt_matching.cpp:: exe_time = %f iteration = %d trans_probability = %f fitness_score = %f
    // ndt_relibility_data = %f", exe_time, iteration, trans_probability, fitness_score, ndt_relibility_data);

    ndt_relibility_data_cnt++;

    ndt_reliability.data = ndt_reliability_data_mean;

    // ROS_INFO("ndt_matching.cpp:: ndt_relibility_data = %f fitness_score = %f exe_time = %f iteration = %f
    // trans_probability = %f", ndt_reliability.data, fitness_score, exe_time, iteration, trans_probability);

    ndt_reliability_pub.publish(ndt_reliability);

    fitness.data = filtered_fitness_score;
    ndt_fitness_pub.publish(fitness);

    ndt_cov_msg.header.frame_id = "map";
    ndt_cov_msg.header.stamp = current_scan_time;
    ndt_cov_msg.pose.pose = ndt_pose_msg.pose;

    for (int i = 0; i < 36; ++i)
    {
      ndt_cov_msg.pose.covariance[i] = 0.0;
    }

    float covariance_coefficient = ndt_reliability_data_mean / 100;
    // note-justin 测试 /10 /100 EVO结果

    ndt_cov_msg.pose.covariance = { 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                                    0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                                    0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.06853891945200942 };

    if (ndt_relibility_data >= 20.0)  // note-justin    不允许过大的阀值
    {
      covariance_coefficient = 1000.0;

      // ROS_ERROR("ndt_matching.cpp:: ndt_relibility_data = %f ndt_cov_msg.pose.pose.orientation.w = %f",
      // ndt_relibility_data, ndt_cov_msg.pose.pose.orientation.w);

      // ROS_ERROR("ndt_matching.cpp:: pose.position.x = %f store_pose.x = %f", ndt_cov_msg.pose.pose.position.x,
      // store_pose.x); ROS_ERROR("ndt_matching.cpp:: pose.position.y = %f store_pose.y = %f",
      // ndt_cov_msg.pose.pose.position.y, store_pose.y); ROS_ERROR("ndt_matching.cpp:: pose.position.z = %f
      // store_pose.z = %f", ndt_cov_msg.pose.pose.position.z, store_pose.z);

      ndt_cov_msg.pose.pose.position.z = store_pose.z;  // note-justin
    }
    else
    {
      covariance_coefficient = 1.0;
    }

    if (waypoint_change_flag == 5)  // note-justin    不允许过大的阀值
    {
      covariance_coefficient = 1000.0;
      // tf::Quaternion quaternion = tf::createQuaternionFromRPY(0, 0, imu_yaw);
      // quaternionTFToMsg(quaternion, ndt_cov_msg.pose.pose.orientation);
      // ROS_ERROR("ndt_matching.cpp:: ndt_relibility_data = %f ndt_cov_msg.pose.pose.orientation.w = %f",
      // ndt_relibility_data, ndt_cov_msg.pose.pose.orientation.w); ndt_cov_msg.pose.pose.position.z = store_pose.z; //
      // note-justin
    }
    else
    {
      covariance_coefficient = 1.0;
    }

    ndt_cov_msg.pose.covariance[0] = 0.039500 * covariance_coefficient;
    ndt_cov_msg.pose.covariance[1] = -0.009644 * covariance_coefficient;
    ndt_cov_msg.pose.covariance[5] = 0.000475 * covariance_coefficient;
    ndt_cov_msg.pose.covariance[6] = -0.009644 * covariance_coefficient;
    ndt_cov_msg.pose.covariance[7] = 0.008010 * covariance_coefficient;
    ndt_cov_msg.pose.covariance[11] = 0.001691 * covariance_coefficient;
    ndt_cov_msg.pose.covariance[30] = 0.000475 * covariance_coefficient;
    ndt_cov_msg.pose.covariance[31] = 0.001691 * covariance_coefficient;
    ndt_cov_msg.pose.covariance[35] = 0.008162 * covariance_coefficient;

    ndt_cov_pub.publish(ndt_cov_msg);

    if (false)
    {
      ROS_ERROR(
          "ndt_matching.cpp:: ndt_pose_msg.pose.position.x = %f ndt_pose_msg.pose.position.y = %f "
          "ndt_pose_msg.pose.position.z = %f",
          ndt_pose_msg.pose.position.x, ndt_pose_msg.pose.position.y, ndt_pose_msg.pose.position.z);
      ROS_ERROR(
          "ndt_matching.cpp:: pre_ndt_pose_msg.pose.position.x = %f pre_ndt_pose_msg.pose.position.y = %f "
          "pre_ndt_pose_msg.pose.position.z = %f",
          pre_ndt_pose_msg.pose.position.x, pre_ndt_pose_msg.pose.position.y, pre_ndt_pose_msg.pose.position.z);
      ndt_pose_msg.pose = pre_ndt_pose_msg.pose;
      ndt_pose_msg.pose.orientation = pre_ndt_pose_msg.pose.orientation;
      previous_pose.x = pre_ndt_pose_msg.pose.position.x;
      previous_pose.y = pre_ndt_pose_msg.pose.position.y;
      previous_pose.z = pre_ndt_pose_msg.pose.position.z;

      tf::Quaternion previous_pose_orientation;
      tf::quaternionMsgToTF(pre_ndt_pose_msg.pose.orientation, previous_pose_orientation);
      tf::Matrix3x3(previous_pose_orientation).getRPY(previous_pose_roll, previous_pose_pitch, previous_pose_yaw);

      previous_pose.roll = 0.0;
      previous_pose.pitch = 0.0;
      previous_pose.yaw = previous_pose_yaw;

      ROS_ERROR("ndt_matching.cpp:: previous_pose_roll = %f previous_pose_pitch = %f previous_pose_yaw = %f",
                previous_pose_roll, previous_pose_pitch, previous_pose_yaw);
    }

    ndt_pose_pub.publish(ndt_pose_msg);

    pre_ndt_pose_msg.header.frame_id = "map";
    pre_ndt_pose_msg.header.stamp = current_scan_time;
    pre_ndt_pose_msg.pose.position.x = ndt_pose_msg.pose.position.x;
    pre_ndt_pose_msg.pose.position.y = ndt_pose_msg.pose.position.y;
    pre_ndt_pose_msg.pose.position.z = ndt_pose_msg.pose.position.z;
    pre_ndt_pose_msg.pose.orientation = ndt_pose_msg.pose.orientation;

    if (_offset == "linear")
    {
      offset_x = abs(diff_x) < 1.5 ? diff_x : offset_imu_x;  // note-justin;
      offset_x = abs(offset_x) < 1.5 ? offset_x : 0.0;

      offset_y = abs(diff_y) < 1.5 ? diff_y : offset_imu_y;  // diff_x;
      offset_y = abs(offset_y) < 1.5 ? offset_y : 0.0;

      offset_z = abs(diff_z) < 1.5 ? diff_z : offset_imu_z;  // diff_x;
      offset_z = abs(offset_z) < 1.5 ? offset_z : 0.0;

      offset_yaw =
          abs(diff_yaw) < 1.5 ? diff_yaw : offset_imu_yaw;  // note-justin  offset_imu_yaw diff_yaw;//  abs(diff_yaw) <
                                                            // 0.03 ? diff_yaw : offset_imu_yaw;
      offset_yaw = abs(offset_yaw) < 1.5 ? offset_yaw : 0.0;
    }
    else if (_offset == "quadratic")
    {
      offset_x = (current_velocity_x + current_accel_x * secs) * secs;
      offset_y = (current_velocity_y + current_accel_y * secs) * secs;
      offset_z = diff_z;
      // offset_yaw = abs(diff_yaw) < 0.03 ? diff_yaw : offset_imu_yaw;
      offset_yaw = offset_imu_yaw;
      // ROS_ERROR("ndt_matching.cpp:: offset_imu_yaw = %f diff_yaw = %f offset_yaw = %f _offset = quadratic",
      // offset_imu_yaw, diff_yaw, offset_yaw);
    }
    else if (_offset == "zero")
    {
      offset_x = 0.0;
      offset_y = 0.0;
      offset_z = 0.0;
      offset_yaw = abs(diff_yaw) < 0.03 ? diff_yaw : offset_imu_yaw;
      // ROS_ERROR("ndt_matching.cpp:: offset_imu_yaw = %f diff_yaw = %f _offset = zero", offset_imu_yaw, diff_yaw);
    }

    offset_imu_x = 0.0;
    offset_imu_y = 0.0;
    offset_imu_z = 0.0;
    offset_imu_roll = 0.0;
    offset_imu_pitch = 0.0;
    offset_imu_yaw = 0.0;

    offset_odom_x = 0.0;
    offset_odom_y = 0.0;
    offset_odom_z = 0.0;
    offset_odom_roll = 0.0;
    offset_odom_pitch = 0.0;
    offset_odom_yaw = 0.0;

    offset_imu_odom_x = 0.0;
    offset_imu_odom_y = 0.0;
    offset_imu_odom_z = 0.0;
    offset_imu_odom_roll = 0.0;
    offset_imu_odom_pitch = 0.0;
    offset_imu_odom_yaw = 0.0;

    // Update previous_***
    previous_pose.x = current_pose.x;
    previous_pose.y = current_pose.y;
    previous_pose.z = current_pose.z;
    previous_pose.roll = current_pose.roll;
    previous_pose.pitch = current_pose.pitch;
    previous_pose.yaw = current_pose.yaw;

    previous_natural_pose = current_natural_pose;

    previous_scan_time.sec = current_scan_time.sec;
    previous_scan_time.nsec = current_scan_time.nsec;

    previous_previous_velocity = previous_velocity;
    previous_velocity = current_velocity;
    previous_velocity_x = current_velocity_x;
    previous_velocity_y = current_velocity_y;
    previous_velocity_z = current_velocity_z;
    previous_accel = current_accel;

    previous_estimated_vel_kmph.data = estimated_vel_kmph.data;

    matching_status();
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "pointcloud_odom");
  pthread_mutex_init(&mutex, NULL);

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // Geting parameters
  private_nh.getParam("use_gnss", _use_gnss);
  private_nh.getParam("queue_size", _queue_size);
  private_nh.getParam("offset", _offset);
  private_nh.getParam("use_openmp", _use_openmp);
  private_nh.getParam("use_gpu", _use_gpu);
  private_nh.getParam("use_fast_pcl", _use_fast_pcl);
  private_nh.getParam("get_height", _get_height);
  private_nh.getParam("use_local_transform", _use_local_transform);
  private_nh.getParam("use_imu", _use_imu);
  private_nh.getParam("use_odom", _use_odom);
  private_nh.getParam("imu_upside_down", _imu_upside_down);
  private_nh.getParam("imu_topic", _imu_topic);
  private_nh.getParam("indoor", indoor_);

#if defined(CUDA_FOUND) && defined(USE_FAST_PCL)
  if (_use_gpu == true && _use_openmp == true)
  {
    std::cout << "use_gpu and use_openmp are exclusive. Set use_gpu true and use_openmp false." << std::endl;
    _use_openmp = false;
  }
#endif

  if (nh.getParam("localizer", _localizer) == false)
  {
    std::cout << "ndt_matching.cpp:: localizer is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_x", _tf_x) == false)
  {
    std::cout << "ndt_matching.cpp:: tf_x is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_y", _tf_y) == false)
  {
    std::cout << "ndt_matching.cpp:: tf_y is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_z", _tf_z) == false)
  {
    std::cout << "ndt_matching.cpp:: tf_z is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_roll", _tf_roll) == false)
  {
    std::cout << "ndt_matching.cpp:: tf_roll is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_pitch", _tf_pitch) == false)
  {
    std::cout << "ndt_matching.cpp:: tf_pitch is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_yaw", _tf_yaw) == false)
  {
    std::cout << "ndt_matching.cpp:: tf_yaw is not set." << std::endl;
    return 1;
  }

  std::cout << "----------------------------ndt_matching.cpp-------------------------------------" << std::endl;
  std::cout << "Log file: " << filename << std::endl;
  std::cout << "use_gnss: " << _use_gnss << std::endl;
  std::cout << "queue_size: " << _queue_size << std::endl;
  std::cout << "offset: " << _offset << std::endl;
  std::cout << "use_gpu: " << _use_gpu << std::endl;
  std::cout << "use_openmp: " << _use_openmp << std::endl;
  std::cout << "use_fast_pcl: " << _use_fast_pcl << std::endl;
  std::cout << "get_height: " << _get_height << std::endl;
  std::cout << "use_local_transform: " << _use_local_transform << std::endl;
  std::cout << "use_imu: " << _use_imu << std::endl;
  std::cout << "use_odom: " << _use_odom << std::endl;
  std::cout << "imu_upside_down: " << _imu_upside_down << std::endl;
  std::cout << "localizer: " << _localizer << std::endl;
  std::cout << "imu_topic: " << _imu_topic << std::endl;
  std::cout << "(tf_x,tf_y,tf_z,tf_roll,tf_pitch,tf_yaw): (" << _tf_x << ", " << _tf_y << ", " << _tf_z << ", "
            << _tf_roll << ", " << _tf_pitch << ", " << _tf_yaw << ")" << std::endl;
  std::cout << "--------------------------------ndt_matching.cpp---------------------------------" << std::endl;

  Eigen::Translation3f tl_btol(_tf_x, _tf_y, _tf_z);
  Eigen::AngleAxisf rot_x_btol(_tf_roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf rot_y_btol(_tf_pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z_btol(_tf_yaw, Eigen::Vector3f::UnitZ());
  tf_btol = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();

  // Publishers
  sub_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/sub_map", 1);

  ndt_pose_pub = nh.advertise<geometry_msgs::PoseStamped> > ("/ndt_pose", 10);
  natural_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/natual_pose", 1000);
  ndt_cov_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/ndt_cov_pose", 10);
  localizer_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/localizer_pose", 1000);
  estimate_twist_pub = nh.advertise<geometry_msgs::TwistStamped>("/estimate_twist", 1000);
  estimated_vel_mps_pub = nh.advertise<std_msgs::Float32>("/estimated_vel_mps", 1000);
  estimated_vel_kmph_pub = nh.advertise<std_msgs::Float32>("/estimated_vel_kmph", 1000);
  estimated_offset_imu_y_pub = nh.advertise<std_msgs::Float32>("/offset_imu_y", 1000);

  estimated_vel_pub = nh.advertise<geometry_msgs::Vector3Stamped>("/estimated_vel", 1000);
  time_ndt_matching_pub = nh.advertise<std_msgs::Float32>("/time_ndt_matching", 1000);
  ndt_stat_pub = nh.advertise<autoware_msgs::ndt_stat>("/ndt_stat", 1000, true);
  ndt_reliability_pub = nh.advertise<std_msgs::Float32>("/ndt_reliability", 100);
  ndt_fitness_pub = nh.advertise<std_msgs::Float32>("/fitness_score", 100);
  lost_path_point_pub = nh.advertise<std_msgs::Int32>("/lost_path_point", 1);

  relocal_flag_pub = nh.advertise<std_msgs::Bool>("/stop_flag_relocalize", 100);
  wheel_odom_trans_pub = nh.advertise<std_msgs::Bool>("/wheel_odom_only", 100);
  into_lift_trans_pub = nh.advertise<std_msgs::Bool>("/into_lift_flag", 100);

  ndt_yaw_pub = nh.advertise<std_msgs::Float32>("ndt_yaw", 10);
  sound_pub_ = nh.advertise<std_msgs::String>("/sound_player", 10, true);
  initialpose_pub_ = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/initialpose", 1, true);

  // Subscribers
  ros::Subscriber gnss_sub = nh.subscribe("/gnss_pose", 10, gnss_callback);
  ros::Subscriber ekf_sub = nh.subscribe("/wheel_odom", 1000, wheel_odom_callback);
  ros::Subscriber map_sub = nh.subscribe("/points_map", 10, map_callback);
  ros::Subscriber points_sub = nh.subscribe("/filtered_points", _queue_size, points_callback);

  ros::Subscriber param_sub = nh.subscribe("/config/ndt", 10, param_callback);
  ros::Subscriber initialpose_sub = nh.subscribe("/initialpose", 1000, initialpose_callback);
  ros::Subscriber lift_pose_sub =
      nh.subscribe("/global_waypoint_change_flag", 1000, stay_lift_localization_flagCallback);
  ros::Subscriber ndt_pose_to_odom_sub = nh.subscribe("/ndt_pose_to_odom", 1000, go_lift_localization_flagCallback);
  ros::Subscriber start_flag_subscriber = nh.subscribe("/start_flag", 1, StartFlagCallback);
  ros::Subscriber global_closest_waypoint_sub = nh.subscribe("/global_closest_waypoint", 1, globalclosestFlagCallback);
  ros::Subscriber manually_into_environment_sub =
      nh.subscribe("/manually_into_environment", 1, manuallyEnvironCallback);
  ros::Subscriber lost_path_point_sub = nh.subscribe("/lost_path_point", 10, lostPathPointCallback);

  ros::Subscriber use_wheel_odom_sub = nh.subscribe("/use_wheel_odom", _queue_size * 10, use_wheel_odom_callback);
  ros::Subscriber waypoint_change_flag_subscriber =
      nh.subscribe("global_waypoint_change_flag", 10, waypointchangeflagCallback);

  ros::spin();

  return 0;
}
