#include "zr_common.h"

#define PI (3.1415926)
static int map_loaded = 0;
static int loss_cnt = 0;
static double fitness_score_threshold;
static double fitness, map_voxel_size, scan_voxel_size, sub_map_size;

static bool need_initial = true;
static bool use_ndt = true;
static bool wheel_odom_only_flag = false;
static bool last_wheel_odom_only_flag = false;
static bool exit_initial_thread = false;

static std::string localization_status = "fail";
static std::string current_map_name = "";
static std::string status_file_dir = "/home/justin/zhenrobot/.status/";
static std::string map_file_dir = "/home/justin/zhenrobot/map/";

Eigen::Matrix4d T_odom_to_map, T_wheel_odom_to_map, T_base_to_map, T_base_to_odom, T_base_to_wheel_odom;

Eigen::Vector2d Twist_in_base_link;

pcl::PointCloud<pcl::PointXYZI>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZI>),
    sub_map(new pcl::PointCloud<pcl::PointXYZI>), keyframes_pcl(new pcl::PointCloud<pcl::PointXYZI>),
    cur_scan_in_odom(new pcl::PointCloud<pcl::PointXYZI>), cur_keypoints_in_odom(new pcl::PointCloud<pcl::PointXYZI>);

static geometry_msgs::PoseStamped current_pose;
static geometry_msgs::PoseWithCovarianceStamped ndt_cov_msg;

static ros::Time cur_scan_stamp, points_odom_stamp, wheel_odom_stamp;
static ros::Time pre_scan_time;
static ros::Duration scan_duration;

static ros::Publisher points_map_pub, current_cov_pose_pub, sub_map_pub, current_pose_pub, fitness_pub,
    reset_wheel_odom_pub, reset_odom_pub, sound_pub, relocal_flag_pub, current_velocity_pub, cur_scan_pub,
    cur_keypoints_pub, keyframes_pcl_pub, localization_status_pub;

ZRParameters zr_param;

struct keyframe
{
  keyframe() : feature_points_ptr(new pcl::PointCloud<pcl::PointXYZI>()){};
  ros::Time timestamp;
  pcl::PointCloud<pcl::PointXYZI>::Ptr feature_points_ptr;
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
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_ptr;
  Eigen::Matrix4d trans;
  double fitness;
  matching_result() : pcl_ptr(new pcl::PointCloud<pcl::PointXYZI>()){};
};

static std::deque<keyframe> keyframes;
static std::mutex mutex;
static std::thread reg_thread, initial_thread;
static std::chrono::time_point<std::chrono::system_clock> matching_start, matching_end;

void reset_state()
{
  need_initial = true;
  loss_cnt = 0;
  T_odom_to_map = Eigen::Matrix4d::Identity();
  T_base_to_odom = Eigen::Matrix4d::Identity();
  T_base_to_map = Eigen::Matrix4d::Identity();
  T_base_to_wheel_odom = Eigen::Matrix4d::Identity();
  T_wheel_odom_to_map = Eigen::Matrix4d::Identity();
  Twist_in_base_link << 0.0, 0.0;
  fitness_score_threshold = use_ndt ? 0.6 : 0.4;
  keyframes.clear();
  ROS_INFO("已初始化状态变量");
}

bool is_loss()
{
  bool loss = false;
  localization_status = "success";
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
    localization_status = "fail";
    reset_state();
    ROS_ERROR("定位丢失");
  }
  return loss;
}


void pub_status(const ros::TimerEvent& evt)
{
  is_loss();
  if (wheel_odom_only_flag)
  {
    T_base_to_map = T_wheel_odom_to_map * T_base_to_wheel_odom * se3_inverse(zr_param.T_enc_to_base);
  }
  else
  {
    T_base_to_map = T_odom_to_map * T_base_to_odom * se3_inverse(zr_param.T_imu_to_base);
  }

  auto cur_time = ros::Time::now();
  static tf::TransformBroadcaster br;
  br.sendTransform(tf::StampedTransform(eigenMatrix4dToTfTransform(T_odom_to_map), cur_time, "map", "camera_init"));
  br.sendTransform(
      tf::StampedTransform(eigenMatrix4dToTfTransform(T_wheel_odom_to_map), cur_time, "map", "wheel_odom"));
  br.sendTransform(tf::StampedTransform(eigenMatrix4dToTfTransform(T_base_to_map), cur_time, "map", "base_link"));

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
  twist_msg.twist.linear.x = Twist_in_base_link(0);
  twist_msg.twist.angular.z = Twist_in_base_link(1);
  current_velocity_pub.publish(twist_msg);

  sensor_msgs::PointCloud2 scan_msg, keyspoints_msg, keyframes_pcl_msg, sub_map_msg;

  pcl::toROSMsg(*sub_map, sub_map_msg);
  sub_map_msg.header.stamp = cur_time;
  sub_map_msg.header.frame_id = "map";
  sub_map_pub.publish(sub_map_msg);

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

  std_msgs::String loc_status_msg;
  loc_status_msg.data = localization_status;
  localization_status_pub.publish(loc_status_msg);
}

void pub_map(const ros::TimerEvent& evt)
{
  if (map_loaded && points_map_pub.getNumSubscribers() > 0)
  {
    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*global_map, map_msg);
    map_msg.header.frame_id = "map";
    map_msg.header.stamp = ros::Time::now();
    points_map_pub.publish(map_msg);
  }
}

void map_callback(const std_msgs::String::Ptr& msg_ptr)
{
  if (current_map_name == msg_ptr->data)
  {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex);
  if (pcl::io::loadPCDFile<pcl::PointXYZI>(map_file_dir + msg_ptr->data, *global_map) == -1)
  {
    ROS_ERROR("地图加载失败，无效的路径!!!");
    return;
  }
  if (global_map->width > 0)
  {
    global_map = voxel_down_sample(global_map, map_voxel_size);
    crop_filter.setInputCloud(global_map);
    current_map_name = msg_ptr->data;
    std::ofstream ofs;
    ofs.open(status_file_dir + "current_map_name", std::ios::out);
    ofs << current_map_name << std::endl;
    ofs.close();
    map_loaded = 1;
    ROS_INFO("地图加载成功!");
  }
  else
  {
    ROS_ERROR("地图加载失败，无效的地图!!!");
  }
}

matching_result registration_at_scale(pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_scan,
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr pc_map, Eigen::Matrix4d& initial_guess,
                                      int scale)
{
  matching_result res;
  if (use_ndt)
  {
    ndt.setResolution(1.0 * scale);
    ndt.setMaximumIterations(10);
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
    icp.setMaximumIterations(10);
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
  timer tiemer;
  Eigen::Matrix4d T_base_to_map_estimation = pose_estimation * T_base_to_odom;
  Eigen::Vector4d start, end;
  start = T_base_to_map_estimation.block<4, 1>(0, 3) -
          Eigen::Vector4d(sub_map_size / 2.0, sub_map_size / 2.0, sub_map_size / 2.0, 0.0);
  end = T_base_to_map_estimation.block<4, 1>(0, 3) +
        Eigen::Vector4d(sub_map_size / 2.0, sub_map_size / 2.0, sub_map_size / 2.0, 0.0);
  crop_filter.setMin(start.cast<float>());
  crop_filter.setMax(end.cast<float>());
  crop_filter.filter(*sub_map);
  tiemer.print("submap");
}

bool global_registration(Eigen::Matrix4d& pose_estimation)
{
  crop_global_map_in_FOV(pose_estimation);
  timer timer;
  matching_result res = registration_at_scale(cur_keypoints_in_odom, sub_map, pose_estimation, 1);
  // res = registration_at_scale(cur_keypoints_in_odom, sub_map, res.trans, 1);
  timer.print("配准");
  fitness = res.fitness;
  if (fitness < fitness_score_threshold)
  {
    T_odom_to_map = res.trans;
    T_wheel_odom_to_map = res.trans;
    loss_cnt = 0;
    return true;
  }
  else
  {
    loss_cnt += 1;
    ROS_INFO("点云匹配失败, loss_cnt: %d, fitness: %lf", loss_cnt, fitness);
    return false;
  }
}

pcl::PointCloud<pcl::PointXYZI>::Ptr detect_iss_keypoints(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
  static pcl::search::KdTree<pcl::PointXYZI>::Ptr iss_tree(new pcl::search::KdTree<pcl::PointXYZI>());
  static pcl::PointCloud<pcl::PointXYZI>::Ptr iss_keypoints(new pcl::PointCloud<pcl::PointXYZI>());
  static pcl::ISSKeypoint3D<pcl::PointXYZI, pcl::PointXYZI> iss_detector;

  iss_detector.setInputCloud(cloud);
  iss_detector.setSearchMethod(iss_tree);
  iss_detector.setSalientRadius(1.0f);
  iss_detector.setNonMaxRadius(1.0f);
  iss_detector.setThreshold21(0.975);
  iss_detector.setThreshold32(0.975);
  iss_detector.setMinNeighbors(5);
  iss_detector.setNumberOfThreads(1);
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
  localization_status = "initializing";

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

  std::unique_lock<std::mutex> lock(mutex);
  if (!wheel_odom_only_flag)
  {
    Eigen::Matrix4d T_odom_to_map_estimation = T_base_to_map_estimation * se3_inverse(T_base_to_odom);

    if (global_registration(T_odom_to_map_estimation))
    {
      need_initial = false;
      localization_status = "success";
      ROS_INFO("初始化定位成功,fitness:%lf", fitness);
    }
    else
    {
      reset_state();
      localization_status = "fail";
      ROS_ERROR("初始化定位失败,fitness:%lf", fitness);
    };
  }
  else
  {
    T_wheel_odom_to_map = T_base_to_map;
  }
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

  pcl::PointCloud<pcl::PointXYZI>::Ptr temp_map(new pcl::PointCloud<pcl::PointXYZI>);
  for (auto& kf : keyframes)
  {
    Eigen::Matrix4d trans = T_base_to_odom * se3_inverse(kf.pose_in_odom);
    pcl::PointCloud<pcl::PointXYZI> trans_cloud;
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
  std::lock_guard<std::mutex> lock(mutex);
  Eigen::Quaterniond quaternion(msg_ptr->pose.pose.orientation.w, msg_ptr->pose.pose.orientation.x,
                                msg_ptr->pose.pose.orientation.y, msg_ptr->pose.pose.orientation.z);
  Eigen::Vector3d translation(msg_ptr->pose.pose.position.x, msg_ptr->pose.pose.position.y,
                              msg_ptr->pose.pose.position.z);
  points_odom_stamp = ros::Time::now();
  T_base_to_odom.block<3, 3>(0, 0) = quaternion.matrix();
  T_base_to_odom.block<3, 1>(0, 3) = translation;
}

void wheel_odom_callback(const nav_msgs::Odometry::ConstPtr& msg_ptr)
{
  std::lock_guard<std::mutex> lock(mutex);
  Eigen::Quaterniond quaternion(msg_ptr->pose.pose.orientation.w, msg_ptr->pose.pose.orientation.x,
                                msg_ptr->pose.pose.orientation.y, msg_ptr->pose.pose.orientation.z);

  Eigen::Vector3d translation(msg_ptr->pose.pose.position.x, msg_ptr->pose.pose.position.y,
                              msg_ptr->pose.pose.position.z);
  wheel_odom_stamp = ros::Time::now();
  T_base_to_wheel_odom.block<3, 3>(0, 0) = quaternion.matrix();
  T_base_to_wheel_odom.block<3, 1>(0, 3) = translation;

  Eigen::Vector2d vec;
  vec << msg_ptr->twist.twist.linear.x, msg_ptr->twist.twist.linear.y;
  Twist_in_base_link(0) = vec.norm();
  Twist_in_base_link(1) = msg_ptr->twist.twist.angular.z;
}

void points_callback(const sensor_msgs::PointCloud2::ConstPtr& msg_ptr)
{
  std::lock_guard<std::mutex> lock(mutex);
  pcl::fromROSMsg(*msg_ptr, *cur_scan_in_odom);
  cur_scan_stamp = ros::Time::now();
  cur_scan_in_odom = voxel_down_sample(cur_scan_in_odom, scan_voxel_size);
  cur_keypoints_in_odom = detect_iss_keypoints(cur_scan_in_odom);
}

void gnss_callback(const geometry_msgs::PoseStamped::ConstPtr& msg_ptr)
{
}

void wheel_odom_only_callback(const std_msgs::BoolConstPtr& msg_ptr)
{
  wheel_odom_only_flag = msg_ptr->data;
}

void check_status()
{
  if (access(status_file_dir.c_str(), 0))
  {
    mkdir(status_file_dir.c_str(), S_IRWXU);
    return;
  }

  std::ifstream ifs;
  ifs.open(status_file_dir + "current_map_name", std::ios::out);
  if (ifs.is_open())
  {
    std_msgs::String::Ptr map_name_ptr(new std_msgs::String());
    getline(ifs, map_name_ptr->data);
    ifs.close();
    map_callback(map_name_ptr);
  }

  ifs.open(status_file_dir + "initialpose", std::ios::out);
  if (ifs.is_open())
  {
    std::string s_content, content[7];
    getline(ifs, s_content);
    ifs.close();

    int s = 0;
    for (int i = 0; i < s_content.length(); i++)
    {
      if (s_content[i] == ' ')
      {
        s++;
        continue;
      }
      content[s] += s_content[i];
    }

    geometry_msgs::PoseWithCovarianceStamped::Ptr initialpose_ptr(new geometry_msgs::PoseWithCovarianceStamped());
    initialpose_ptr->header.frame_id = "map";
    initialpose_ptr->header.stamp = ros::Time::now();

    initialpose_ptr->pose.pose.position.x = std::stod(content[0]);
    initialpose_ptr->pose.pose.position.y = std::stod(content[1]);
    initialpose_ptr->pose.pose.position.z = std::stod(content[2]);
    initialpose_ptr->pose.pose.orientation.x = std::stod(content[3]);
    initialpose_ptr->pose.pose.orientation.y = std::stod(content[4]);
    initialpose_ptr->pose.pose.orientation.z = std::stod(content[5]);
    initialpose_ptr->pose.pose.orientation.w = std::stod(content[6]);
    initialpose_callback(initialpose_ptr);
  }
}

void thread_registration()
{
  ROS_INFO("配准子线程已开启");
  ros::Rate rate(20);
  while (ros::ok())
  {
    if (wheel_odom_only_flag != last_wheel_odom_only_flag)
    {
      last_wheel_odom_only_flag = wheel_odom_only_flag;
      if (initial_thread.joinable())
      {
        exit_initial_thread = true;
        initial_thread.join();
        exit_initial_thread = false;
      }
      std::string flag = wheel_odom_only_flag ? "wheel_odom" : "points_odom";
      reset_odom(flag);
      init_function(T_base_to_map, flag);
    }

    double diff = (cur_scan_stamp - points_odom_stamp).toSec();

    if (0.05 > diff && diff > -0.05)
    {
      std::lock_guard<std::mutex> lock(mutex);
      add_keyframe();
      if (map_loaded == 1 && !need_initial)
      {
        global_registration(T_odom_to_map);
      }
    }
    rate.sleep();
  }
  ROS_INFO("配准子线程已退出");
}

int main(int argc, char** argv)
{
  setlocale(LC_CTYPE, "zh_CN.utf8");
  ros::init(argc, argv, "pointcloud_odom");

  ros::NodeHandle private_nh("~");
  zr_param.get_paramters(private_nh);

  private_nh.param<double>("/map_voxel_size", map_voxel_size, 0.3);
  private_nh.param<double>("/sub_map_size", sub_map_size, 50);
  private_nh.param<double>("/scan_voxel_size", scan_voxel_size, 0.3);
  private_nh.param<bool>("/use_ndt", use_ndt, false);

  // Publishers
  points_map_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/points_map", 10);
  sub_map_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/sub_map", 1);
  keyframes_pcl_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/keyframes_pcl", 1);
  cur_scan_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/cur_scan_in_map", 1);
  cur_keypoints_pub = private_nh.advertise<sensor_msgs::PointCloud2>("/cur_keypoints_in_map", 1);
  current_pose_pub = private_nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 10);
  current_cov_pose_pub = private_nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/current_cov_pose", 10);
  localization_status_pub = private_nh.advertise<std_msgs::String>("/localization_status", 10);
  current_velocity_pub = private_nh.advertise<geometry_msgs::TwistStamped>("/current_velocity", 1000);

  fitness_pub = private_nh.advertise<std_msgs::Float32>("/fitness_score", 100);
  sound_pub = private_nh.advertise<std_msgs::String>("/sound_player", 10);
  reset_odom_pub = private_nh.advertise<std_msgs::Bool>("/reset_odom", 10);
  reset_wheel_odom_pub = private_nh.advertise<std_msgs::Bool>("/reset_wheel_odom", 10);

  // Subscribers
  ros::Subscriber initialpose_sub = private_nh.subscribe("/initialpose", 1, initialpose_callback);
  ros::Subscriber set_map_sub = private_nh.subscribe("/set_map", 1, map_callback);
  ros::Subscriber wheel_odom_only_sub = private_nh.subscribe("/wheel_odom_only", 1, wheel_odom_only_callback);

  ros::Subscriber points_sub = private_nh.subscribe("/cloud_registered", 1, points_callback);
  ros::Subscriber gnss_sub = private_nh.subscribe("/gnss_pose", 1, gnss_callback);
  ros::Subscriber pcl_odom_sub = private_nh.subscribe("/Odometry", 1, points_odom_callback);
  ros::Subscriber wheel_odom_sub = private_nh.subscribe("/wheel_odom", 1, wheel_odom_callback);

  reset_state();
  check_status();
  // ros::Timer timer = private_nh.createTimer(ros::Duration(5), pub_map);
  ros::Timer timer = private_nh.createTimer(ros::Duration(0.1), pub_status);
  reg_thread = std::thread(thread_registration);
  ros::spin();
  reg_thread.join();
  return 0;
}
