#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q;
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu;
  V3D cov_acc;
  V3D cov_gyr;
  V3D cov_acc_scale;
  V3D cov_gyr_scale;
  V3D cov_bias_gyr;
  V3D cov_bias_acc;
  double first_lidar_time;

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  deque<sensor_msgs::ImuConstPtr> v_imu_;
  vector<Pose6D> IMUpose;
  vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_acc;
  V3D mean_gyr;
  V3D angvel_last;
  V3D acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time_;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true;
  bool   imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1;
  Q = process_noise_cov();
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  //mean_acc      = V3D(1, 0, 0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d;
  Lidar_T_wrt_IMU = Zero3d;
  Lidar_R_wrt_IMU = Eye3d;
  last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  //mean_acc      = V3D(1, 0, 0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
  cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}
/*
meas：包含IMU和LiDAR数据的测量组。
kf_state：滤波器的状态，类型为 esekfom::esekf<state_ikfom, 12, input_ikfom>。
N：用于记录初始化过程中IMU数据的计数。
*/
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  //初始化协方差
  V3D cur_acc, cur_gyr;
  //处理第一帧IMU数据：
  if (b_first_frame_)
  {
    Reset(); // 重置初始化参数
    N = 1; // 设置初始计数
    b_first_frame_ = false; // 标记第一帧已经处理
    const auto &imu_acc = meas.imu.front()->linear_acceleration; //提取线加速度
    const auto &gyr_acc = meas.imu.front()->angular_velocity; //提取角速度
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z; // 初始化平均加速度
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z; // 初始化平均陀螺仪
    first_lidar_time = meas.lidar_beg_time; // 记录第一帧LiDAR时间
  }
  //计算加速度和陀螺仪的均值和协方差：
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    mean_acc      += (cur_acc - mean_acc) / N; // 更新平均加速度
    mean_gyr      += (cur_gyr - mean_gyr) / N; // 更新平均陀螺仪
    // 更新加速度协方差
    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    // 更新陀螺仪协方差
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);
    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;
    N ++;
  }
  //初始化滤波器状态
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2); //设置重力方向
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr; // 设置陀螺仪偏差
  init_state.offset_T_L_I = Lidar_T_wrt_IMU; // 设置LiDAR相对于IMU的平移偏差
  init_state.offset_R_L_I = Lidar_R_wrt_IMU; // 设置LiDAR相对于IMU的旋转偏差
  kf_state.change_x(init_state); // 更新滤波器状态
  //初始化滤波器协方差矩阵
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P(); //在esekfom.hpp获得P_的协方差矩阵
  init_P.setIdentity();                                                        //将协方差矩阵置为单位阵
  init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;                        //将协方差矩阵的位置和旋转的协方差置为0.00001
  init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;                    //将协方差矩阵的速度和位姿的协方差置为0.00001
  init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;                   //将协方差矩阵的重力和姿态的协方差置为0.0001
  init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;                    //将协方差矩阵的陀螺仪偏差和姿态的协方差置为0.001
  init_P(21, 21) = init_P(22, 22) = 0.00001;                                   //将协方差矩阵的lidar和imu外参位移量的协方差置为0.00001
  kf_state.change_P(init_P);                                                   //将初始化协方差矩阵传入esekfom.hpp中的P_
  last_imu_ = meas.imu.back();                                                 //将最后一帧的imu数据传入last_imu_中，在UndistortPcl使用到了
}
/*
1将当前帧的IMU数据与上一帧末尾的IMU数据合并，以获得完整的IMU数据序列。
2将激光雷达点云根据时间戳进行排序。
3初始化IMU姿态信息，并在每个IMU数据点处进行前向传播。
4在IMU数据点之间插值，以在点云的每个时间戳处获取IMU数据。
5在点云的每个时间戳处，使用IMU数据进行状态估计和预测。
6在点云的末尾，使用最新的状态估计结果对姿态和位置进行最后一次预测。
7最后，使用插值得到的IMU数据对激光雷达点云进行校正。
*/
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  //添加上一帧尾部的IMU数据到当前帧头部：
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  //排序点云数据：
  pcl_out = *(meas.lidar);
  //根据时间偏移对点云数据进行排序
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  //获取滤波器的当前状态
  state_ikfom imu_state = kf_state.get_x();
  //初始化IMU位姿列表 IMUpose，并添加初始位姿。
  IMUpose.clear();
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  //前向传播每个IMU数据点：
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;
	
  input_ikfom in;
  //遍历IMU数据点，进行前向传播，更新滤波器状态。保存每个IMU测量时的位姿信息。
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
	//head 指向当前IMU数据点，tail 指向下一个IMU数据点。这两个变量用于计算每个时间段内的平均角速度和加速度。  
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    //如果下一个IMU数据点的时间戳早于上一次LiDAR数据处理的结束时间，则跳过这个IMU数据点
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    //计算平均角速度和加速度
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
	//将加速度归一化，以匹配地球重力加速度的单位
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); //归一化加速度
	//如果当前IMU数据点的时间戳早于上一次LiDAR数据处理的结束时间
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      //dt 为从上一次LiDAR数据处理结束时间到下一个IMU数据点时间戳的时间间隔。
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      //否则，dt 为两个IMU数据点之间的时间间隔
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    //设置滤波器输入 in
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    //设置过程噪声协方差矩阵 Q 的对角线元素为预定义的协方差值
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    //调用 kf_state.predict 函数，使用 dt 和 Q 进行滤波器状态预测。
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    //保存每个IMU测量时的位姿
    //前向传播
    imu_state = kf_state.get_x();
    //计算当前IMU状态下的角速度和加速度，减去滤波器估计的偏置
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    //将当前IMU状态保存到 IMUpose 列表中，包括时间偏移量、加速度、角速度、速度、位置和旋转矩阵
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  //更新滤波器状态并保存每个IMU数据点的位姿
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  //向后传播
  kf_state.predict(dt, Q, in);
  
  imu_state = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  //对每个LiDAR点进行去畸变将点云中的每个点变换到同一个坐标系
  if (pcl_out.points.begin() == pcl_out.points.end()) return; //如果点云数据为空，则直接返回
  auto it_pcl = pcl_out.points.end() - 1;
  //从IMU位姿列表 IMUpose 的最后一个元素开始，向前遍历每个IMU位姿
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1; //当前IMU位姿
    auto tail = it_kp; //下一个IMU位姿
    R_imu<<MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);
	//遍历点云数据，处理每个LiDAR点，直到当前点的时间戳小于当前IMU位姿的时间偏移
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      //计算当前LiDAR点的时间偏移 
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      //使用IMU的平均角速度 angvel_avr 和时间偏移 dt 计算旋转矩阵 R_i，表示当前点的旋转。
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      //表示当前LiDAR点的坐标
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      //从当前IMU位姿到点云末尾IMU位姿的平移
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      //是去畸变后的点坐标
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate!
      //将去畸变后的坐标赋值给当前LiDAR点。
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}
/*
meas：包含IMU和LiDAR数据的测量组。
kf_state：滤波器的状态，类型为 esekfom::esekf<state_ikfom, 12, input_ikfom>。
cur_pcl_un_：当前未去畸变的点云数据，类型为 PointCloudXYZI::Ptr。
*/
void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  //初始化时间测量：
  double t1,t2,t3;
  t1 = omp_get_wtime();
  //检查IMU数据是否为空：
  if(meas.imu.empty()) {return;};
  //检查LiDAR数据指针是否为NULL：
  ROS_ASSERT(meas.lidar != nullptr);
  //IMU初始化：
  if (imu_need_init_)
  {
    /// The very first lidar frame
    //调用 IMU_init 函数进行初始化
    IMU_init(meas, kf_state, init_iter_num);
    imu_need_init_ = true;
    //保存最新的IMU数据
    last_imu_   = meas.imu.back();
    //获取滤波器状态 
    state_ikfom imu_state = kf_state.get_x();
    //如果初始化迭代次数超过最大计数 
    if (init_iter_num > MAX_INI_COUNT)
    {
      //调整加速度协方差 
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;
	  //重置加速度和陀螺仪的协方差比例
      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      //打印IMU初始化完成信息。
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }
  //前向传播，点云去畸变
  UndistortPcl(meas, kf_state, *cur_pcl_un_);
  //记录时间
  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
