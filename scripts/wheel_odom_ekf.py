#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import time
import numpy as np
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from zr_msgs.msg import motor_info
import rospy
import copy


def get_ratation(origin_vector, location_vector):
    origin_vector = origin_vector/np.linalg.norm(origin_vector)
    location_vector = location_vector/np.linalg.norm(location_vector)
    c = np.dot(origin_vector, location_vector)
    n_vector = np.cross(origin_vector, location_vector)

    n_vector_invert = np.array((
        [0, -n_vector[2], n_vector[1]],
        [n_vector[2], 0, -n_vector[0]],
        [-n_vector[1], n_vector[0], 0]
    ))
    I = np.eye(3)

    R_w2c = I + n_vector_invert + \
        np.dot(n_vector_invert, n_vector_invert)/(1+c)
    return R_w2c


class OdomCalculator:
    def __init__(self):
        rospy.init_node('odom_calculator')
        self.pub = rospy.Publisher('wheel_odom', Odometry, queue_size=10)
        self.pub_twist_msg = rospy.Publisher(
            'twist', TwistStamped, queue_size=10)
        self.sub = rospy.Subscriber(
            'motor_info', motor_info, self.encoders_callback)
        self.sub_imu = rospy.Subscriber(
            'imu_data', Imu, self.imu_callback)

        # 车体参数
        self.baseline = 0.5
        self.max_v = 1.6

        # 机器人位姿 [x, y, vx, vy, theta]
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.last_encoder_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.last_a = 0.0
        self.last_w = 0.0
        self.last_encoder_w = 0.0
        self.data_num = 0
        self.first_encoder_msg = True
        self.first_imu_msg = True
        self.last_encoder_time = rospy.Time.now()
        self.last_imu_time = rospy.Time.now()
        mid_w = np.array([0.0, 0.0, 0.0])
        mid_a = np.array([0.0, 0.0, 0.0])
        self.g = np.array([0, 0, 9.81])
        self.sum_g = np.array([0.0, 0.0, 0.0])
        self.sum_w = np.array([0.0, 0.0, 0.0])
        self.R_b_w = np.identity(3)
        self.b_w = np.array([0.0, 0.0, 0.0])
        self.b_a = np.array([0.0, 0.0, 0.0])

        # 卡尔曼滤波参数
        self.Q = np.diag([1, 1, 1, 1, 0.1])  # 系统噪声协方差
        self.R = np.diag([0.1, 0.1, 0.1, 0.1, 1])  # 观测噪声协方差
        self.P = np.diag([1, 1, 1, 1, 1])  # 估计误差协方差
        self.F = None
        self.H = np.array([[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0], [
                          0.0, 0.0, 0.0, 0.0, 1.0]])  # 观测矩阵

    def encoders_callback(self, msg):
        start = time.time()
        if self.first_imu_msg:
            return

        if self.first_encoder_msg:
            self.last_encoder_time = msg.header.stamp
            self.first_encoder_msg = False

        v = (msg.left_vel + msg.right_vel)/2.0
        w = (msg.right_vel - msg.left_vel)/self.baseline

        current_time = msg.header.stamp
        dt = (self.last_imu_time - self.last_encoder_time).to_sec()
        self.last_encoder_time = current_time

        delta_yaw = (self.last_encoder_w + w)/2 * dt

        # 观测矩阵3x5
        # delta_p = v * dt
        # z_m = np.array([delta_p, v, delta_yaw])

        # delta_p_x = self.state[0] - self.last_encoder_state[0]
        # delta_p_y = self.state[1] - self.last_encoder_state[1]
        # delta_q_z = self.state[4] - self.last_encoder_state[4]
        # z_p = np.array([((delta_p_x)**2 + (delta_p_y)**2)**0.5,
        #                 (self.state[2]**2 + self.state[3]**2)**0.5,
        #                 delta_q_z
        #                 ])

        # self.H = np.array([[self.state[0]*((delta_p_x)**2 + (delta_p_y)**2)**-0.5, self.state[1]*((delta_p_x)**2 + (delta_p_y)**2)**-0.5, 0.0, 0.0, 0.0],
        #                    [0.0, 0.0, self.state[2]*(self.state[2]**2 + self.state[3]**2)**-0.5, self.state[3]*(
        #                        self.state[2]**2 + self.state[3]**2)**-0.5, 0.0],
        #                    [0.0, 0.0, 0.0, 0.0, 1.0]
        #                    ])

        # 观测矩阵5x5
        v_x = v*math.cos(delta_yaw+self.last_encoder_state[4])
        v_y = v*math.sin(delta_yaw+self.last_encoder_state[4])
        p_x = v_x * dt + self.last_encoder_state[0]
        p_y = v_y * dt + self.last_encoder_state[1]
        z_m = np.array([p_x, p_y, v_x, v_y, delta_yaw +
                        self.last_encoder_state[4]])
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        z_p = np.dot(self.H, self.state)

        # 计算卡尔曼增益
        R = self.R * abs(v);
        R[4, 4] = self.R[4, 4] * abs(w)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(
            np.dot(np.dot(self.H, self.P), self.H.T) + R))

        # 更新位姿估计
        dz = z_m - z_p
        d_state = np.dot(K, dz)
        self.state = self.state + d_state

        # 更新估计误差协方差
        self.P = np.dot((np.eye(5) - np.dot(K, self.H)), self.P)

        # 发布状态
        self.pub_odom()
        self.pub_twist(v, w)

        # print("z_m:{}".format(z_m))
        # print("d_z:{}".format(dz))
        # print("z_p:{}".format(z_p))

        # print("last_encoder_state:{}".format(self.last_encoder_state))
        # print("d_state:{}".format(d_state))
        # print("state:{}".format(self.state))

        # 花费时间要小于(1/imu频率)
        print("cost:{}".format(time.time()-start))

        self.last_encoder_w = w
        self.last_encoder_state = copy.deepcopy(self.state)

    def imu_callback(self, msg):
        a_m = np.array([msg.linear_acceleration.x,
                        msg.linear_acceleration.y, msg.linear_acceleration.z])
        w_m = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        if np.abs(np.linalg.norm(a_m) - self.g[2]) < 0.02:
            self.sum_g = a_m + self.sum_g
            self.sum_w = w_m + self.sum_w
            self.data_num += 1

        if self.data_num == 100:
            g = self.sum_g / self.data_num
            self.b_w = self.sum_w / self.data_num
            self.R_b_w = get_ratation(self.g, g)
            self.b_a = np.dot(self.R_b_w.transpose(), g) - self.g
            # print("R_b_w:{}".format(self.R_b_w))
            # print("b_a:{}".format(self.b_a))
            # print("b_w:{}".format(self.b_w))
            self.data_num = 0

        if self.first_imu_msg:
            self.first_imu_msg = False
            self.last_imu_time = msg.header.stamp
            return

        # 求车体加速度、角速度
        a_body = np.dot(self.R_b_w.transpose(), a_m) - self.b_a
        w_body = np.dot(self.R_b_w.transpose(), w_m) - self.b_w

        a_body[2] = 0.0
        if a_body[0] > 0:
            a_body[0] = np.linalg.norm(a_body)
        else:
            a_body[0] = -1 * np.linalg.norm(a_body)
        a_body[1] = 0.0

        # 将车体加速度、角速度转到世界系
        current_time = msg.header.stamp
        dt = (current_time - self.last_imu_time).to_sec()
        self.last_imu_time = current_time
        w_w = w_body
        mid_w = (w_w+self.last_w)/2
        delta_yaw = mid_w[2] * dt

        a_w = np.zeros(3)
        a_w[0] = a_body[0] * math.cos(self.state[4]+delta_yaw)
        a_w[1] = a_body[0] * math.sin(self.state[4]+delta_yaw)
        mid_a = (a_w+self.last_a)/2

        # 状态转移矩阵更新
        self.F = np.array([
            [1.0, 0.0, dt, 0.0, -dt**2 * mid_a[1]/2],
            [0.0, 1.0, 0.0, dt, dt**2 * mid_a[0]/2],
            [0.0, 0.0, 1.0, 0.0, -dt * mid_a[1]],
            [0.0, 0.0, 0.0, 1.0, dt * mid_a[0]],
            [0.0, 0.0, 0.0, 0.0, 1.0]])

        # 状态更新
        self.state[0] += self.state[2] * dt + dt**2 * mid_a[0] / 2.0
        self.state[1] += self.state[3] * dt + dt**2 * mid_a[1] / 2.0
        self.state[2] += mid_a[0] * dt
        self.state[3] += mid_a[1] * dt
        self.state[4] += delta_yaw

        # 预测过程中的估计误差协方差更新
        Q[0:4,0:4] = self.Q[0:4,0:4] / (abs(a_body[0])+1e-5)
        Q[4, 4] = self.Q[4, 4] /(abs(w_body[2])+1e-5)
        print("Q:{}".format(Q))
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + Q
        self.last_w = copy.deepcopy(w_w)
        self.last_a = copy.deepcopy(a_w)

    def pub_odom(self):
        # 发布里程计消息
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = 'camera_init'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = self.state[0]
        odom_msg.pose.pose.position.y = self.state[1]
        quat = quaternion_from_euler(0, 0, self.state[4])
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        self.pub.publish(odom_msg)

    def pub_twist(self, v, w):
        # 发布里程计消息
        twist_msg = TwistStamped()
        twist_msg.header.stamp = rospy.Time.now()
        twist_msg.header.frame_id = 'base_link'
        twist_msg.twist.linear.x = v
        twist_msg.twist.linear.y = 0.0
        twist_msg.twist.linear.z = 0.0
        twist_msg.twist.angular.x = 0.0
        twist_msg.twist.angular.y = 0.0
        twist_msg.twist.angular.z = w
        self.pub_twist_msg.publish(twist_msg)


if __name__ == '__main__':
    odom_calculator = OdomCalculator()
    rospy.spin()
