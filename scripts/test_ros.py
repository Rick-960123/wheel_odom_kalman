import tf.transformations
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion

def pose_to_mat(pose_msg):
    return np.matmul(
        tf.listener.xyz_to_mat44(pose_msg.position),
        tf.listener.xyzw_to_mat44(pose_msg.orientation),
    )

baserot = tf.transformations.quaternion_matrix((0.0, 0.0, -0.13275, 0.9911))
euler = tf.transformations.euler_from_quaternion((0.0, 0.0, -0.13275, 0.9911))

print(baserot)
roll, pitch, yaw = euler
print("Roll: ", roll)
print("Pitch: ", pitch)
print("Yaw: ", yaw)

quat = tf.transformations.quaternion_from_euler(*euler)
pose = Pose(Point(0,0,0), Quaternion(*quat))
print(pose_to_mat(pose))