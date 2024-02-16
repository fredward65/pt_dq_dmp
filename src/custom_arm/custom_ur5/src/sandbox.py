#!/usr/bin/env python3

import numpy as np
import rospy
import time
from custom_tools.arm_manager import ArmManager
from custom_tools.projectile_launching import gen_movement
from custom_tools.pt_dq_dmp import PTDQDMP
from dual_quaternions import DualQuaternion
from pyquaternion import Quaternion
from rospkg import RosPack

np.set_printoptions(precision=3, suppress=True)


def main():
    # Load movement data
    # fpath = rospkg.RosPack().get_path('custom_ur5') + "/resources/"
    # mdata = np.genfromtxt(fpath + 'demo_basket_left_1.csv', delimiter=',', dtype='float', skip_header=1)
    # _, idx_list = np.unique(mdata[:, 1:], axis=0, return_index=True)
    # idx_list = np.sort(idx_list)
    # pose_vec = mdata[idx_list, 1:]
    # t_vec = mdata[idx_list, 0]
    # p_scale = 1
    # p_vec = p_scale * (pose_vec[:, 0:3] - pose_vec[0, 0:3])
    # q_vec = pose_vec[:, 3:]
    # p_0 = np.array([0.30, 0.00, 0.30])
    # q_0 = Quaternion(axis=[0, 1, 0], angle=-.25 * np.pi)

    # Generate movement data
    t_vec, dq_vec = gen_movement(r=.40, n=1000)    # .35
    p_0 = Quaternion(vector=[-.65, -.05, .40])      # -.35 .35 .45 
    q_0 = Quaternion(axis=[0, 0, 1], angle=0.00 * np.pi)
    dq_0 = DualQuaternion.from_quat_pose_array(np.append(q_0.elements, p_0.elements[1:]))
    dq_vec = np.array([dq_0 * dq_i for dq_i in dq_vec], dtype=DualQuaternion)

    # Set Cartesian target
    p_r = Quaternion(axis=[0, 0, 1], angle=-1.00 * np.pi)
    p_t = Quaternion(vector=[1.50, 0.00, 0.02])
    p_t = p_r.rotate(p_t)

    # DMP Model
    dmp_obj = PTDQDMP(n=100, alpha_y=20)
    dmp_obj.train_model(t_vec, dq_vec)

    # Projectile launching
    t_rec = np.linspace(0, t_vec[-1], num=300)
    dq_rec, tw_rec = dmp_obj.aim_model(t_rec, p_t)

    # Instance Arm Manager
    p_offset = Quaternion(vector=[.0, .0, -.075])
    q_offset = Quaternion(axis=[0, 1, 0], angle= .5 * np.pi) * \
               Quaternion(axis=[0, 0, 1], angle= .5 * np.pi)
    arm_mng = ArmManager(p_offset, q_offset, "ProjEST")
    rospy.sleep(1)

    # Execution preliminars
    t_scale = .25
    dq_t = DualQuaternion.from_quat_pose_array(np.append([1, 0, 0, 0], p_t.elements[1:]))
    arm_mng.load_gazebo_ball(dq_t, id_=0)
    # Get original ball pose
    target_pos = arm_mng.get_gazebo_object_pose("ball_%i" % 0).position
    target_pos = np.array([target_pos.x, target_pos.y, target_pos.z])
    rospy.sleep(1)
    arm_mng.delete_gazebo_balls()

    # Initial Pose
    dq_init = dq_rec[0]
    dq_ball = DualQuaternion.from_quat_pose_array([1, 0, 0, 0, .00, .00, .03]) * dq_init

    b_pos = []
    reps = 100

    while len(b_pos) < reps: # not rospy.is_shutdown():
        # Start at home position
        print("HOMING ARM...")
        arm_mng.go_home(t_scale=t_scale)
        rospy.sleep(1)

        # Move to pose + tool offset
        print("LOWERING ARM...")
        flag = arm_mng.move_pose(dq_init, t_scale=t_scale)
        rospy.sleep(1)

        if flag == True:
            # Spawn ball
            arm_mng.load_gazebo_ball(dq_ball)
            rospy.sleep(1)

            # Linear trajectory (Point-to-Point)
            print("PTP MOVEMENT...")
            flag = arm_mng.follow_waypoints(dq_rec, t_rec)
            rospy.sleep(t_rec[-1] + 0.5)

            # Get current ball pose
            c_b_pos = arm_mng.get_ball_position()
            c_b_pos = np.array([c_b_pos.x, c_b_pos.y, c_b_pos.z])
            if flag == True:
                b_pos.append(c_b_pos)

            print("Original Target Pose")
            print(target_pos)
            print("Current Measured Ball Pose")
            print(c_b_pos)
            print("Current Ball Pose Difference Norm")
            print(np.linalg.norm(c_b_pos - target_pos))

            rospy.sleep(1)
            arm_mng.delete_gazebo_balls()

    planner = arm_mng.arm_group.get_planner_id()
    t_pos = np.array(p_t.elements[1:]).reshape((-1, 3))
    b_pos = np.array(b_pos).reshape((-1, 3))
    p_mean = np.mean(b_pos, axis=0)
    p_err = b_pos - t_pos
    p_err_mean = np.mean(p_err, axis=0)
    p_err_norm_mean = np.mean(np.linalg.norm(p_err, axis=1))
    p_err_norm_stdd = np.std(np.linalg.norm(b_pos - t_pos, axis=1))

    c_time = time.strftime("%Y%m%d%H%M%S")
    file_name = "-".join(["throw_data", planner, str(c_time)])
    file_path = RosPack().get_path("custom_ur5") + "/resources/"
    np.savez_compressed(file_path + file_name, planner_id=planner, target_pos=t_pos, ball_pos_l=b_pos)

    print("Original Target Pose")
    print(target_pos)
    print("Measured Ball Poses")
    print(b_pos)
    print("Mean Ball Pose")
    print(p_mean)
    print("Mean Ball Pose Difference")
    print(p_err_mean)
    print("Mean Ball Pose Difference Norm")
    print(p_err_norm_mean)
    print("Mean Ball Pose Difference Std. Dev.")
    print(p_err_norm_stdd)


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
