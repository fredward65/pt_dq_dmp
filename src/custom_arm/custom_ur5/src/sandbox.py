#!/usr/bin/env python3

import numpy as np
import rospy
from custom_tools.arm_manager import ArmManager
from custom_tools.math_tools import dx_dt, quat_rot
from custom_tools.math_tools.dq_tools import dq_log
from custom_tools.projectile_launching import gen_movement, ProjectileLaunching
from custom_tools.pt_dq_dmp import DQDMP, PTDQDMP
from dual_quaternions import DualQuaternion
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion

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
    p_r = Quaternion(axis=[0, 0, 1], angle=-1.00 * np.pi)
    p_t = Quaternion(vector=[1.00, 0.00, 0.02])
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
    arm_mng = ArmManager(p_offset, q_offset)
    rospy.sleep(1)

    # Plot movement data
    q_vec, p_vec = dmp_obj.pose_from_dq(dq_vec)
    q_rec, p_rec = dmp_obj.pose_from_dq(dq_rec)
    q_vc_, p_vc_ = dmp_obj.pose_from_dq(dq_vec * arm_mng.dq_offset)
    q_rc_, p_rc_ = dmp_obj.pose_from_dq(dq_rec * arm_mng.dq_offset)
    fig = plt.figure()
    ax_1 = fig.add_subplot(111, projection='3d')
    ax_1.plot(p_vec[:, 0], p_vec[:, 1], p_vec[:, 2], '--k')
    ax_1.plot(p_rec[:, 0], p_rec[:, 1], p_rec[:, 2], 'b')
    # ax_2 = fig.add_subplot(212, projection='3d')
    ax_1.plot(p_vc_[:, 0], p_vc_[:, 1], p_vc_[:, 2], '--k')
    ax_1.plot(p_rc_[:, 0], p_rc_[:, 1], p_rc_[:, 2], 'b')
    ax_1.axes.set_xlim3d(left=-0.5, right=1.0)
    ax_1.axes.set_ylim3d(bottom=-0.5, top=1.0)
    ax_1.axes.set_zlim3d(bottom=0., top=1.5)
    ax_1.set_proj_type('ortho')
    # plt.show()

    # Execution preliminars
    t_scale = .25
    id_ = 1
    dq_t = DualQuaternion.from_quat_pose_array(np.append([1, 0, 0, 0], p_t.elements[1:]))
    arm_mng.load_gazebo_ball(dq_t, id_=0)
    # Get original ball pose
    target_pos = arm_mng.get_gazebo_object_pose("ball_%i" % 0).position
    target_pos = np.array([target_pos.x, target_pos.y, target_pos.z])
    rospy.sleep(1)
    arm_mng.delete_gazebo_balls()

    b_pos = []
    reps = 100

    while id_ < reps: # not rospy.is_shutdown():
        # Start at home position
        print("HOMING ARM...")
        arm_mng.go_home(t_scale=t_scale)
        rospy.sleep(1)

        t_f = t_rec
        dq_f = dq_rec

        # Goal Pose
        dq = dq_f[0]
        dq_ = DualQuaternion.from_quat_pose_array([1, 0, 0, 0, .00, .00, .03]) * dq

        # Move to pose + tool offset
        print("LOWERING ARM...")
        arm_mng.move_pose(dq, t_scale=t_scale)
        rospy.sleep(1)

        # Spawn ball
        arm_mng.load_gazebo_ball(dq_, id_=id_)
        rospy.sleep(1)

        # Linear trajectory (Point-to-Point)
        print("PTP MOVEMENT...")
        arm_mng.follow_waypoints(dq_f, t_f)
        rospy.sleep(t_rec[-1] + 0.5)

        # Get current ball pose
        c_b_pos = arm_mng.get_gazebo_object_pose("ball_%i" % id_).position
        c_b_pos = np.array([c_b_pos.x, c_b_pos.y, c_b_pos.z])
        b_pos.append(c_b_pos)

        print("Original Target Pose")
        print(target_pos)
        print("Current Measured Ball Pose")
        print(c_b_pos)
        print("Current Ball Pose Difference Norm")
        print(np.linalg.norm(c_b_pos - target_pos))

        rospy.sleep(1)
        arm_mng.delete_gazebo_balls()

        id_ += 1

    t_pos = np.array(p_t.elements[1:]).reshape((-1, 3))
    b_pos = np.array(b_pos).reshape((-1, 3))
    print("Original Target Pose")
    print(target_pos)
    print("Measured Ball Poses")
    print(b_pos)
    print("Mean Ball Pose")
    print(np.mean(b_pos, axis=0))
    print("Mean Ball Pose Difference")
    print(np.mean(b_pos - t_pos, axis=0))
    print("Mean Ball Pose Difference Norm")
    print(np.mean(np.linalg.norm(b_pos - t_pos, axis=1)))
    print("Mean Ball Pose Difference Std. Dev.")
    print( np.std(np.linalg.norm(b_pos - t_pos, axis=1)))

if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
