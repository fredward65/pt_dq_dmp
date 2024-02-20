#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from dual_quaternions import DualQuaternion
from custom_tools.math_tools import *
from custom_tools.math_tools.dq_tools import twist_from_dq_list, vel_from_twist
from custom_tools.pt_dq_dmp import PTDQDMP
from custom_tools.projectile_launching import gen_movement, ProjectileLaunching
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion

np.set_printoptions(precision=3, suppress=True)
plt.rcParams.update({"text.usetex": True})


def draw_plane(ax, n, p):
    p_1 = Quaternion([0, 0, 0, 1])
    q_1 = (0.5 * (p_1*n - n*p_1)).normalised
    ln1 = np.linspace(-1, 1, num=10)
    ln2 = np.linspace(-.5, 1.5, num=10)
    ln1, ln2 = np.meshgrid(ln1, ln2)
    xx = p.x + p_1.x*ln1 + q_1.x*ln2
    yy = p.y + p_1.y*ln1 + q_1.y*ln2
    zz = p.z + p_1.z*ln1 + q_1.z*ln2
    ax.plot_surface(xx, yy, zz, alpha=0.25, antialiased=False)


def draw_axes(ax, dq_list):
    for dq_i in dq_list[::10]:
        p_i = dq_log(dq_i).q_d
        q_i = dq_i.q_r
        v_x = q_i.rotate([1, 0, 0])
        v_y = q_i.rotate([0, 1, 0])
        v_z = q_i.rotate([0, 0, 1])
        ax.quiver(p_i.x, p_i.y, p_i.z, v_x[0], v_x[1], v_x[2], length=0.05, colors=[1,0,0,1])
        ax.quiver(p_i.x, p_i.y, p_i.z, v_y[0], v_y[1], v_y[2], length=0.05, colors=[0,1,0,1])
        ax.quiver(p_i.x, p_i.y, p_i.z, v_z[0], v_z[1], v_z[2], length=0.05, colors=[0,0,1,1])


def main():
    # Generate movement data
    t_vec, dq_vec = gen_movement(r=.40, n=1000)    # .35
    p_0 = Quaternion(vector=[-.65, -.05, .40])      # -.35 .35 .45 
    q_0 = Quaternion(axis=[0, 0, 1], angle=0.00 * np.pi)
    dq_0 = DualQuaternion.from_quat_pose_array(np.append(q_0.elements, p_0.elements[1:]))
    dq_vec = np.array([dq_0 * dq_i for dq_i in dq_vec], dtype=DualQuaternion)
    tw_vec = twist_from_dq_list(t_vec, dq_vec)

    # Set Cartesian target
    p_r = Quaternion(axis=[0, 0, 1], angle=-0.25 * np.pi)
    p_t = Quaternion(vector=[1.00, 0.00, 0.02])
    p_t = p_r.rotate(p_t)

    p_l_obj = ProjectileLaunching()

    dmp_obj = PTDQDMP(n=100, alpha_y=20)
    _, p_vec = pose_from_dq(dq_vec)
    dmp_obj.train_model(t_vec, dq_vec)

    # Projectile launching
    dq_g = dq_vec[-1]
    tw_g = tw_vec[-1]
    v_g = vel_from_twist(dq_g, dq_g * tw_g * dq_g.quaternion_conjugate())
    n_z = Quaternion([0.0, 0.0, 0.0, 1.0])
    n_g = (.5 * (v_g*n_z - n_z*v_g)).normalised

    p_g = 2 * dq_log(dq_g).q_d

    q_r = p_l_obj.estimate_plane_rotation(n_g, p_g, p_t)
    n_p = q_r.rotate(n_g)
    p_p = q_r.rotate(p_g)

    t_f, v_0 = p_l_obj.optimal_v_launch(p_p, p_t)
    p_est, t_est = p_l_obj.simulate_launch(t_f, v_0, p_p)
    p_dem, t_dem = p_l_obj.simulate_launch(t_f, v_g, p_g)

    q_v = quat_rot(q_r.conjugate.rotate(v_0), v_g)
    tau = v_g.norm / v_0.norm

    t_rec = np.linspace(0, t_vec[-1] * tau, num=100) 
    tw_0 = DualQuaternion.from_dq_array(np.zeros(8))

    dq_off_0 = DualQuaternion.from_quat_pose_array(np.append(q_r.elements, [0, 0, 0]))
    dq_off_1 = DualQuaternion.from_quat_pose_array(np.append(q_v.elements, [0, 0, 0]))
    dq_0, dq_g = p_l_obj.correct_poses(dq_vec[0], dq_g, dq_off_0, dq_off_1)
    
    dq_rec, tw_rec = dmp_obj.fit_model(t_rec, dq_0, tw_0, dq_g, tau=tau)
    
    _, p_rec = pose_from_dq(dq_rec)
    dq_f = dq_rec[-1]
    tw_f = tw_rec[-1]
    p_f = 2 * dq_log(dq_f).q_d
    v_f = vel_from_twist(dq_f, dq_f * tw_f * dq_f.quaternion_conjugate())
    p_fnl, t_fnl = p_l_obj.simulate_launch(t_f, v_f, p_f)
    err_g = (p_p - p_f).norm
    err_t = (p_est[-1] - p_fnl[-1]).norm
    err_v = (v_f - v_0).norm
    print("target : ", p_t.elements[1:])
    print("err goal = %5.5f, err target = %5.3f" % (err_g, err_t))
    print("tau = %5.3f, err vel = %5.3f" % (tau, err_v))

    p_the = p_fnl[-1]

    p_lnc = np.array([p_i.elements[1:] for p_i in p_est])
    p_fnl = np.array([p_i.elements[1:] for p_i in p_fnl])
    p_thr = np.array([p_i.elements[1:] for p_i in p_dem])

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    # ax.plot([0], [0], [0], '*k')
    ax.plot([p_the.x], [p_the.y], [p_the.z], 'ob')
    ax.plot([p_t.x], [p_t.y], [p_t.z], 'or')
    ax.plot([p_g.x], [p_g.y], [p_g.z], 'xk')
    ax.plot(p_vec[:, 0], p_vec[:, 1], p_vec[:, 2], 'k')
    # ax.plot(p_thr[:, 1], p_thr[:, 2], p_thr[:, 3], ':k')
    ax.quiver(p_g.x, p_g.y, p_g.z, n_g.x, n_g.y, n_g.z, length=0.2, colors=[0,0,0,1])
    ax.quiver(p_g.x, p_g.y, p_g.z, v_g.x, v_g.y, v_g.z, length=0.2, colors=[1,0,0,1])
    ax.plot([p_p.x], [p_p.y], [p_p.z], 'xb')
    ax.plot(p_lnc[:, 0], p_lnc[:, 1], p_lnc[:, 2], ':b')
    ax.plot(p_fnl[:, 0], p_fnl[:, 1], p_fnl[:, 2], ':m')
    ax.plot(p_rec[:, 0], p_rec[:, 1], p_rec[:, 2], 'b')
    ax.quiver(p_p.x, p_p.y, p_p.z, n_p.x, n_p.y, n_p.z, length=0.2, colors=[0,0,1,1])
    ax.quiver(p_p.x, p_p.y, p_p.z, v_0.x, v_0.y, v_0.z, length=0.2, colors=[0,0,1,1])
    ax.quiver(p_f.x, p_f.y, p_f.z, v_f.x, v_f.y, v_f.z, length=0.2, colors=[1,0,1,1])
    # draw_axes(ax, dq_vec)
    # draw_axes(ax, dq_rec)
    draw_plane(ax, n_g, p_g)
    draw_plane(ax, n_p, p_p)
    # ax.set_box_aspect((1, 1, 1))
    ax.axes.set_xlim3d(left=-1, right=1)
    ax.axes.set_ylim3d(bottom=-1, top=1)
    ax.axes.set_zlim3d(bottom=0, top=2)
    ax.set_proj_type('ortho')
    ax.view_init(elev=15, azim=-150)
    # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
