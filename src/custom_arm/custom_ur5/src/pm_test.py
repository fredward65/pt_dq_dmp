#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from dual_quaternions import DualQuaternion
from custom_tools.math_tools import dq_log, dx_dt, quat_rot, twist_from_dq_list, vel_from_twist
from custom_tools.pt_dq_dmp import PTDQDMP
from custom_tools.projectile_launching import gen_movement, ProjectileLaunching
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion


def draw_plane(ax, n, p):
    p_1 = Quaternion([0, 0, 0, 1])
    q_1 = (0.5 * (p_1*n - n*p_1)).normalised
    ln1 = np.linspace(-1, 1, num=10)
    ln2 = np.linspace(-2, 2, num=10)
    ln1, ln2 = np.meshgrid(ln1, ln2)
    xx = p.x + p_1.x*ln1 + q_1.x*ln2
    yy = p.y + p_1.y*ln1 + q_1.y*ln2
    zz = p.z + p_1.z*ln1 + q_1.z*ln2
    ax.plot_surface(xx, yy, zz, alpha=0.25)


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
    t_vec, dq_vec = gen_movement(.35, 100)
    p_0 = Quaternion(vector=[0.00, 0.25, 0.40])
    q_0 = Quaternion(axis=[0, 0, 1], angle=.00 * np.pi)
    dq_0 = DualQuaternion.from_quat_pose_array(np.append(q_0.elements, p_0.elements[1:]))
    dq_vec = np.array([dq_0 * dq_i for dq_i in dq_vec], dtype=DualQuaternion)
    # tw_vec = twist_from_dq_list(t_vec, dq_vec)

    p_l_obj = ProjectileLaunching()

    dmp_obj = PTDQDMP(n=100, alpha_y=20)
    _, p_vec = dmp_obj.pose_from_dq(dq_vec)

    p_g = dq_log(dq_vec[-1]).q_d
    p_t = Quaternion(vector=[1.5, 1.5, 0.0])

    # v_g =  vel_from_twist(dq_vec[-1], tw_vec[-1])
    v_g =  Quaternion(vector=dx_dt(t_vec, p_vec)[-1])
    n_z = Quaternion([0.0, 0.0, 0.0, 1.0])
    n_g = (.5 * (v_g*n_z - n_z*v_g)).normalised

    q_r = p_l_obj.estimate_plane_rotation(n_g, p_g, p_t)
    n_p = q_r * n_g * q_r.conjugate
    p_p = q_r * p_g * q_r.conjugate

    t_f, v_0, p_ = p_l_obj.optimal_v_launch(p_p, p_t)

    q_v = quat_rot(q_r.conjugate * v_0 * q_r, v_g)
    tau = np.linalg.norm(v_g.elements) / np.linalg.norm(v_0.elements)
    # print([q_v.axis, q_v.angle, tau])
    dmp_obj.train_model(t_vec, dq_vec)

    t_rec = t_vec * tau
    tw_0 = DualQuaternion.from_dq_array(np.zeros(8))
    dq_off_0 = DualQuaternion.from_quat_pose_array(np.append(q_r.elements, [0, 0, 0]))
    dq_off_1 = DualQuaternion.from_quat_pose_array(np.append(q_v.elements, [0, 0, 0]))
    dq_i, dq_g = p_l_obj.correct_poses(dq_vec[0], dq_vec[-1], dq_off_0, dq_off_1)
    dq_rec, tw_rec = dmp_obj.fit_model(t_rec, dq_i, tw_0, dq_g, tau=tau)
    _, p_rec = dmp_obj.pose_from_dq(dq_rec)
    p_f = Quaternion(vector=p_rec[-1])
    v_f = Quaternion(vector=dx_dt(t_rec, p_rec)[-1])
    p_fnl = p_l_obj.simulate_launch(t_f, v_f, p_f)
    print("tau = %5.3f, err = %5.3f" % (tau, np.linalg.norm((p_[-1] - p_fnl[-1]).elements)))

    p_lnc = np.array([p_i.elements for p_i in p_])
    p_fnl = np.array([p_i.elements for p_i in p_fnl])

    ax = plt.figure().add_subplot(111, projection='3d')
    ax.plot([0], [0], [0], 'ok')
    ax.plot([p_t.x], [p_t.y], [p_t.z], 'or')
    ax.plot([p_g.x], [p_g.y], [p_g.z], 'xk')
    ax.plot(p_vec[:, 0], p_vec[:, 1], p_vec[:, 2], 'k')
    ax.quiver(p_g.x, p_g.y, p_g.z, n_g.x, n_g.y, n_g.z, length=0.2, colors=[0,0,0,1])
    ax.quiver(p_g.x, p_g.y, p_g.z, v_g.x, v_g.y, v_g.z, length=0.2, colors=[1,0,0,1])
    ax.plot([p_p.x], [p_p.y], [p_p.z], 'xb')
    ax.plot(p_lnc[:, 1], p_lnc[:, 2], p_lnc[:, 3], ':b')
    ax.plot(p_fnl[:, 1], p_fnl[:, 2], p_fnl[:, 3], ':m')
    ax.plot(p_rec[:, 0], p_rec[:, 1], p_rec[:, 2], 'b')
    ax.quiver(p_p.x, p_p.y, p_p.z, n_p.x, n_p.y, n_p.z, length=0.2, colors=[0,0,1,1])
    ax.quiver(p_p.x, p_p.y, p_p.z, v_0.x, v_0.y, v_0.z, length=0.2, colors=[0,0,1,1])
    ax.quiver(p_f.x, p_f.y, p_f.z, v_f.x, v_f.y, v_f.z, length=0.2, colors=[1,0,1,1])
    draw_axes(ax, dq_vec)
    draw_axes(ax, dq_rec)
    draw_plane(ax, n_g, p_g)
    draw_plane(ax, n_p, p_p)
    ax.axes.set_xlim3d(left=-0.5, right=1.5)
    ax.axes.set_ylim3d(bottom=-0.5, top=1.5)
    ax.axes.set_zlim3d(bottom=0., top=1.5)
    ax.set_proj_type('ortho')
    plt.show()


if __name__ == '__main__':
    main()
