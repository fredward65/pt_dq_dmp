#!/usr/bin/env python3

import numpy as np
from custom_tools.math_tools import *
from custom_tools.projectile_launching import gen_movement
from custom_tools.pt_dq_dmp import PTDQDMP
from dual_quaternions import DualQuaternion
from matplotlib import pyplot as plt, ticker as tkr
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion

np.set_printoptions(precision=3, suppress=True)
plt.rcParams.update({'text.usetex': True, 'font.size': 7, 'figure.dpi': 150})


def draw_axes(ax, dq_list, l_=.05):
    for dq_i in dq_list:
        r_i = dq_i.q_r
        p_i = 2 * dq_log(dq_i).q_d
        u_i = r_i.rotate(Quaternion(vector=[1, 0, 0]))
        v_i = r_i.rotate(Quaternion(vector=[0, 1, 0]))
        w_i = r_i.rotate(Quaternion(vector=[0, 0, 1]))
        ax.quiver(p_i.x, p_i.y, p_i.z, u_i.x, u_i.y, u_i.z, length=l_, color=[1,0,0])
        ax.quiver(p_i.x, p_i.y, p_i.z, v_i.x, v_i.y, v_i.z, length=l_, color=[0,1,0])
        ax.quiver(p_i.x, p_i.y, p_i.z, w_i.x, w_i.y, w_i.z, length=l_, color=[0,0,1])


def convergence_study():
    t_f = 1

    dq_0 = DualQuaternion.identity()
    tw_0 = DualQuaternion.from_dq_array(np.zeros(8))
    
    r_g = Quaternion(axis=[0, 1, 0], angle=0.5*np.pi)
    p_g = Quaternion(vector=[.5, 0, 1])
    dq_g = DualQuaternion.from_quat_pose_array(np.append(r_g.elements, p_g.elements[1:]))

    # r_off = Quaternion(axis=[0, 1, 0], angle=0.0*np.pi) * \
    #         Quaternion(axis=[0, 0, 1], angle=1.0*np.pi)
    # dq_off = DualQuaternion.from_quat_pose_array(np.append(r_off.elements, [0, 0, 0]))
    # dq_0 = dq_off * dq_0
    # dq_g = dq_off * dq_g

    dt = .001
    t_c = 0.0
    
    alpha = 20
    beta = alpha / 4
    dq = dq_0
    tw = tw_0
    t_rec = []
    dq_rec = []
    edq_rec = []
    tw_rec = []
    dtw_rec = []
    fac = 2
    while t_c < fac*t_f:
        edq = 2 * dq_log(dq.quaternion_conjugate() * dq_g)
        dtw = alpha * ((beta * edq) + (-1*tw))
        dtw_rec.append(dtw)
        tw_rec.append(tw)
        dq_rec.append(dq)
        edq_rec.append(edq)
        t_rec.append(t_c)
        tw = tw + dt * dtw
        # edq = edq + -dt * tw
        # dq = dq_g * dq_exp(0.5 * edq).quaternion_conjugate()
        # dq = dq * dq_exp(0.5 * dt * tw)
        # dq = dq_exp(0.5 * dt * tw) * dq
        dq = dq_exp(0.5 * dt * dq * tw * dq.quaternion_conjugate()) * dq
        t_c += dt

    r_rec, p_rec = pose_from_dq(dq_rec)
    dq_rec = dql_to_npa(dq_rec)
    tw_rec = dql_to_npa(tw_rec)
    dtw_rec = dql_to_npa(dtw_rec)
    edq_rec = dql_to_npa(edq_rec)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("Prime")
    plt.plot(t_rec, dq_rec[:, 0:4])
    plt.hlines(dq_g.q_r.elements, 0, t_f, None, 'dotted')
    plt.subplot(2, 2, 2)
    plt.title("Dual")
    plt.plot(t_rec, dq_rec[:, 4: ])
    plt.hlines(dq_g.q_d.elements, 0, t_f, None, 'dotted')
    plt.subplot(2, 2, 3)
    plt.title("quaternion")
    plt.plot(t_rec, r_rec)
    plt.hlines(r_g.elements, 0, t_f, None, 'dotted')
    plt.subplot(2, 2, 4)
    plt.title("position")
    plt.plot(t_rec, p_rec)
    plt.hlines(p_g.elements[1:], 0, t_f, None, 'dotted')

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.title("Prime Error")
    plt.plot(t_rec, edq_rec[:, 0:4])
    plt.subplot(3, 2, 2)
    plt.title("Dual Error")
    plt.plot(t_rec, edq_rec[:, 4: ])
    plt.subplot(3, 2, 3)
    plt.title("Prime Twist")
    plt.plot(t_rec, tw_rec[:, 0:4])
    plt.subplot(3, 2, 4)
    plt.title("Dual Twist")
    plt.plot(t_rec, tw_rec[:, 4: ])
    plt.subplot(3, 2, 5)
    plt.title("Prime Twist Derivative")
    plt.plot(t_rec, dtw_rec[:, 0:4])
    plt.subplot(3, 2, 6)
    plt.title("Dual Twist Derivative")
    plt.plot(t_rec, dtw_rec[:, 4: ])

    fig = plt.figure()
    ax_1 = fig.add_subplot(projection='3d')
    ax_1.view_init(elev=.001, azim=-90)
    ax_1.plot(p_rec[:, 0], p_rec[:, 1], p_rec[:, 2], 'b')
    ax_1.axes.set_xlim3d(left=-0.5, right=1.0)
    ax_1.axes.set_ylim3d(bottom=-0.5, top=1.0)
    ax_1.axes.set_zlim3d(bottom=0., top=1.5)
    ax_1.set_proj_type('ortho')

    def draw_axes(dq_list):
        l_ = .05
        for dq_i in dq_list:
            r_i = dq_i.q_r
            p_i = 2 * dq_log(dq_i).q_d
            u_i = r_i.rotate(Quaternion(vector=[1, 0, 0]))
            v_i = r_i.rotate(Quaternion(vector=[0, 1, 0]))
            w_i = r_i.rotate(Quaternion(vector=[0, 0, 1]))
            ax_1.quiver(p_i.x, p_i.y, p_i.z, u_i.x, u_i.y, u_i.z, length=l_, color=[1,0,0])
            ax_1.quiver(p_i.x, p_i.y, p_i.z, v_i.x, v_i.y, v_i.z, length=l_, color=[0,1,0])
            ax_1.quiver(p_i.x, p_i.y, p_i.z, w_i.x, w_i.y, w_i.z, length=l_, color=[0,0,1])

    dq_rec = npa_to_dql(dq_rec)
    draw_axes(dq_rec[::10])

    plt.show()


def dq_tf_example():
    p_0 = Quaternion(vector=[0, 0, 0])
    r_0 = Quaternion(axis=[0, 0, 1], angle=0.0)
    dq0 = DualQuaternion.from_quat_pose_array(np.append(r_0.elements, p_0.elements[1:]))
    
    p_1 = Quaternion(vector=[0.5, 0.0, 0])
    r_1 = Quaternion(axis=[0, 0, 1], angle=0.33*np.pi)
    dq1 = DualQuaternion.from_quat_pose_array(np.append(r_1.elements, p_1.elements[1:]))
    
    p_2 = Quaternion(vector=[1.0, 0.5, 0])
    r_2 = Quaternion(axis=[0, 0, 1], angle=0.50*np.pi)
    dq2 = DualQuaternion.from_quat_pose_array(np.append(r_2.elements, p_2.elements[1:]))
    
    eqb = dq1.quaternion_conjugate() * dq2
    
    ax = plt.figure(figsize=(3,3), tight_layout=True).add_subplot(projection='3d')
    ax.view_init(elev=90, azim=-180)

    def draw_txt_axes(dqi, color=None, txt="", label=None):
        if not color:
            c1 = [1,0,0]
            c2 = [0,1,0]
            c3 = [0,0,1]
            ct = [0,0,0]
        else:
            c1 = color
            c2 = color
            c3 = color
            ct = color
        r_i = dqi.q_r
        log = 2 * dq_log(dqi)
        p_i = log.q_d
        u_i = r_i.rotate(Quaternion(vector=[1, 0, 0]))
        v_i = r_i.rotate(Quaternion(vector=[0, 1, 0]))
        w_i = r_i.rotate(Quaternion(vector=[0, 0, 1]))
        ax.quiver(p_i.x, p_i.y, p_i.z, u_i.x, u_i.y, u_i.z, length=.25, color=c1)
        ax.quiver(p_i.x, p_i.y, p_i.z, v_i.x, v_i.y, v_i.z, length=.25, color=c2)
        ax.quiver(p_i.x, p_i.y, p_i.z, w_i.x, w_i.y, w_i.z, length=.25, color=c3, label=label)
        txt_vec = p_i - 0.125 * u_i
        txt_dir = (v_i.x, v_i.y, v_i.z)
        ax.text(txt_vec.x, txt_vec.y, txt_vec.z, txt, txt_dir, c=ct, va="center", ha="center", rotation_mode="anchor", rotation=180*r_i.angle/np.pi, fontsize=10)
    
    draw_txt_axes(dq0, [0, 0, 0], txt=r'$\underline{\mathbf{q}}_0$')
    draw_txt_axes(dq1, txt=r'$\underline{\mathbf{q}}^0_1$')
    draw_txt_axes(dq2, txt=r'$\underline{\mathbf{q}}^0_2$')
    draw_txt_axes(eqb, [1, 0, 1], txt=r'$\underline{\mathbf{q}}^1_2$', label=r'$\underline{\mathbf{q}}^1_2 = {{\underline{\mathbf{q}}}^0_1}^\ast{\underline{\mathbf{q}}}^0_2$')

    ax.axes.set_xlim3d(left=-0.5, right=1.5)
    ax.axes.set_ylim3d(bottom=-1.0, top=1.0)
    ax.axes.set_zlim3d(bottom=-0.5, top=0.5)
    ax.set_proj_type('ortho')

    ax.xaxis.set_major_locator(tkr.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(tkr.MultipleLocator(0.5))
    ax.set_xlabel("$x$", fontsize=10, loc="left")
    ax.set_ylabel("$y$", fontsize=10, loc="bottom")
    ax.set_zticks([])

    plt.tight_layout()
    plt.legend(loc="upper center")
    plt.show()


if __name__ == "__main__":
    # convergence_study()
    dq_tf_example()
