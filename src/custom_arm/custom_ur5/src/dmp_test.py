#!/usr/bin/env python3

import numpy as np
from custom_tools.math_tools import *
from custom_tools.pt_dq_dmp import PTDQDMP
from dual_quaternions import DualQuaternion
from matplotlib import pyplot as plt
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
        ax.quiver(p_i.x, p_i.y, p_i.z, u_i.x, u_i.y, u_i.z, length=l_, color=[1,0,0], linewidth=.5)
        ax.quiver(p_i.x, p_i.y, p_i.z, v_i.x, v_i.y, v_i.z, length=l_, color=[0,1,0], linewidth=.5)
        ax.quiver(p_i.x, p_i.y, p_i.z, w_i.x, w_i.y, w_i.z, length=l_, color=[0,0,1], linewidth=.5)


def min_jerk_traj(init_pos, target_pos, t_vec):
    """ https://github.com/ekorudiawan/Minimum-Jerk-Trajectory/tree/master """
    xi = init_pos
    xf = target_pos
    d = t_vec[-1]
    list_x = []
    t = 0
    for t in t_vec:
        x = xi + (xf-xi) * (10*(t/d)**3 - 15*(t/d)**4 + 6*(t/d)**5)
        list_x.append(x)
    return np.array(list_x)


def gen_data():
    t_vec = np.linspace(0, 1, num=1000)
    
    x = min_jerk_traj(0.00, 0.50, t_vec)
    y = min_jerk_traj(0.00, 0.75, t_vec)
    z = min_jerk_traj(0.00, 1.00, t_vec)

    axis = Quaternion(vector=[1, 1, 1]).normalised.elements[1:]
    angle = 0.75 * np.pi * t_vec
    q_rot = [Quaternion(axis=axis, angle=angle_i) for angle_i in angle]
    p = np.c_[x, y, z]

    dq_pose = lambda q, p : DualQuaternion.from_quat_pose_array(np.append(q, p))
    dq_vec = np.array([dq_pose(q_i.elements, p_i) for q_i, p_i in zip(q_rot, p)])

    return t_vec, dq_vec


def main():
    # Generate movement data
    t_vec, dq_vec = gen_data()

    # DMP Model
    dmp_obj = PTDQDMP(n=100, alpha_y=20)
    dmp_obj.train_model(t_vec, dq_vec)

    dq_0, dq_g = [dq_vec[0], dq_vec[-1]]
    tw_0 = DualQuaternion.from_dq_array(np.zeros(8))

    tau = 1
    fac = 1
    t_rec = np.linspace(t_vec[0], 2*fac*t_vec[-1], num=fac*t_vec.shape[0])

    dq_rec, tw_rec = dmp_obj.fit_model(t_rec, dq_0, tw_0, dq_g, tau=tau)

    dq_vec_npa = dql_to_npa(dq_vec)
    dq_rec_npa = dql_to_npa(dq_rec)

    q_vec, p_vec = pose_from_dq(dq_vec)
    q_rec, p_rec = pose_from_dq(dq_rec)

    # Plot synthetic data
    fig_synth = plt.figure(figsize=(3, 2), tight_layout=True)
    fig_3d = plt.figure(figsize=(2, 2), tight_layout=True)
    axs = fig_synth.subplots(4, 2)
    axs[0][0].set_title(r"$\mathbf{q}_\mathrm{r}$", fontsize=12)
    axs[0][1].set_title(r"$\mathbf{q}_\mathrm{t}$", fontsize=12)
    fig_3d.suptitle(r"$\underline{\mathbf{q}}$", fontsize=12)
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    txt_var = ["w", "x", "y", "z"]
    for i in range(4):
        label = txt_var[i] 
        evn = 2 * (1 + i % 4)
        min_lim = np.round(np.min(dq_vec_npa[:, i]), 1) - .1
        max_lim = np.round(np.max(dq_vec_npa[:, i]), 1) + .1
        axs[i][0].plot(t_vec, dq_vec_npa[:, i])
        axs[i][0].set_ylim(min_lim, max_lim)
        axs[i][0].set_ylabel(r"$%s$" % label, fontsize=12)
        axs[i][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        min_lim = np.round(np.min(dq_rec_npa[:, i + 4]), 1) - .1
        max_lim = np.round(np.max(dq_rec_npa[:, i + 4]), 1) + .1
        axs[i][1].plot(t_vec, dq_vec_npa[:, i + 4])
        axs[i][1].set_ylim(min_lim, max_lim)
        axs[i][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if (i == 3):
            axs[i][0].set_xlabel("$t$", fontsize=12)
            axs[i][1].set_xlabel("$t$", fontsize=12)
        else:
            axs[i][0].set_xticks([])
            axs[i][1].set_xticks([])

    
    ax_3d.plot(p_vec[:, 0], p_vec[:, 1], p_vec[:, 2])
    ax_3d.plot(p_vec[:, 0], p_vec[:, 1], 0*p_vec[:, 2], 'k', alpha=0.25)
    draw_axes(ax_3d, dq_vec[::99], .1)
    ax_3d.set_proj_type('ortho')
    ax_3d.set_xlim([0, 1])
    ax_3d.set_ylim([0, 1])
    ax_3d.set_zlim([0, 1])
    ax_3d.set_xlabel(r'$x$', fontsize=12)
    ax_3d.set_ylabel(r'$y$', fontsize=12)
    ax_3d.set_zlabel(r'$z$', fontsize=12)
    ax_3d.xaxis.labelpad = -5
    ax_3d.yaxis.labelpad = -5
    ax_3d.zaxis.labelpad = -5
    ax_3d.tick_params(axis='both', which='major', labelsize=5, pad=-2, grid_alpha=1)
    ax_3d.view_init(elev=45, azim=-45)
    ax_3d.set_box_aspect((1, 1, 1), zoom=.90)
    
    # Plot reconstructed data
    fig_synthrec = plt.figure(figsize=(5, 2.5), constrained_layout=True)
    subfigs = fig_synthrec.subfigures(1, 2, width_ratios=[1, 1])
    axs = subfigs[0].subplots(4, 2)
    axs[0][0].set_title(r"$\mathbf{q}_\mathrm{r}$", fontsize=12)
    axs[0][1].set_title(r"$\mathbf{q}_\mathrm{t}$", fontsize=12)
    ax_3d = subfigs[1].add_subplot(projection='3d')
    ax_3d.set_title(r"$\underline{\mathbf{q}}$", fontsize=12)
    subfigs[1].set_facecolor((0,0,0,0))
    ax_3d.set_facecolor((0,0,0,0))
    
    txt_var = ["w", "x", "y", "z"]
    for i in range(4):
        label = txt_var[i] 
        min_lim = np.round(np.min(dq_rec_npa[:, i]), 1) - .1
        max_lim = np.round(np.max(dq_rec_npa[:, i]), 1) + .1
        axs[i][0].hlines(dq_vec_npa[-1, i], 0, t_rec[-1], [(1, 0, 0)], "dotted", linewidth=.75)
        axs[i][0].plot(t_rec, dq_rec_npa[:, i], linewidth=.75)
        axs[i][0].plot(t_vec, dq_vec_npa[:, i], "--k", linewidth=1)
        axs[i][0].set_ylim(min_lim, max_lim)
        axs[i][0].set_ylabel(r"$%s$" % label, fontsize=12)
        axs[i][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        min_lim = np.round(np.min(dq_rec_npa[:, i + 4]), 1) - .1
        max_lim = np.round(np.max(dq_rec_npa[:, i + 4]), 1) + .1
        axs[i][1].hlines(dq_vec_npa[-1, i + 4], 0, t_rec[-1], [(1, 0, 0)], "dotted", linewidth=.75, label=r"$\underline{\mathbf{q}}_\mathrm{g}$" if i == 3 else None)
        axs[i][1].plot(t_rec, dq_rec_npa[:, i + 4], linewidth=.75, label=r"$\underline{\mathbf{q}}_\mathrm{f}$" if i == 3 else None)
        axs[i][1].plot(t_vec, dq_vec_npa[:, i + 4], "--k", linewidth=1, label=r"$\underline{\mathbf{q}}_\mathrm{d}$" if i == 3 else None)
        axs[i][1].set_ylim(min_lim, max_lim)
        axs[i][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if (i == 3):
            axs[i][0].set_xlabel("$t$", fontsize=10)
            axs[i][1].set_xlabel("$t$", fontsize=10)
        else:
            axs[i][0].set_xticks([])
            axs[i][1].set_xticks([])
    subfigs[0].legend(loc="upper left", bbox_to_anchor=(1, .92))

    ax_3d.plot(p_rec[:, 0], p_rec[:, 1], 0 * p_rec[:, 2], 'k', linewidth=1, alpha=0.5)
    ax_3d.plot(p_rec[:, 0], p_rec[:, 1], p_rec[:, 2], linewidth=1)
    ax_3d.plot(p_vec[:, 0], p_vec[:, 1], p_vec[:, 2], '--k', linewidth=1)
    draw_axes(ax_3d, dq_rec[::30], .075)
    ax_3d.set_proj_type('ortho')
    ax_3d.set_xlim([0, 1])
    ax_3d.set_ylim([0, 1])
    ax_3d.set_zlim([0, 1])
    ax_3d.view_init(elev=20, azim=-60)
    ax_3d.set_box_aspect((1, 1, 1), zoom=.9)
    ax_3d.set_xlabel(r'$x$', fontsize=12)
    ax_3d.set_ylabel(r'$y$', fontsize=12)
    ax_3d.set_zlabel(r'$z$', fontsize=12)
    ax_3d.xaxis.labelpad = -5
    ax_3d.yaxis.labelpad = -5
    ax_3d.zaxis.labelpad = -5
    ax_3d.xaxis._axinfo["grid"]['linewidth'] = .5
    ax_3d.yaxis._axinfo["grid"]['linewidth'] = .5
    ax_3d.zaxis._axinfo["grid"]['linewidth'] = .5
    ax_3d.tick_params(axis='both', which='major', labelsize=5, pad=-2, grid_alpha=1)

    fig_synthrec.savefig("./src/figures/figure_dqdmp.png", dpi=200, bbox_inches="tight")

    plt.show()

    # plt.figure()
    # labels = ['x', 'y', 'z']
    # for i in range(3):
    #     min_lim = np.round(np.min(p_rec[:, i]), 1) - .05
    #     max_lim = np.round(np.max(p_rec[:, i]), 1) + .05
    #     plt.subplot(3, 1, i + 1)
    #     plt.ylabel(r'$\mathrm{p}_%s$' % labels[i])
    #     plt.hlines(p_vec[-1, i], 0, t_rec[-1], [(1, 0, 0)], 'dotted', label=r'$\mathrm{p}_g$')
    #     plt.plot(t_rec, p_rec[:, i], label=r'$\mathrm{p}_f$')
    #     plt.plot(t_vec, p_vec[:, i], '--k', label=r'$\mathrm{p}_d$')
    #     plt.ylim(min_lim, max_lim)
    #     if (i < 2):
    #         plt.xticks([])
    #     else:
    #         plt.xlabel(r'$t$')
    #         plt.legend(loc='lower right')

    # plt.figure()
    # plt.subplot(2, 1, 2)
    # plt.xlabel(r'$t$')
    # plt.ylabel(r'$\mathrm{q}_r$')
    # plt.hlines(q_vec[-1, :], 0, t_rec[-1], None, 'dotted')
    # plt.plot(t_rec, q_rec)
    # plt.plot(t_vec, q_vec, '--')

    # plt.figure()
    # from custom_tools.math_tools.dq_tools import twist_from_dq_list
    # tw_vec = twist_from_dq_list(t_vec, dq_vec)
    # tw_vec = dql_to_npa(tw_vec)
    # tw_rec = dql_to_npa(tw_rec)
    # plt.subplot(2, 2, 1)
    # plt.plot(t_rec, tw_rec[:, 5: ])
    # plt.plot(t_vec, tw_vec[:, 5: ], '--')
    # plt.subplot(2, 2, 2)
    # plt.plot(t_rec, tw_rec[:, 1:4])
    # plt.plot(t_vec, tw_vec[:, 1:4], '--')

    # fig = plt.figure()
    # ax_1 = fig.add_subplot(projection='3d')
    # ax_1.plot(p_vec[:, 0], p_vec[:, 1], p_vec[:, 2], '--k')
    # ax_1.plot(p_rec[:, 0], p_rec[:, 1], p_rec[:, 2], 'b')
    # ax_1.axes.set_xlim3d(left=-0.5, right=1.0)
    # ax_1.axes.set_ylim3d(bottom=-0.5, top=1.0)
    # ax_1.axes.set_zlim3d(bottom=0., top=1.5)
    # ax_1.set_proj_type('ortho')

    # draw_axes(ax_1, dq_vec[::100])
    # draw_axes(ax_1, dq_rec[::100])
    
    # plt.show()


if __name__ == "__main__":
    main()
