#!/usr/bin/env python3

import numpy as np
from custom_tools.math_tools import *
from custom_tools.projectile_throwing import gen_movement
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


def main():
    # Generate movement data
    t_vec, dq_vec = gen_movement(r=.40, n=500)   # .35
    p_0 = Quaternion(vector=[-.65, -.05, .40])    # -.35 .35 .45 
    q_0 = Quaternion(axis=[0, 0, 1], angle=0.00 * np.pi)
    dq_0 = DualQuaternion.from_quat_pose_array(np.append(q_0.elements, p_0.elements[1:]))
    dq_vec = np.array([dq_0 * dq_i for dq_i in dq_vec], dtype=DualQuaternion)
    
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
    fig_synth = plt.figure(figsize=(3, 2.5), constrained_layout=True)
    fig_synth3d = plt.figure(figsize=(3, 3), constrained_layout=True)
    axs = fig_synth.subplots(4, 2)
    axs[0][0].set_title(r"$\mathbf{q}_\mathrm{r}$", fontsize=12)
    axs[0][1].set_title(r"$\mathbf{q}_\mathrm{t}$", fontsize=12)
    ax_3d = fig_synth3d.add_subplot(projection='3d')
    ax_3d.set_title(r"$\underline{\mathbf{q}}$", fontsize=12)
    # fig_synth3d.set_facecolor((0,0,0,0))
    # ax_3d.set_facecolor((0,0,0,0))
    
    txt_var = ["w", "x", "y", "z"]
    for i in range(4):
        label = txt_var[i] 
        min_lim = np.round(np.min(dq_vec_npa[:, i]), 1) - .1
        max_lim = np.round(np.max(dq_vec_npa[:, i]), 1) + .1
        axs[i][0].plot(t_vec, dq_vec_npa[:, i], linewidth=1)
        axs[i][0].set_ylim(min_lim, max_lim)
        axs[i][0].set_ylabel(r"$%s$" % label, fontsize=12)
        axs[i][0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        min_lim = np.round(np.min(dq_vec_npa[:, i + 4]), 1) - .1
        max_lim = np.round(np.max(dq_vec_npa[:, i + 4]), 1) + .1
        axs[i][1].plot(t_vec, dq_vec_npa[:, i + 4], linewidth=1, label=r"$\underline{\mathbf{q}}_\mathrm{d}$" if i == 3 else None)
        axs[i][1].set_ylim(min_lim, max_lim)
        axs[i][1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if (i == 3):
            axs[i][0].set_xlabel("$t$", fontsize=10)
            axs[i][1].set_xlabel("$t$", fontsize=10)
        else:
            axs[i][0].set_xticks([])
            axs[i][1].set_xticks([])
    # fig_synth.legend(loc="upper left", bbox_to_anchor=(.975, .93))

    ax_3d.plot(p_vec[:, 0], p_vec[:, 1], p_vec[:, 2], linewidth=1)
    ax_3d.plot(p_vec[0, 0], p_vec[0, 1], p_vec[0, 2], 'm*', markersize=5, zorder=1000, label=r'$\underline{\mathbf{q}}_0$')
    ax_3d.plot(p_vec[-1, 0], p_vec[-1, 1], p_vec[-1, 2], 'mo', markersize=3, zorder=1000, label=r'$\underline{\mathbf{q}}_\mathrm{g}$')
    draw_axes(ax_3d, dq_vec[::31], .03)
    ax_3d.set_proj_type('ortho')
    ax_3d.set_xlim([-0.75, -0.25])
    ax_3d.set_ylim([-0.25,  0.25])
    ax_3d.set_zlim([ 0.35,  0.85])
    ax_3d.view_init(elev=0, azim=-90)
    ax_3d.set_box_aspect((1, 1, 1), zoom=1.25)
    ax_3d.set_xlabel(r'$x$', fontsize=12)
    # ax_3d.set_ylabel(r'$y$', fontsize=12)
    ax_3d.set_yticks([])
    ax_3d.set_zlabel(r'$z$', fontsize=12)
    ax_3d.xaxis.labelpad = -5
    ax_3d.yaxis.labelpad = -5
    ax_3d.zaxis.labelpad = -5
    ax_3d.xaxis._axinfo["grid"]['linewidth'] = .1
    ax_3d.yaxis._axinfo["grid"]['linewidth'] = .1
    ax_3d.zaxis._axinfo["grid"]['linewidth'] = .1
    ax_3d.tick_params(axis='both', which='major', labelsize=7, pad=-2, grid_alpha=1)
    fig_synth3d.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    
    # Plot reconstructed data
    fig_rec = plt.figure(figsize=(3, 3), constrained_layout=True)
    fig_rec3d = plt.figure(figsize=(3, 3), constrained_layout=True)
    axs = fig_rec.subplots(4, 2)
    axs[0][0].set_title(r"$\mathbf{q}_\mathrm{r}$", fontsize=12)
    axs[0][1].set_title(r"$\mathbf{q}_\mathrm{t}$", fontsize=12)
    ax_3d = fig_rec3d.add_subplot(projection='3d')
    ax_3d.set_title(r"$\underline{\mathbf{q}}$", fontsize=12)
    # fig_rec3d.set_facecolor((0,0,0,0))
    # ax_3d.set_facecolor((0,0,0,0))
    
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
    fig_rec.legend(ncols=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig_rec.tight_layout(rect=[0, 0.075, 1, 1])

    # ax_3d.plot(p_rec[:, 0], p_rec[:, 1], 0 * p_rec[:, 2], 'k', linewidth=1, alpha=0.5)
    ax_3d.plot(p_rec[:, 0], p_rec[:, 1], p_rec[:, 2], linewidth=1)
    ax_3d.plot(p_rec[0, 0], p_rec[0, 1], p_rec[0, 2], 'm*', markersize=5, zorder=1000, label=r'$\underline{\mathbf{q}}_0$')
    ax_3d.plot(p_rec[-1, 0], p_rec[-1, 1], p_rec[-1, 2], 'mo', markersize=3, zorder=1000, label=r'$\underline{\mathbf{q}}_\mathrm{g}$')
    # ax_3d.plot(p_vec[:, 0], p_vec[:, 1], p_vec[:, 2], '--k', linewidth=1)
    draw_axes(ax_3d, dq_rec[::30], .03)
    ax_3d.set_proj_type('ortho')
    ax_3d.set_xlim([-0.75, -0.25])
    ax_3d.set_ylim([-0.25,  0.25])
    ax_3d.set_zlim([ 0.35,  0.85])
    ax_3d.view_init(elev=0, azim=-90)
    ax_3d.set_box_aspect((1, 1, 1), zoom=1.25)
    ax_3d.set_xlabel(r'$x$', fontsize=12)
    # ax_3d.set_ylabel(r'$y$', fontsize=12)
    ax_3d.set_yticks([])
    ax_3d.set_zlabel(r'$z$', fontsize=12)
    ax_3d.xaxis.labelpad = -5
    ax_3d.yaxis.labelpad = -5
    ax_3d.zaxis.labelpad = -5
    ax_3d.xaxis._axinfo["grid"]['linewidth'] = .1
    ax_3d.yaxis._axinfo["grid"]['linewidth'] = .1
    ax_3d.zaxis._axinfo["grid"]['linewidth'] = .1
    ax_3d.tick_params(axis='both', which='major', labelsize=7, pad=-2, grid_alpha=1)
    fig_rec3d.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, 0.93))

    fig_synth.savefig("./src/figure_syntheticdemo.png", dpi=200, bbox_inches="tight")
    fig_synth3d.savefig("./src/figure_syntheticdemo3d.png", dpi=200, bbox_inches="tight")
    fig_rec.savefig("./src/figure_syntheticreconstruction.png", dpi=200, bbox_inches="tight")
    fig_rec3d.savefig("./src/figure_syntheticreconstruction3d.png", dpi=200, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
