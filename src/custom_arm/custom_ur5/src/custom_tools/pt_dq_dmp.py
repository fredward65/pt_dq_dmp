#!/usr/bin/env python
import numpy as np
from dual_quaternions import DualQuaternion
from pyquaternion import Quaternion
from .math_tools import *
from .math_tools.dq_tools import next_dq_from_twist, twist_from_dq_list, vel_from_twist
from .projectile_launching import ProjectileLaunching


class DQDMP(object):
    """
    A class for computing Dynamic Movement Primitives (DMP)
    from full pose Dual Quaternion data
    """
    def __init__(self, n:int=20, alpha_y:float=4) -> None:
        """
        Dual Quaternion Dynamic Movement Primitives
        -----
        ### Parameters
        @n: Number of Gaussian kernels to reconstruct the forcing term
        @alpha_y: Damping coefficient
        """
        # Previous time to compute timestep
        self.prev_t = 0.0
        # Critically damped point attractor system parameters
        self.alpha_y = alpha_y
        self.beta_y = self.alpha_y / 4
        self.alpha_x = self.alpha_y / 3
        # Number of Gaussian kernels
        self.n = n
        # Weights, centers and widths of Gaussian kernels
        self.w_i = np.empty((8, self.n))
        self.c_i = np.nan
        self.h_i = np.nan
        # Initial and goal dynamic conditions
        self.dq0d = np.nan
        self.dqgd = np.nan
        self.twgd = np.nan

    @staticmethod
    def __edq_from_dq(pdq, cdq):
        """
        Dual Quaternion Pose Error
        
        :param DualQuaternion dq: Previous Pose
        :param DualQuaternion dqg: Current Pose
        :return: Dual Quaternion Pose Error
        :rtype: DualQuaternion
        """
        return 2 * dq_log(pdq.quaternion_conjugate() * cdq)

    @staticmethod
    def __edq_from_dq_list(dq_list:np.ndarray) -> np.ndarray:
        """
        Dual Quaternion Error w.r.t. Goal from Dual Quaternion List
        -----
        ### Parameters
        @dq_list: Dual Quaternion Pose List
        ### Returns
        @edq: Dual Quaternion Pose Error List
        """
        dqg = dq_list[-1]
        edq = np.array([DQDMP.__edq_from_dq(dq_i, dqg) for dq_i in dq_list], dtype=DualQuaternion)
        return edq

    @staticmethod
    def __fn_rct(x:np.ndarray, psi:np.ndarray, w:np.ndarray) -> DualQuaternion:
        """
        Forcing term reconstruction function
        -----
        ### Parameters
        @x: Canonical System vector
        @psi: Gaussian Kernel array
        @w: Gausian Kernel weigths array
        ### Returns
        @fn: Forcing term Dual Quaternion
        """
        fn = DualQuaternion.from_dq_array(((x * np.inner(psi, w)) / np.sum(psi, 1)).ravel())
        return fn

    @staticmethod
    def dq_from_pose(r:np.ndarray, p:np.ndarray) -> np.ndarray:
        """
        Dual Quaternion list from Pose Data
        -----
        ### Parameters
        @r: Quaternion parameters Orientation List (w, x, y, z)
        @p: Cartesian Position List (x, y, z)
        ### Returns
        @dq: Dual Quaternion list
        """
        dq =  np.empty(r.shape[0], dtype=DualQuaternion)
        for i, (ri, pi) in enumerate(zip(r, p)):
            dq[i] = DualQuaternion.from_quat_pose_array(np.append(ri, pi))
        return dq

    @staticmethod
    def pose_from_dq(dq:np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """
        Pose Data from Dual Quaternion list
        -----
        ### Parameters
        @dq: Dual Quaternion List
        ### Returns
        @(r, p): Orientation (w, x, y, z) and Position (x, y, z) vector arrays
        """
        r, p = [], []
        for dqi in dq:
            pt = dqi.quat_pose_array()
            r.append(pt[0:4])
            p.append(pt[4:7])
        r = np.array(r).reshape((-1, 4))
        p = np.array(p).reshape((-1, 3))
        return r, p

    def __fit_dtw(self, tw:DualQuaternion, edq:DualQuaternion, fn:DualQuaternion) -> DualQuaternion:
        """
        Fit second order system
        -----
        ### Parameters
        @tw: Twist as Dual Quaternion
        @edq: Dual Quaternion Error w.r.t. Goal
        @fn: Forcing Term as Dual Quaternion
        ### Returns
        @dtw: Twist Derivative w.r.t. time Dual Quaternion
        """
        dtw = fn + self.alpha_y * ((self.beta_y * edq) + (-1 * tw))
        return dtw

    def __fn_learn(self, dtw:np.ndarray, tw:np.ndarray, edq:np.ndarray) -> np.ndarray:
        """
        Forcing function learning
        -----
        ### Parameters
        @tw: Twist DualQuaternion List
        @dtw: Twist Derivative w.r.t. time DualQuaternion List
        @edq: Dual Quaternion Error w.r.t. Goal DualQuaternion List
        ### Returns
        @fn: Forcing Term DualQuaternion List
        """
        fn = dtw + (-1 * self.alpha_y * ((self.beta_y * edq) + (-1 * tw)))
        return fn

    def __set_cfc(self, tg:float) -> None:
        """
        Computes coefficients for canonical system,
        centers and widths for Gaussian Kernels.
        -----
        ### Parameters
        @tg: Time period to reach the goal
        """
        # Coefficient for canonical system adjusted to time period
        self.alpha_x = (self.alpha_y / 3) * (1 / tg)
        # Centers and Widths of Gaussian kernels
        self.c_i = np.exp(-self.alpha_x * ((np.linspace(1, self.n, self.n) - 1) / (self.n - 1)) * tg)
        self.h_i = self.n / np.power(self.c_i, 2)

    def __can_sys(self, t:np.ndarray, tau:float=1) -> "tuple[np.ndarray, np.ndarray]":
        """
        Computes Canonical System and Gaussian Kernels
        -----
        ### Parameters
        @t: Time vector, (m)
        @tau: Time scaling parameter
        ### Returns
        @(x, psi_i): Canonical System vector and Gaussian Kernel array
        """
        # Canonical system
        x = np.exp(-self.alpha_x * t / tau).reshape((-1, 1))
        # Gaussian kernels
        psi_i = np.empty([len(x), self.n], dtype=float)
        for i in range(self.n):
            psi_i[:, i] = np.exp(-1 * np.inner(self.h_i[i], np.power(x - self.c_i[i], 2))).reshape(-1)
        return x, psi_i

    def __w_learn(self, x:np.ndarray, psi_i:np.ndarray, fd:np.ndarray) -> np.ndarray:
        """
        Gaussian Kernel weights learning function
        -----
        ### Parameters
        @x: Canonical System vector
        @psi_i: Gaussian Kernel array
        @fd: Forcing Term array
        ### Returns
        @w_i: Gaussian Kernel weights
        """
        fd = dql_to_npa(fd)
        # Compute weights
        w_i = np.empty([fd.shape[1], self.n])
        for i in range(self.n):
            psi_m = np.diag(psi_i[:, i])
            w_i[:, i] = np.dot(np.dot(x.T, psi_m), fd) / np.dot(np.dot(x.T, psi_m), x)
        return w_i

    def reset_t(self, t:float=0.0) -> None:
        """
        Reset Starting Time
        -----
        ### Parameters
        @t: Starting time value (optional)
        """
        self.prev_t = t

    def train_model(self, t:np.ndarray, dq:np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
        """
        Get DMP Model from Dual Quaternion pose list
        -----
        ### Parameters
        @t: Time vector
        @dq: Dual Quaternion pose data
        ### Returns
        @(w_i, x): Gaussian Kernel weights array and Canonical System vector
        """
        # Coefficient for canonical system, Centers and Widths of Gaussian kernels
        self.__set_cfc(t[-1]-t[0])
        # Compute Canonical system and Gaussian kernels
        x, psi_i = self.__can_sys(t)
        # Time derivatives from q
        edq = self.__edq_from_dq_list(dq)
        tw = twist_from_dq_list(t, dq)
        dtw = dx_dt(t, tw)
        # Store training initial and goal conditions
        self.dq0d = dq[0]
        self.dqgd = dq[-1]
        self.twgd = tw[-1]
        # Forcing term from captured data
        fd = self.__fn_learn(dtw, tw, edq)
        # Weight learning
        w_i = self.__w_learn(x, psi_i, fd)
        self.w_i = w_i
        return w_i, x

    def fit_step(self, t:float, dq:DualQuaternion, tw:DualQuaternion, dqg:DualQuaternion, tau:float=1) -> 'tuple[DualQuaternion, DualQuaternion]':
        """
        Step-fit DMP Model to Dual Quaterion conditions
        -----
        ### Parameters
        @t: Current time
        @dq: Current Dual Quaternion pose
        @tw: Current Twist
        @dqg: Goal Dual Quaternion pose
        @tau: Time scaling parameter
        ### Return
        @(dq_n, tw_n): Next Dual Quaternion pose and Next Twist
        """
        # Timestep
        dt = (t - self.prev_t)
        # Recalculate Canonical system
        x, psi_i = self.__can_sys(np.array([t]), tau)
        # Reconstruct forcing term
        fn = self.__fn_rct(x, psi_i, self.w_i)
        # Reconstruct pose
        edq = self.__edq_from_dq(dq, dqg)
        dtw = (1 / tau) * self.__fit_dtw(tau * tw, edq, fn)
        tw_n = tw + ((dt / tau) * dtw)
        dq_n = next_dq_from_twist(dt, dq, tw_n)
        # Store current time
        self.prev_t = t
        return dq_n, tw_n

    def fit_model(self, t:np.ndarray, dq0:DualQuaternion, tw0:DualQuaternion, dqg:DualQuaternion, tau:float=1) -> 'tuple[np.ndarray, np.ndarray]':
        """
        Fit DMP Model to Dual Quaternion conditions
        -----
        ### Parameters
        @t: Time vector
        @dq0: Initial Dual Quaternion pose
        @tw0: Initial Twist
        @dqg: Goal Dual Quaternion pose
        @tau: Time scaling parameter
        ### Returns
        @(dq_arr, tw_arr): Reconstructed Dual Quaternion pose and Twist lists
        """
        # Initial conditions
        dq = dq0
        tw = tw0
        # Reconstruct pose
        dq_arr = np.empty(t.shape[0], dtype=DualQuaternion)
        tw_arr = np.empty(t.shape[0], dtype=DualQuaternion)
        for i, t_i in enumerate(t):
            dq_arr[i] = dq
            tw_arr[i] = tw
            dq, tw = self.fit_step(t_i, dq, tw, dqg, tau=tau)
        return dq_arr, tw_arr


class PTDQDMP(DQDMP, ProjectileLaunching):
    """
    A class for computing Projectile Throwing Dynamic Movement Primitives (DMP)
    from full pose Dual Quaternion data
    """
    def __init__(self, n:int=20, alpha_y:float=4) -> None:
        """
        Projectile Throwing Dual Quaternion Dynamic Movement Primitives
        -----
        ### Parameters
        @n: Number of Gaussian kernels to reconstruct the forcing term
        @alpha_y: Damping coefficient
        """
        super().__init__(n=n, alpha_y=alpha_y)
        ProjectileLaunching.__init__(self)

    def adapt_poses(self, p_t:Quaternion) -> 'tuple(DualQuaternion, DualQuaternion, float)':
        """
        Adapt poses to throwing target
        -----
        ### Parameters
        @p_t: Pure Quaternion target point
        ### Returns
        @(dqn0, dqng, tau): Adapted Dual Quaternion initial and goal poses, and time scaling factor tau
        """
        p_g = 2 * dq_log(self.dqgd).q_d
        
        v_g = vel_from_twist(self.dqgd, self.dqgd * self.twgd * self.dqgd.quaternion_conjugate())
        
        n_z = Quaternion(vector=[0.0, 0.0, 1.0])
        n_g = (.5 * (v_g * n_z - n_z * v_g)).normalised

        q_r = self.estimate_plane_rotation(n_g, p_g, p_t)
        # n_p = q_r.rotate(n_g)
        p_p = q_r.rotate(p_g)

        _, v_0 = self.optimal_v_launch(p_p, p_t)

        q_v = quat_rot(q_r.conjugate.rotate(v_0), v_g)

        dq_off_0 = DualQuaternion.from_quat_pose_array(np.append(q_r.elements, [0, 0, 0]))
        dq_off_1 = DualQuaternion.from_quat_pose_array(np.append(q_v.elements, [0, 0, 0]))

        dqn0, dqng = self.correct_poses(self.dq0d, self.dqgd, dq_off_0, dq_off_1)
        tau = v_g.norm / v_0.norm

        return dqn0, dqng, tau

    def aim_model(self, t:np.ndarray, p_t:DualQuaternion) -> 'tuple[np.ndarray, np.ndarray]':
        """
        Aim DMP Model towards a Cartesian target
        -----
        ### Parameters
        @t: Time vector to reconstruct the DMP model
        @p_t: Pure Quaternion Cartesian target
        ### Returns
        @(dqn0, dqng): Reconstructed Dual Quaternion pose and Twist lists
        """
        tw_0 = DualQuaternion.from_dq_array(np.zeros(8))
        dq_0, dq_g, tau = self.adapt_poses(p_t)
        dq_rec, tw_rec = self.fit_model(t, dq_0, tw_0, dq_g, tau=tau)
        return dq_rec, tw_rec
        

def main():
    pass


if __name__ == '__main__':
    main()
