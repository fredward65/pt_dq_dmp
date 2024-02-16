#!/usr/bin/env python

import numpy as np
from copy import deepcopy as dcp
from dual_quaternions import DualQuaternion
from pyquaternion import Quaternion as Quat


def dx_dt(t:np.ndarray, x:np.ndarray) -> np.ndarray:
    """
    Differentiate x w.r.t. t
    -----
    ### Parameters
    @t: Time vector, (m)
    @x: Data vector, (m, dim)
    ### Returns
    @dx: Differentiated data vector, (m, dim)
    """
    flag = x.dtype == DualQuaternion
    x = dql_to_npa(x) if flag else x
    # Timestep vector
    dt = np.diff(t, axis=0).reshape((-1, 1))
    # Differentiation
    dx = np.divide(np.diff(x, axis=0), dt)
    dx = np.append(dx, [dx[-1, :]], axis=0)
    dx = npa_to_dql(dx) if flag else dx
    return dx


def dql_to_npa(dql:np.ndarray) -> np.array:
    """
    Dual Quaternion list to numpy Array
    -----
    ### Parameters
    @dql: Dual Quaternion List
    ### Returns
    @npa: numpy Array
    """
    npa = np.array([dqi.dq_array() for dqi in dql])
    return npa


def npa_to_dql(npa:np.ndarray) -> np.ndarray:
    """
    numpy Array to Dual Quaternion list
    -----
    ### Parameters
    @npa: numpy Array
    ### Returns
    @dql: Dual Quaternion List
    """
    dql = np.array([DualQuaternion.from_dq_array(xi) for xi in npa],
                   dtype=DualQuaternion)
    return dql


def quat_rot(a:Quat, b:Quat) -> Quat:
    """
    Quaternion Rotation from two pure quaternions
    -----
    ### Parameters
    @a: Pure Quaternion a
    @b: Pure Quaternion b
    ### Returns
    @q: Quaternion Rotation from a to b
    """
    axb = a * b
    q = (axb.norm - axb.conjugate).normalised
    return q


def q_rot_from_vec(vec:np.ndarray) -> Quat:
    """
    Quaternion Orientation from Approach Vector
    -----
    ### Parameters
    @vec: Approach Vector (Z Vector of the Body Frame)
    ### Returns
    @q_r: Quaternion Orientation
    """
    vex = np.multiply(np.copy(vec), np.array([1, 1, 0]))
    q_z = quat_rot(Quat(vector=[0, 0, 1]), Quat(vector=vec).normalised)
    q_x = quat_rot(Quat(vector=[1, 0, 0]), Quat(vector=vex).normalised)
    q_r = q_z * q_x
    return q_r


def main():
    pass


if __name__ == '__main__':
    main()
