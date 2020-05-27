#!/usr/bin/env/python

from __future__ import print_function

import copy
import numpy as np

import transformations as tf
import robot_defs

class RecursiveNewtonEuler(object):
    def __init__(self, robot_params, robot_state, kinematics):

        # Shallow copies of the state objects:
        self.robot = robot_params
        self.rob = robot_state
        self.kin = kinematics

    def compute_grav_comp(self, joint_state):
        # Assume the base is fixed and set joint derivatives to zero to compute static torques:
        base_vel = np.zeros(6)
        base_acc = np.zeros(6)
        g0 = np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])

        # Create a deep copy of the joint state to modify in computing dynamics:
        jstate = copy.deepcopy(joint_state)
        for joint in jstate:
            joint.thd = 0.0
            joint.thdd = 0.0

        tau_grav_comp = self.compute_joint_torques(base_vel, base_acc-g0, jstate)

        return tau_grav_comp

    def compute_inertia_mat(self, joint_state):
        # Construct the inertia matrix from column i by setting qddoti=1,qddotj=0 for i=/=j
        # and qdot = 0 as well as removing grav acc from the base acc:
        base_vel = np.zeros(6)
        base_acc = np.zeros(6)

        inertia_mat = np.zeros((self.robot.n_dofs, self.robot.n_dofs))

        jstate = copy.deepcopy(joint_state)
        for i in range(self.robot.n_dofs):
            # Set all joint accelerations except the ith to zero:
            for joint in jstate:
                joint.thd = 0.0
                joint.thdd = 0.0
            jstate[i].thdd = 1.0

            inertia_mat[:,i] = self.compute_joint_torques(base_vel,
                                                          base_acc,
                                                          jstate)
        return inertia_mat

    def compute_qddot(self, joint_state, joint_state_des):
        # Assume the base is fixed and set joint accelerations to zero to compute nonlinear terms:
        base_vel = np.zeros(6)
        base_acc = np.zeros(6)
        g0 = np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])

        # Create a deep copy of the joint state to modify in computing dynamics:
        jstate = copy.deepcopy(joint_state)
        for joint in jstate:
            joint.thdd = 0.0

        tau_nonlin = self.compute_joint_torques(base_vel, base_acc-g0, jstate)

        tau_joint_lim = self.compute_joint_lim_torques(jstate)

        # Construct the inertia matrix from column i by setting qddoti=1,qddotj=0 for i=/=j
        # and qdot = 0 as well as removing grav acc from the base acc:
        base_acc = np.zeros(6)

        inertia_mat = np.zeros((self.robot.n_dofs, self.robot.n_dofs))

        for i in range(self.robot.n_dofs):
            # Set all joint accelerations except the ith to zero:
            for joint in jstate:
                joint.thd = 0.0
                joint.thdd = 0.0
            jstate[i].thdd = 1.0

            inertia_mat[:,i] = self.compute_joint_torques(base_vel,
                                                          base_acc,
                                                          jstate)

        # Compute the joint accelerations:
        tau_applied = [j.u for j in joint_state_des]

        return np.linalg.inv(inertia_mat).dot(tau_applied + tau_joint_lim - tau_nonlin)

    def compute_joint_torques(self, base_vel, base_acc, joint_state):

        # Create the local internal force array and output joint torque array:
        link_frc = [np.zeros(6) for i in range(self.robot.n_links + 1)]
        tau = np.zeros(self.robot.n_dofs)

        # First, compute the link motion from the current base and joint state:
        joint_tf, link_tf, link_com, link_vel, link_acc, com_vel, com_acc = self.kin.compute_link_motion(tf.HomogeneousTransform(), base_vel, base_acc, joint_state)

        # Explicitly set the endeffector forces to zero (for now):
        link_frc[-1] = np.zeros(6)

        # Backwards recursion using kinematics-computed link vel/acc to compute forces/moments:
        for i in range(self.robot.n_links, 0, -1):
            link_frc[i-1][:3] = link_frc[i][:3] + self.robot.link_mass[i-1]*com_acc[i]
            r_im1_i = joint_tf[i].t() - joint_tf[i-1].t() # world frame vector from prev to current link
            r_i_Ci = link_com[i] - joint_tf[i].t()
            link_inertia_W = joint_tf[i].R().dot(self.robot.link_inertia[i-1]).dot(joint_tf[i].R().T) # world frame link inertia
            link_frc[i-1][3:] = np.cross(-link_frc[i-1][:3], (r_im1_i + r_i_Ci)) + \
                                link_frc[i][3:] + \
                                np.cross(link_frc[i][:3], r_i_Ci) + \
                                link_inertia_W.dot(link_acc[i][3:]) + \
                                np.cross(link_vel[i][3:], link_inertia_W.dot(link_vel[i][3:]))

            tau[i-1] = link_frc[i-1][3:].dot(joint_tf[i-1].Rz()) + \
                       robot_defs.VISCOUS_DAMP_COEFF * joint_state[i-1].thd

        return tau

    def compute_joint_lim_torques(self, joint_state):

        tau_joint_lim = np.zeros(self.robot.n_dofs)

        for i in range(self.robot.n_dofs):
            if joint_state[i].th > self.robot.joint_max[i]:
                tau_joint_lim[i] = robot_defs.JOINT_LIM_STIFF*(
                    self.robot.joint_max[i] - joint_state[i].th)
            elif joint_state[i].th < self.robot.joint_min[i]:
                tau_joint_lim[i] = robot_defs.JOINT_LIM_STIFF*(
                    self.robot.joint_min[i] - joint_state[i].th)

        return tau_joint_lim
