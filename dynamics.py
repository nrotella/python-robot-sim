#!/usr/bin/env/python

from __future__ import print_function

import copy
import numpy as np
import scipy.linalg as sp

import transformations as tf
import robot_defs

class RecursiveNewtonEuler(object):
    """
    Class for computing the internal wrenches exerted on links in a kinematic chain using the
    Newton-Euler formulation, translate these wrenches into joint torques, and compute various
    dynamics quantities. The full dynamic model is:

        M(q)\ddot{q} + C(q,\dot{q}) + g(q) = \tau + J(q)^{T}F

    where the joint state is:

        q:        n_dofs x 1 vector of joint angles (robot "configuration")
        \dot{q}:  n_dofs x 1 vector of joint velocities
        \ddot{q}: n_dofs x 1 vector of joint accelerations

        \tau:     n_dofs x 1 vector of applied (actuated) joint torques

    and the dynamic quantities are:

        M(q): n_dofs x n_dofs symmetric, positive-definite mass-inertia matrix
        C(q): n_dofs x 1 vector of torques due to inertial forces, friction etc
        g(q): n_dofs x 1 vector of torques due to the effect of gravity
        J(q): 6 x n_dofs Jacobian matrix, whose transpose maps an endeffector wrench to joint torques

    With the exception of the Jacobian (which is a kinematics-computed quantity), these terms
    are computed in the update() via repeated applications of the recursive Newton-Euler
    rigid body dynamics algorithm.

    """

    def __init__(self, robot_params, robot_state, kinematics):
        """ Initialize the dynamics computation class.

        Initializes the class used for dynamics computations by storing references to the robot
        parameters, current state and kinematics objects.

        Args:
            robot_params: Instance of RobotParams class which specifies robot physical parameters.
            kinematics: Instance of Kinematics class which performs kinematics computations.

        Returns:
            (None)

        """

        # Shallow copies of the state objects:
        self.robot = robot_params
        self.kin = kinematics

        # Dynamics quantities to be computed in update:
        self.inertia_mat = np.zeros((self.robot.n_dofs, self.robot.n_dofs))
        self.tau_nonlin = np.zeros(self.robot.n_dofs)
        self.tau_joint_lim = np.zeros(self.robot.n_dofs)

        # Vector of joint accelerations resulting from applied torques:
        self.qddot = np.zeros(self.robot.n_dofs)

    def update(self, joint_state):
        """ Update the dynamics object by computing various dynamics quantities.

        Updates the dynamics using the current joint state by computing the mass-inertia matrix,
        the nonlinear torques and the joint limit torques.  These are used in other functions to
        simulate the system (compute output joint accelerations) and compute inverse dynamics from
        desired joint accelerations.

        The nonlinear joint torques are the combined effects of gravity, centripetal/Coriolis forces,
        and joint friction. These torques are computed by setting the base generalized velocity to zero,
        setting the base generalized acceleration to include the effects of gravity, and setting the
        joint accelerations to zero.

        The mass-inertia matrix is the mapping between joint accelerations and inertial torques
        resulting from them; it is a function of the physical properties of the robot (link masses and
        inertias) as well as the current robot configuration.  It is computed by setting base state
        derivatives to zero and setting one joint acceleration to 1 at a time to compute each column.

        Args:
            joint_state: Array of n_dofs JointState objects containing current joint information.

        Returns:
            (None)

        """

        # Assume the base is fixed and set joint accelerations to zero to compute nonlinear terms:
        base_vel = np.zeros(6)
        base_acc = np.zeros(6)
        g0 = np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])

        # Create a deep copy of the joint state to modify in computing dynamics:
        jstate = copy.deepcopy(joint_state)

        # Compute the nonlinear torques by setting joint accelerations to zero:
        for joint in jstate:
            joint.thdd = 0.0

        self.tau_nonlin = self.compute_joint_torques(base_vel, base_acc-g0, jstate)

        # Compute the joint limit torques:
        self.tau_joint_lim = self.compute_joint_lim_torques(jstate)

        # Construct the inertia matrix from column i by setting qddoti=1,qddotj=0 for i=/=j
        # and qdot = 0 as well as removing grav acc from the base acc:
        base_acc = np.zeros(6)

        # Set the external forces to zero when using NE to compute dynamics quantities:
        for joint in jstate:
            joint.fext = np.zeros(3)

        for i in range(self.robot.n_dofs):
            # Set all joint accelerations except the ith to zero:
            for joint in jstate:
                joint.thd = 0.0
                joint.thdd = 0.0
            jstate[i].thdd = 1.0

            self.inertia_mat[:,i] = self.compute_joint_torques(base_vel,
                                                               base_acc,
                                                               jstate)

    def compute_joint_torques(self, base_vel, base_acc, joint_state):
        """ Compute the joint torques resulting from applied and inertial forces.

        Computes the joint torques resulting from the combination of applied external forces and
        inertial forces due to base/joint motion using the recursive Newton-Euler formulation. The
        computation of resulting joint torques proceeds in two steps:

        1. Starting from the base acceleration, compute the acceleration of each link (angular
        acceleration of the link and linear acceleration of the link COM) down the chain using
        kinematics.

        2. Starting from the wrench applied at the endeffector, compute the force and moment
        (Newton and Euler) acting on the preceding link in the chain due to the link motion from
        step one, up to the base.

        In terms of computational complexity, each of these steps is linear in the number of DoFs
        and thus the overall Newton-Euler algorithm complexity is O(n_dofs).

        """

        # Create the local internal force array and output joint torque array:
        link_frc = [np.zeros(6) for i in range(self.robot.n_links + 1)]
        tau = np.zeros(self.robot.n_dofs)

        # First, compute the link motion from the current base and joint state:
        joint_tf, link_tf, link_com, link_vel, link_acc, com_vel, com_acc = self.kin.compute_link_motion(tf.HomogeneousTransform(), base_vel, base_acc, joint_state)

        # Explicitly set the endeffector forces to zero (for now):
        link_frc[-1] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Backwards recursion using kinematics-computed link vel/acc to compute forces/moments:
        for i in range(self.robot.n_links, 0, -1):
            link_frc[i-1][:3] = link_frc[i][:3] + self.robot.link_mass[i-1]*com_acc[i] + joint_state[i-1].fext
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

    def compute_qddot(self, joint_state_des):
        """ Compute the joint accelerations resulting from the applied torques.

        Computes the joint accelerations (qddot) given the desired torques (\tau) specified
        in the input desired JointState array and the current dynamics mode (after calling
        update()).  Joint accelerations used to simulate the system are computed as:

        qddot = M^{-1}(\tau + C(q,\dot{q}) + g(q))

        where the mass-inertia matrix inverse is computed using a Cholesky decomposition for
        efficiency and numerical stability.

        """

        # Get the applied joint torques from the desired state:
        tau_applied = [j.u for j in joint_state_des]

        # Compute the resulting joint accelerations, using Cholesky decomposition:
        qddot = sp.cho_solve(sp.cho_factor(self.inertia_mat), tau_applied +
                             self.tau_joint_lim - self.tau_nonlin)

        return qddot

    def compute_grav_comp(self, joint_state):
        """ Computes the torques required to compensate for the effect of gravity.

        Computes the torques required to compensate for the joint torques induced by gravity.
        This is the static portion of the nonlinear torques and thus must be computed explicitly
        if needed eg for pure gravity compensation control.

        Args:
            joint_state: Array of n_dofs JointState objects containing current joint information.

        Returns:
            n_dofs x 1 numpy array: The torques required to compensate gravity.

        """

        # Assume the base is fixed and set joint derivatives to zero to compute static torques.
        # Note that static torques may be extended to include static friction, in which case this
        # will compute friction plus gravity torques.
        base_vel = np.zeros(6)
        base_acc = np.zeros(6)
        g0 = np.array([0.0, 0.0, -9.81, 0.0, 0.0, 0.0])

        # Create a deep copy of the joint state to modify in computing dynamics:
        jstate = copy.deepcopy(joint_state)
        for joint in jstate:
            joint.fext = np.zeros(3)  # set external forces to zero for NE unless simulating
            joint.thd = 0.0
            joint.thdd = 0.0

        # Compute the gravity-induced torques using the Newton-Euler algorithm:
        tau_grav_comp = self.compute_joint_torques(base_vel, base_acc-g0, jstate)

        return tau_grav_comp

    def compute_joint_lim_torques(self, joint_state):
        """ Compute reaction torques which simulate joint limits.

        Computes the reaction torques which simulate joint limits be checking whether each
        joint is beyond the min/max joint limit and computing a torsional spring-like reaction
        torque using a defined (very high) stiffness.  This effectively simulates the joint
        hitting a physical endstop and being forced away due to the deformation of the endstop.

        Args:
            joint_state: Array of n_dofs JointState objects containing current joint information.

        Returns:
            n_dofs x 1 numpy array: The torques resulting from the joint limit simulation.

        """

        tau_joint_lim = np.zeros(self.robot.n_dofs)

        for i in range(self.robot.n_dofs):
            if joint_state[i].th > self.robot.joint_max[i]:
                tau_joint_lim[i] = robot_defs.JOINT_LIM_STIFF*(
                    self.robot.joint_max[i] - joint_state[i].th)
            elif joint_state[i].th < self.robot.joint_min[i]:
                tau_joint_lim[i] = robot_defs.JOINT_LIM_STIFF*(
                    self.robot.joint_min[i] - joint_state[i].th)

        return tau_joint_lim
