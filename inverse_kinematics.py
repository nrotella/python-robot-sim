#!/usr/bin/env/python

from __future__ import print_function

import copy
import numpy as np
import transformations as tf

class InverseKinematics(object):
    """
    This class implements differential inverse kinematics.

    """

    def __init__(self, robot_params, joint_state):
        """
        Args:
            robot: Instance of RobotParams class which defines kinematic/dynamic parameters.

        Returns:
            (None)

        """

        # Store the robot parameters:
        self.robot = robot_params

        self.ik_pos_p_gain = 5e-1
        self.ik_ori_p_gain = 1e-1

        # Deep copy the initial joint state to create an internal IK joint state
        # to be integrated using the results of differential IK:
        self.ik_jstate = copy.deepcopy(joint_state)

    def update(self, kinematics,
               endeff_pos_ref, endeff_ori_ref,
               endeff_linvel_ref, endeff_angvel_ref,
               dt):
        """
        Perform inverse kinematics using the given kinematics object and integrate the
        internal IK joint state using the result.

        Args:
            kinematics: Kinematics object which stores the current kinematics of the robot
            endeff_pose_des: HomogeneousTransform object for the desired endeffector pose
            endeff_vel_des: 6x1 numpy vector of desired linear/angular endeffector velocity
            dt: Control period, in seconds

        Outputs:
            (None)

        """

        # Compute the desired endeffector velocity from the reference endeffector velocity and
        # a P controller on desired endeffector pose, separated into linear and angular parts:
        endeff_vel_des = np.zeros(6)

        # Desired linear velocity:
        endeff_vel_des[:3] = endeff_linvel_ref + self.ik_pos_p_gain*(
            endeff_pos_ref - kinematics.h_tf_links[-1].t())

        # Desired angular velocity:
        endeff_vel_des[3:] = endeff_angvel_ref + self.ik_ori_p_gain*(
            tf.log_map_SO3(endeff_ori_ref.transpose().dot(kinematics.h_tf_links[-1].R())))

        # Invert the Jacobian to get the joint velocities and store them:
        jac_pinv = np.linalg.pinv(kinematics.endeff_jac)
        qdot = jac_pinv.dot(endeff_vel_des)
        for i in range(self.robot.n_dofs):
            self.ik_jstate[i].thd = qdot[i]

        # Integrate the resulting joint velocities to get the internal IK joint posture:
        for i in range(self.robot.n_dofs):
            self.ik_jstate[i].th += dt*qdot[i]

        return copy.copy(self.ik_jstate)
