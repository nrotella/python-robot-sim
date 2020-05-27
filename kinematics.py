#!/usr/bin/env/python

from __future__ import print_function

import numpy as np
import transformations as tf
import robot_defs


class RobotParams(object):
    def __init__(self):

        # # N-LINK PLANAR:
        # self.n_dofs = 3
        # self.dh_a = 0.3*np.ones(self.n_dofs)
        # self.dh_alpha = np.zeros(self.n_dofs)
        # self.dh_th = np.zeros(self.n_dofs)
        # self.dh_d = np.zeros(self.n_dofs)
        # self.default_th = 0.0*np.ones(self.n_dofs)

        # # ANTHROPOMORPHIC:
        # self.n_dofs = 3
        # self.dh_a = np.array([0.0, 0.3, 0.3])
        # self.dh_alpha = np.array([np.pi/2, 0.0, 0.0])
        # self.dh_th = np.array([0.0, 0.0, 0.0])
        # self.dh_d = np.array([0.5, 0.0, 0.0])
        # self.default_th = np.array([0.0, 0.0, 0.0])
        # self.joint_min = np.array([-np.pi/2, -np.pi/2, -np.pi/2])
        # self.joint_max = np.array([np.pi/2, np.pi/2, np.pi/2])

        # # SPHERICAL WRIST:
        # self.n_dofs = 3
        # self.dh_a = np.array([0.0, 0.1, 0.0])
        # self.dh_alpha = np.array([-np.pi/2, np.pi/2, 0.0])
        # self.dh_th = np.array([0.0, 0.0, 0.0])
        # self.dh_d = np.array([0.5, 0.0, 0.5])
        # self.default_th = np.array([0.0, 0.0, 0.0])

        # # ANTHROPOMORPHIC WITH SPHERICAL WRIST:
        # self.n_dofs = 6
        # self.dh_a = np.array([0.0, 0.3, 0.3, 0.0, 0.1, 0.0])
        # self.dh_alpha = np.array([np.pi/2, 0.0, np.pi/2, -np.pi/2, np.pi/2, 0.0])
        # self.dh_th = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # self.dh_d = np.array([0.5, 0.0, 0.0, 0.1, 0.0, 0.2])
        # self.default_th = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # self.joint_min = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2])
        # self.joint_max = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])

        # HUMANOID ARM:
        self.n_dofs = 7
        self.dh_a = np.array([0.0, 0.0, 0.3, 0.3, 0.1, 0.1, 0.1])
        self.dh_alpha = np.array([np.pi/2, np.pi/2, -np.pi, np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2])
        self.dh_th = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.dh_d = np.array([0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.default_th = np.array([0.0, 0.0, 1.0, np.pi/2, 0.0, 1.0, 0.0])
        self.joint_min = np.array([-np.pi/2, -np.pi/2, -np.pi/2, 0.0, -np.pi/2, -np.pi/2, -np.pi/2])
        self.joint_max = np.array([np.pi/2, np.pi/2, np.pi/2, 0.75*np.pi, np.pi/2, np.pi/2, np.pi/2])

        # Link COMs are negative in local link frame since link frame is located at
        # the joint FOLLOWING the link:
        #self.n_links = np.sum(np.logical_or(self.dh_a != 0.0, self.dh_d != 0.0)) + 1
        self.n_links = self.n_dofs
        self.link_mass = 50.0 * np.ones(self.n_links)
        self.link_com_frac = self.n_links*[0.5]
        self.link_inertia = [10.0*np.eye(3) for i in range(self.n_links)]

class RobotState(object):
    def __init__(self, robot_params):
        self.robot = robot_params
        self.joint_state = [JointState(self.robot.default_th[i], self.robot.dh_d[i])
                            for i in range(self.robot.n_dofs)]
        self.joint_state_des = [JointState(self.robot.default_th[i], self.robot.dh_d[i])
                                for i in range(self.robot.n_dofs)]


class JointState(object):
    def __init__(self, th, d):

        # Joint is revolute by default.
        self.jtype = robot_defs.JTYPE_REVOLUTE

        self.th = th
        self.thd = 0.0
        self.thdd = 0.0

        self.d = d
        self.dd = 0.0
        self.ddd = 0.0

        self.u = 0.0

    def set_th(self, val):
        """ Set the joint angle. Needed for lambda functions. """
        self.th = val

    def set_thd(self, val):
        """ Set the joint velocity. Needed for lambda functions. """
        self.thd = val

    def set_d(self, val):
        """ Set the joint displacement. Needed for lambda functions. """
        self.d = val

    def set_u(self, val):
        """ Set the actuator-applied joint torque. Needed for lambda functions. """
        self.u = val


class Kinematics(object):
    """
    Basic kinematics functions collected into a class.

    """
    def __init__(self, robot_params):
        """ Initialize the kinematics class.

        Initialize the kinematics class from the given robot parameters.

        Args:
            robot: Instance of RobotParams class which defines kinematic/dynamic parameters.

        Returns:
           (None)

        """

        # Shallow copies of the state objects:
        self.robot = robot_params

        # We compute forward kinematics in the form of homogeneous transformation matrices
        # denoted h_tf for each of the n_links plus the root link (considered link 0).  For
        # now, the root link is fixed to coincide with the world origin frame.
        self.h_tf = [tf.HomogeneousTransform() for i in range(self.robot.n_dofs + 1)]
        self.h_tf_links = [tf.HomogeneousTransform() for i in range(self.robot.n_links + 1)]

        self.endeff_jac = np.zeros((6, self.robot.n_dofs))
        self.endeff_jac_prev = np.zeros((6, self.robot.n_dofs))
        self.endeff_jac_deriv = np.zeros((6, self.robot.n_dofs))

        self.link_vel = [np.zeros(6) for i in range(self.robot.n_links + 1)]
        self.link_acc = [np.zeros(6) for i in range(self.robot.n_links + 1)]

        self.link_com = [np.zeros(3) for i in range(self.robot.n_links + 1)]
        self.com_vel = [np.zeros(3) for i in range(self.robot.n_links + 1)]
        self.com_acc = [np.zeros(3) for i in range(self.robot.n_links + 1)]

        self.endeff_vel = np.zeros(6)
        self.endeff_acc = np.zeros(6)

    def initialize(self, base_tf, base_vel, base_acc, joint_state, dt):
        self.update(base_tf, base_vel, base_acc, joint_state, dt)

    def update(self, base_tf, base_vel, base_acc, joint_state, dt):
        """ Update the kinematics class using measured data.

        In the adopted convention, joint i-1 moves link i-1, and the link i-1
        frame is located at the origin of joint i:

        joint i-1 O==link i-1===O joint i
                                \\
                                 \\ link i
                                  \\
                                   O joint i+1

        - The axis z_{i} is always the axis for joint i+1

        - The axis x_{i} is chosen as the common normal to z_{i-1} and z_{i},
        +ve from joint i to i+1 (points along link i)

        - The axis y_{i} is chosen to complete a right-handed frame with x_{i} and z_{i]

        Exceptions:
        - For frame 0, only z_{0} is defined; x_{0} is abirtrary
        - For frame N, z_{N} is not unique since joint N+1 doesn't exist; if revolute,
        align with z_{N-1}
        - When z_{i-1} and z_{i} are parallel, common normal (x_{n}) is not unique
        - When z_{i-1] and z_{i} intersect, +ve direction of x_{i} is arbitrary
        - For prismatic joints, only z_{i-1} is specified

        We seek transformations which relate a vector in link frame i to the same vector
        in link frame i-1.

        We can write this incremental transform in terms of the state of joint i and
        geometry of link i.  This transform inherently has 6 DOF, but by having chosen
        the frames carefully as above we can express it in terms of 4 DOF known as the
        link's Denavit-Hartenberg (DH) parameters.

        DH Parameters are specified in terms of the series of individual transformations
        required to bring link frame i-1 to link frame i through an intermediate frame
        i,prime (ip):
        - alpha_{i} is the distance between origins of frame i and ip
        - d_{i} is the coordinate of frame ip long z_{i-1} (distance between ip and i-1
        along z_{i-1})
        - alpha_{i} is the angle between z_{i-1} and z_{i} about x_{i} (+ve when CCW)
        - theta_{i} is the angle between x_{i-1} and x_{i} about z_{i-1} (+ve when CCW)

        The DH parameters are used to compute incremental link transforms as follows:
        - Starting with the link i-1 frame, translate by d_{i} along z_{i-1} and
        simultaneously rotate by theta_{i} about z_{i-1}; this aligns with frame i,prime
        and is computed as

        A_{ip}^{i-1} = [cos(theta_{i})    -sin(theta_{i})    0         0]
                       [sin(theta_{i})     cos(theta_{i})    0         0]
                       [0                  0                 1     d_{i}]
                       [0                  0                 0         1]

        - Next, translate by a_{i} along axis x_{ip} (has the same direction as x_{i})
        and simultaneously rotate about this axis by alpha_{i}; this aligns with frame i
        and is denoted A_{i}^{ip}

        A_{i}^{ip} =   [1    0                  0                 a_{i}]
                       [0    cos(alpha_{i})    -sin(alpha_{i})        0]
                       [0    sin(alpha_{i})     cos(alpha_{i})        0]
                       [0    0                  0                     1]

        - Finally, compose the full transformation as A_{i}^{i-1} = A_{ip}^{i-1} * A_{i]^{ip}
          which transforms from link i-1 frame to link i frame:

        A_{i}^{i-1} = A_{ip}^{i-1} * A_{i}^{ip}

        We can compute the transform relating any link i frame to the world as:

        self.h_tf[i] = A_{i}^{W} = A_{W}^{0} * A_{1}^{0} * ... * A_{i}^{i-1}

        where A_{W}^{0} is the 4x4 identity matrix for a fixed-base robot.

        """

        # Compute the transformations and motion of all links:
        self.h_tf, self.h_tf_links, self.link_com, self.link_vel, self.link_acc, self.com_vel, self.com_acc = self.compute_link_motion(base_tf, base_vel, base_acc, joint_state)

        # Compute the 6DOF endeffector Jacobian by building up entries DOF by DOF:
        for i in range(1, self.robot.n_links + 1):
            if joint_state[i-1].jtype == robot_defs.JTYPE_PRISMATIC:
                self.endeff_jac[:3, i-1] = self.h_tf[i-1].Rz()
                self.endeff_jac[3:, i-1] = 0.0
            elif joint_state[i-1].jtype == robot_defs.JTYPE_REVOLUTE:
                self.endeff_jac[:3, i-1] = np.cross(self.h_tf[i-1].Rz(),
                                             self.h_tf[-1].t()-self.h_tf[i-1].t())
                self.endeff_jac[3:, i-1] = self.h_tf[i-1].Rz()

        # Compute the 6DOF endeffector Jacobian derivative numerically:
        self.endeff_jac_deriv = (1.0 / dt) * (self.endeff_jac - self.endeff_jac_prev)
        self.endeff_jac_prev = np.copy(self.endeff_jac)

        # Compute the 6DOF endeffector velocity:
        self.endeff_vel = self.endeff_jac.dot(np.array([j.thd for j in joint_state]))
        # Compute the 6DOF endeffector acceleration:
        self.endeff_acc = self.endeff_jac_deriv.dot(np.array([j.thd for j in joint_state])) + \
                          self.endeff_jac.dot(np.array([j.thdd for j in joint_state]))

    def compute_link_motion(self, base_tf, base_vel, base_acc, joint_state):

        # Create the output arrays:
        joint_tf = [tf.HomogeneousTransform() for i in range(self.robot.n_dofs + 1)]
        link_tf = [tf.HomogeneousTransform() for i in range(self.robot.n_links + 1)]

        link_com = [np.zeros(3) for i in range(self.robot.n_links + 1)]
        com_vel = [np.zeros(3) for i in range(self.robot.n_links + 1)]
        com_acc = [np.zeros(3) for i in range(self.robot.n_links + 1)]

        link_vel = [np.zeros(6) for i in range(self.robot.n_links + 1)]
        link_acc = [np.zeros(6) for i in range(self.robot.n_links + 1)]

        # Compute homogeneous transformations specifying each link's pose relative to the
        # world frame:
        joint_tf[0].mat = base_tf.mat
        link_tf[0].mat = joint_tf[0].mat

        # We start with h_tf[1] which corresponds to link frame 0, since the base is included:
        for i in range(self.robot.n_dofs):

            # First, get the DH parameters for the link:
            a = self.robot.dh_a[i]
            alpha = self.robot.dh_alpha[i]
            d = joint_state[i].d
            th = joint_state[i].th

            # Use the homogeneous transformation for link i-1 and the incremental homogeneous
            # transformation between link i-1 and link i frames to compute the transformation
            # for link i:
            joint_tf[i+1].mat = joint_tf[i].mat.dot(
                np.array([[np.cos(th), -np.sin(th)*np.cos(alpha), np.sin(th)*np.sin(alpha), a*np.cos(th)],
                          [np.sin(th), np.cos(th)*np.cos(alpha), -np.cos(th)*np.sin(alpha), a*np.sin(th)],
                          [0.0, np.sin(alpha), np.cos(alpha), d],
                          [0.0, 0.0, 0.0, 1.0]]))
            link_tf[i+1].mat = joint_tf[i+1].mat

        # # Compute the endeffector transform separately:
        # link_tf[-1].mat = link_tf[-2].mat.dot(np.array([[1.0, 0.0, 0.0, 0.0],
        #                                                 [0.0, 1.0, 0.0, 0.0],
        #                                                 [0.0, 0.0, 1.0, 0.2],
        #                                                 [0.0, 0.0, 0.0, 1.0]]))

        # Base link is assumed fixed, for now:
        link_vel[0] = base_vel
        link_acc[0] = base_acc
        for i in range(1, self.robot.n_links + 1):
            r_im1_i = link_tf[i].t() - link_tf[i-1].t()  # vector from previous to current link pos
            link_vel[i][3:] = (link_vel[i-1][3:] +
                               joint_state[i-1].thd*link_tf[i-1].Rz())
            link_vel[i][:3] = (link_vel[i-1][:3] +
                               np.cross(link_vel[i][3:], r_im1_i))

            link_acc[i][3:] = (link_acc[i-1][3:] +
                               joint_state[i-1].thdd*link_tf[i-1].Rz() +
                               joint_state[i-1].thd*np.cross(link_vel[i-1][3:],
                                                             link_tf[i-1].Rz()))
            link_acc[i][:3] = (link_acc[i-1][:3] +
                               np.cross(link_acc[i][3:], r_im1_i) +
                               np.cross(link_vel[i][3:], np.cross(link_vel[i][3:],
                                                                  r_im1_i)))

        # Use the link frame vel/acc to compute the COM vel/acc:
        link_com[0] = link_tf[0].t()
        com_vel[0] = base_vel
        com_acc[0] = base_acc
        for i in range(1, self.robot.n_links + 1):
            link_com[i] = joint_tf[i-1].t() + self.robot.link_com_frac[i-1] * (
                joint_tf[i].t() - joint_tf[i-1].t())  # world frame COM
            r_i_Ci = link_com[i] - joint_tf[i-1].t()  # world frame COM rel to preceding joint

            com_vel[i] = link_vel[i-1][:3] + np.cross(link_vel[i-1][3:], r_i_Ci)
            com_acc[i] = (link_acc[i-1][:3] +
                          np.cross(link_acc[i-1][3:], r_i_Ci) +
                          np.cross(link_vel[i-1][3:], np.cross(link_vel[i-1][3:],
                                                               r_i_Ci)))

        return joint_tf, link_tf, link_com, link_vel, link_acc, com_vel, com_acc


if __name__ == '__main__':

    # Create the robot state structure:
    robot_params = RobotParams()
    rob = RobotState(robot_params)

    rob.joint_state[0].th = 0.1
    kinematics = Kinematics(robot_params)
    kinematics.update(rob)
