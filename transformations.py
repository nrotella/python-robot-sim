#!/usr/bin/env/python

import numpy as np

class HomogeneousTransform(object):
    def __init__(self, R=np.identity(3), t=np.zeros(3)):
        self.mat = np.identity(4)
        self.set(R, t)

    def set(self, R, t):
        self.mat[:3, :3] = R
        self.mat[:3, -1] = t

    def R(self):
        return self.mat[:3, :3]

    def Rx(self):
        return self.mat[:3, 0]

    def Ry(self):
        return self.mat[:3, 1]

    def Rz(self):
        return self.mat[:3, 2]

    def t(self):
        return self.mat[:3, -1]

    def tx(self):
        return self.mat[0, -1]

    def ty(self):
        return self.mat[1, -1]

    def tz(self):
        return self.mat[2, -1]

    def dot(self, vec):
        return self.mat.dot(np.append(vec, 1.0))[:3]

    def inv(self):
        return HomogeneousTransform(self.R().transpose(),
                                    -self.R().transpose().dot(self.t()))

def rotX(th):
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(th), -np.sin(th)],
                     [0.0, np.sin(th), np.cos(th)]])


def rotY(th):
    return np.array([[np.cos(th), 0.0, np.sin(th)],
                     [0.0, 1.0, 0.0],
                     [-np.sin(th), 0.0, np.cos(th)]])


def rotZ(th):
    return np.array([[np.cos(th), -np.sin(th), 0.0],
                     [np.sin(th), np.cos(th), 0.0],
                     [0.0, np.sin(th), 1.0]])



def angle_axis_from_vec_dir(vec, dir):
    """ Compute the angle-axis rotation require to orient the vector along dir: """
    # First, make sure to normalize the direction vector:
    dir_norm = np.linalg.norm(dir)
    if dir_norm > 0:
        dir = dir / dir_norm
        axis = np.cross(vec, dir)
        trip_prod = np.linalg.det(np.dstack((vec, dir, axis)))
        if trip_prod > 0:
            angle = np.arccos(np.dot(vec, dir))
        else:
            angle = 2*np.pi - np.arccos(np.dot(vec, dir))

        return angle, axis

    else:
        return 0.0, np.zeros(3)

def exp_map_so3(rot_vec):
    """ Returns the SO(3) rotation matrix formed from the so(3) rotation vector. """
    angle = np.linalg.norm(rot_vec)
    axis = rot_vec / angle

    wx = vec_to_skew_sym(axis)
    return np.identity(3) + np.sin(angle)*wx + (1.0 - np.cos(angle))*(wx.dot(wx))

def log_map_SO3(rot_mat):
    """ Extracts the so(3) rotation vector from the SO(3) rotation matrix. """
    angle = np.arccos(0.5*(np.trace(rot_mat)-1.0))
    axis = np.array([rot_mat[2, 1]-rot_mat[1, 2],
                     rot_mat[0, 2]-rot_mat[2, 0],
                     rot_mat[1, 0]-rot_mat[0, 1]])
    return angle*axis

def rot_mat_to_angle_axis(rot_mat):
    """ Converts an orientation from rotation matrix to angle axis representation. """
    rot_vec = log_map_SO3(rot_mat)
    angle = np.linalg.norm(rot_vec)
    axis = rot_vec / angle
    return angle, axis

def angle_axis_to_rot_mat(angle, axis):
    """ Converts an orientation from angle axis to rotation matrix representation. """
    return exp_map_so3(angle*axis)

def vec_to_skew_sym(vec):
    """ Returns the skew-symmetric cross product matrix formed from the input vector. """
    return np.array([[0.0, -vec[2], vec[1]],
                     [vec[2], 0.0, -vec[0]],
                     [-vec[1], vec[0], 0.0]])
