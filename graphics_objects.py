#!/usr/bin/env/python

from __future__ import print_function

import numpy as np

from OpenGL import GLU
import OpenGL.GL as gl
from OpenGL.arrays import vbo
import transformations as tf
import graphics_defs as gdefs


class GroundGraphics(object):
    """
    Class to create and render a grid (triangular mesh) representing the ground plane.

    """

    def __init__(self, length, width):
        self.len = length
        self.w = width
        self.res = 10
        self.n_sq = self.res**2
        self.n_vert = 6 * self.n_sq

        # Define the vertex (x,y,z) values as a grid:
        self.vx = np.linspace(-0.5*self.len, 0.5*self.len, self.res + 1)
        self.vy = np.linspace(-0.5*self.w, 0.5*self.w, self.res + 1)
        self.vz = np.zeros((self.res + 1, self.res + 1))

        self.vert = np.zeros((self.n_vert, 3))

        # Organize the vertices into triangles for storing in a VBO:
        sq_ind = 0
        for i in range(self.res):
            for j in range(self.res):
                # Upper triangle in square:
                self.vert[6*sq_ind, :] = np.array([self.vx[i], self.vy[j], self.vz[i, j]])
                self.vert[6*sq_ind+1, :] = np.array([self.vx[i+1], self.vy[j+1], self.vz[i+1, j+1]])
                self.vert[6*sq_ind+2, :] = np.array([self.vx[i], self.vy[j+1], self.vz[i, j+1]])

                # Lower triangle in square:
                self.vert[6*sq_ind+3, :] = np.array([self.vx[i], self.vy[j], self.vz[i, j]])
                self.vert[6*sq_ind+4, :] = np.array([self.vx[i+1], self.vy[j], self.vz[i+1, j]])
                self.vert[6*sq_ind+5, :] = np.array([self.vx[i+1], self.vy[j+1], self.vz[i+1, j+1]])

                sq_ind += 1

        # Pack the triangle vertices into a dedicated VBO:
        self.vert_stride = 12  # number of bytes between successive vertices
        self.vert_vbo = vbo.VBO(np.reshape(self.vert, (1, -1), order='C').astype(np.float32))

    def render(self):

        gl.glPushMatrix()
        try:
            # Bind the vertex data buffer to the VBO all future rendering
            # (or until unbound with 'unbind'):
            self.vert_vbo.bind()

            # Set the vertex pointer for rendering:
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glVertexPointer(3, gl.GL_FLOAT, self.vert_stride, self.vert_vbo)

            # Set the polygons to have front and back faces and to not be filled:
            gl.glColor3f(gdefs.GROUND_EDGE_COLOR_R,
                         gdefs.GROUND_EDGE_COLOR_B,
                         gdefs.GROUND_EDGE_COLOR_G)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)

            # Render triangle edges using the loaded vertex pointer data:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.n_vert)

            # Set the polygons to have front and back faces and to not be filled:
            gl.glColor3f(gdefs.GROUND_FACE_COLOR_R,
                         gdefs.GROUND_FACE_COLOR_B,
                         gdefs.GROUND_FACE_COLOR_G)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            # Render triangle faces using the loaded vertex pointer data:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.n_vert)

        except Exception as e:
            print(e)

        finally:
            self.vert_vbo.unbind()
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

        gl.glPopMatrix()


class RobotGraphics(object):
    """
    Class for rendering the robot given a kinematics object and the current state.

    """

    def __init__(self):
        self.quadric = GLU.gluNewQuadric()
        GLU.gluQuadricNormals(self.quadric, GLU.GLU_SMOOTH)  # create Smooth Normals
        GLU.gluQuadricTexture(self.quadric, gl.GL_TRUE)  # create Texture Coords
        GLU.gluQuadricDrawStyle(self.quadric, GLU.GLU_FILL)
        self.vec_graphics = VectorGraphics()
        self.axes_graphics = AxesGraphics()

    def render(self, kinematics, options):
        """ Render the robot graphics given the current kinematic state. """

        # Change to wireframe graphics if desired:
        if options.draw_wireframe:
            GLU.gluQuadricDrawStyle(self.quadric, GLU.GLU_LINE)
        else:
            GLU.gluQuadricDrawStyle(self.quadric, GLU.GLU_FILL)

        if options.use_alpha:
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        else:
            gl.glDisable(gl.GL_BLEND)

        # Draw each link as a cylinder:
        for i in range(1, kinematics.robot.n_links + 1):
            link_len = np.linalg.norm(kinematics.h_tf[i].t() - kinematics.h_tf[i-1].t())
            ang, ax = tf.angle_axis_from_vec_dir(np.array([0.0, 0.0, 1.0]),
                                                 (kinematics.h_tf[i].t() -
                                                  kinematics.h_tf[i-1].t()))

            gl.glPushMatrix()
            gl.glColor4f(gdefs.LINK_COLOR_R,
                         gdefs.LINK_COLOR_G,
                         gdefs.LINK_COLOR_B,
                         gdefs.ROBOT_ALPHA)
            gl.glTranslatef(*kinematics.h_tf[i-1].t())
            gl.glRotate((180.0/np.pi)*ang, *ax)
            GLU.gluCylinder(self.quadric, gdefs.LINK_CYLINDER_RAD, gdefs.LINK_CYLINDER_RAD,
                            link_len, gdefs.CYLINDER_SLICES, gdefs.CYLINDER_STACKS)
            gl.glPopMatrix()

        # Draw each joint frame using an AxesGraphic object if desired (always draw base):
        for i in range(kinematics.robot.n_dofs+1):
            if options.draw_joint_frame[i]:
                self.axes_graphics.render(kinematics.h_tf[i])

        # # Draw each link frame using an AxesGraphic object:
        # for i in range(kinematics.robot.n_links + 1):
        #     self.axes_graphics.render(kinematics.h_tf_links[i])

        # Draw each joint as a cylinder aligned with the joint axis:
        for i in range(1, kinematics.robot.n_dofs + 1):
            ang, ax = tf.angle_axis_from_vec_dir(np.array([0.0, 0.0, 1.0]),
                                                 kinematics.h_tf[i-1].Rz())
            gl.glPushMatrix()
            gl.glColor4f(gdefs.JOINT_COLOR_R,
                         gdefs.JOINT_COLOR_G,
                         gdefs.JOINT_COLOR_B,
                         gdefs.ROBOT_ALPHA)
            gl.glTranslatef(*kinematics.h_tf[i-1].t())
            gl.glRotate((180.0/np.pi)*ang, *ax)
            gl.glTranslatef(0.0, 0.0, -0.5*gdefs.JOINT_CYLINDER_LEN)
            GLU.gluCylinder(self.quadric, gdefs.JOINT_CYLINDER_RAD, gdefs.JOINT_CYLINDER_RAD,
                            gdefs.JOINT_CYLINDER_LEN, gdefs.CYLINDER_SLICES, gdefs.CYLINDER_STACKS)
            #gl.glDisable(gl.GL_BLEND)
            gl.glPopMatrix()

            # Draw a disk for the top of the joint actuator:
            gl.glPushMatrix()
            gl.glColor4f(gdefs.JOINT_COLOR_R,
                         gdefs.JOINT_COLOR_G,
                         gdefs.JOINT_COLOR_B,
                         gdefs.ROBOT_ALPHA)
            gl.glTranslatef(*kinematics.h_tf[i-1].t())
            gl.glRotate((180.0/np.pi)*ang, *ax)
            gl.glTranslatef(0.0, 0.0, 0.5*gdefs.JOINT_CYLINDER_LEN)
            GLU.gluDisk(self.quadric, 0.0, gdefs.JOINT_CYLINDER_RAD,
                        gdefs.DISK_SLICES, gdefs.DISK_STACKS)
            gl.glPopMatrix()

            # Draw a disk for the bottom of the joint actuator:
            gl.glPushMatrix()
            gl.glColor4f(gdefs.JOINT_COLOR_R,
                         gdefs.JOINT_COLOR_G,
                         gdefs.JOINT_COLOR_B,
                         gdefs.ROBOT_ALPHA)
            gl.glTranslatef(*kinematics.h_tf[i-1].t())
            gl.glRotate((180.0/np.pi)*ang, *ax)
            gl.glTranslatef(0.0, 0.0, -0.5*gdefs.JOINT_CYLINDER_LEN)
            GLU.gluDisk(self.quadric, 0.0, gdefs.JOINT_CYLINDER_RAD,
                        gdefs.DISK_SLICES, gdefs.DISK_STACKS)
            gl.glPopMatrix()

        # Draw the endeffector as a sphere:
        gl.glPushMatrix()
        gl.glColor4f(gdefs.ENDEFF_COLOR_R,
                     gdefs.ENDEFF_COLOR_G,
                     gdefs.ENDEFF_COLOR_B,
                     gdefs.ROBOT_ALPHA)
        gl.glTranslatef(*kinematics.h_tf[-1].t())
        GLU.gluSphere(self.quadric, gdefs.ENDEFF_SPHERE_RAD,
                      gdefs.SPHERE_SLICES, gdefs.SPHERE_STACKS)
        gl.glPopMatrix()

        # Draw the link centers of mass as spheres:
        for i in range(kinematics.robot.n_links + 1):
            gl.glPushMatrix()
            gl.glColor4f(gdefs.LINK_COM_COLOR_R,
                         gdefs.LINK_COM_COLOR_G,
                         gdefs.LINK_COM_COLOR_B,
                         gdefs.ROBOT_ALPHA)
            gl.glTranslatef(*(kinematics.link_com[i]))
            GLU.gluSphere(self.quadric, gdefs.LINK_COM_SPHERE_RAD,
                          gdefs.SPHERE_SLICES, gdefs.SPHERE_STACKS)
            gl.glPopMatrix()

        if options.draw_motion_vectors:
            for i in range(kinematics.robot.n_links + 1):
                # Draw the link linear velocity using a VectorGraphics object:
                linvel = kinematics.link_vel[i][:3]
                linvel_norm = np.linalg.norm(linvel)
                if linvel_norm > 0.0:
                    self.vec_graphics.render(kinematics.h_tf[i].t(), linvel/linvel_norm,
                                             gdefs.LINVEL_VEC_SCALE*linvel_norm, gdefs.VEC_RAD,
                                             np.array([gdefs.LINVEL_COLOR_R,
                                                       gdefs.LINVEL_COLOR_G,
                                                       gdefs.LINVEL_COLOR_B,
                                                       gdefs.ROBOT_ALPHA]))

                # Draw the COM velocity using a VectorGraphics object:
                linvel = kinematics.com_vel[i]
                linvel_norm = np.linalg.norm(linvel)
                if linvel_norm > 0.0:
                    self.vec_graphics.render(kinematics.link_com[i], linvel/linvel_norm,
                                             gdefs.LINVEL_VEC_SCALE*linvel_norm, gdefs.VEC_RAD,
                                             np.array([gdefs.LINVEL_COLOR_R,
                                                       gdefs.LINVEL_COLOR_G,
                                                       gdefs.LINVEL_COLOR_B,
                                                       gdefs.ROBOT_ALPHA]))

                # Draw the link angular velocity using a VectorGraphics object:
                angvel = kinematics.link_vel[i][3:]
                angvel_norm = np.linalg.norm(angvel)
                if angvel_norm > 0.0:
                    self.vec_graphics.render(kinematics.h_tf[i].t(), angvel/angvel_norm,
                                             gdefs.ANGVEL_VEC_SCALE*angvel_norm, gdefs.VEC_RAD,
                                             np.array([gdefs.ANGVEL_COLOR_R,
                                                       gdefs.ANGVEL_COLOR_G,
                                                       gdefs.ANGVEL_COLOR_B,
                                                       gdefs.ROBOT_ALPHA]))

            for i in range(kinematics.robot.n_links + 1):
                # Draw the link linear acceleration using a VectorGraphics object:
                linacc = kinematics.link_acc[i][:3]
                linacc_norm = np.linalg.norm(linacc)
                if linacc_norm > 0.0:
                    self.vec_graphics.render(kinematics.h_tf[i].t(), linacc/linacc_norm,
                                             gdefs.LINACC_VEC_SCALE*linacc_norm, gdefs.VEC_RAD,
                                             np.array([gdefs.LINACC_COLOR_R,
                                                       gdefs.LINACC_COLOR_G,
                                                       gdefs.LINACC_COLOR_B,
                                                       gdefs.ROBOT_ALPHA]))

                # Draw the COM accocity using a VectorGraphics object:
                linacc = kinematics.com_acc[i]
                linacc_norm = np.linalg.norm(linacc)
                if linacc_norm > 0.0:
                    self.vec_graphics.render(kinematics.link_com[i], linacc/linacc_norm,
                                             gdefs.LINACC_VEC_SCALE*linacc_norm, gdefs.VEC_RAD,
                                             np.array([gdefs.LINACC_COLOR_R,
                                                       gdefs.LINACC_COLOR_G,
                                                       gdefs.LINACC_COLOR_B,
                                                       gdefs.ROBOT_ALPHA]))

                # Draw the link angular accocity using a VectorGraphics object:
                angacc = kinematics.link_acc[i][3:]
                angacc_norm = np.linalg.norm(angacc)
                if angacc_norm > 0.0:
                    self.vec_graphics.render(kinematics.h_tf[i].t(), angacc/angacc_norm,
                                             gdefs.ANGACC_VEC_SCALE*angacc_norm, gdefs.VEC_RAD,
                                             np.array([gdefs.ANGACC_COLOR_R,
                                                       gdefs.ANGACC_COLOR_G,
                                                       gdefs.ANGACC_COLOR_B,
                                                       gdefs.ROBOT_ALPHA]))

class VectorGraphics(object):
    """
    Class for rendering a three-dimensional vector.

    """

    def __init__(self):
        self.quadric = GLU.gluNewQuadric()
        GLU.gluQuadricNormals(self.quadric, GLU.GLU_FLAT)  # create Smooth Normals
        GLU.gluQuadricTexture(self.quadric, gl.GL_TRUE)  # create Texture Coords

    def render(self, start, dir, length, width, color):

        if length > 0.0:
            up_vec = np.array([0.0, 0.0, 1.0])
            angle, axis = tf.angle_axis_from_vec_dir(up_vec, dir)

            # Draw the shaft using a cylinder:
            gl.glPushMatrix()
            gl.glColor4f(*color)
            gl.glTranslatef(*start)
            gl.glRotate((180.0/np.pi)*angle, *axis)
            GLU.gluCylinder(self.quadric, width, width, length, 100, 10)
            gl.glPopMatrix()

            # Draw the head using a cylinder having zero width on top:
            gl.glPushMatrix()
            gl.glColor4f(*color)
            gl.glTranslatef(*start)
            gl.glRotate((180.0/np.pi)*angle, *axis)
            gl.glTranslatef(0.0, 0.0, length)
            GLU.gluCylinder(self.quadric, 2.0*width, 0.0, gdefs.VEC_HEAD_RATIO*length, 100, 10)
            gl.glPopMatrix()


class AxesGraphics(object):
    """
    Class for rendering an axes (frame vectors) object.

    """

    def __init__(self):
        self.x_axis = VectorGraphics()
        self.y_axis = VectorGraphics()
        self.z_axis = VectorGraphics()

    def render(self, frame_tf):
        self.x_axis.render(frame_tf.t(), frame_tf.R()[:, 0],
                           gdefs.AXES_LEN, gdefs.VEC_RAD, np.array([1.0, 0.0, 0.0, gdefs.ROBOT_ALPHA]))
        self.y_axis.render(frame_tf.t(), frame_tf.R()[:, 1],
                           gdefs.AXES_LEN, gdefs.VEC_RAD, np.array([0.0, 1.0, 0.0, gdefs.ROBOT_ALPHA]))
        self.z_axis.render(frame_tf.t(), frame_tf.R()[:, 2],
                           gdefs.AXES_LEN, gdefs.VEC_RAD, np.array([0.0, 0.0, 1.0, gdefs.ROBOT_ALPHA]))


class GraphicsOptions(object):
    """ Class for setting graphics options, normally via user input. """

    def __init__(self, n_dofs):
        self.n_dofs = n_dofs

        self.draw_wireframe = False

        self.draw_motion_vectors = False

        self.use_alpha = False

        self.draw_joint_frame = np.array((self.n_dofs+1)*[False])
        self.draw_joint_frame[0] = True  # always draw the base frame

    def set_draw_wireframe(self, draw_bool):
        self.draw_wireframe = draw_bool

    def toggle_draw_wireframe(self):
        self.draw_wireframe = not self.draw_wireframe

    def set_draw_motion_vectors(self, draw_bool):
        self.draw_motion_vectors = draw_bool

    def toggle_draw_motion_vectors(self):
        self.draw_motion_vectors = not self.draw_motion_vectors

    def set_use_alpha(self, alpha_bool):
        self.use_alpha = alpha_bool

    def toggle_use_alpha(self):
        self.use_alpha = not self.use_alpha

    def set_draw_joint_frame(self, joint_id, draw_bool):
        if joint_id >= 0 and joint_id <= self.n_dofs+1:
            self.draw_joint_frame[joint_id] = draw_bool

    def toggle_draw_joint_frame(self, joint_id):
        if joint_id >= 0 and joint_id < self.n_dofs+1:
            self.draw_joint_frame[joint_id] = not self.draw_joint_frame[joint_id]

    def toggle_draw_all_joint_frames(self):
        # If all the frames are currently drawn, turn them all off:
        if np.all(self.draw_joint_frame):
            for i in range(self.n_dofs+1):
                self.draw_joint_frame[i] = False
            self.draw_joint_frame[0] = True  # always draw the base frame
        # Otherwise, turn them all on:
        else:
            for i in range(self.n_dofs+1):
                self.draw_joint_frame[i] = True
