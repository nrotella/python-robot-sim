#!/usr/bin/env/python

from __future__ import print_function

import copy
import sys
import time
import numpy as np

from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4 import QtOpenGL
from OpenGL import GLU
from OpenGL.GL import *
from OpenGL.arrays import vbo

import kinematics as kin
import dynamics as dyn

from graphics_objects import GroundGraphics
from graphics_objects import RobotGraphics
from graphics_objects import GraphicsOptions

import robot_defs
import transformations as tf

import inverse_kinematics as ik

class GLWidget(QtOpenGL.QGLWidget):
    """
    This Qt OpenGL widget is the main object for performing all openGL graphics
    programming.  The functions initializeGL, resizeGL, paintGL must be defined.

    """

    def __init__(self, parent=None):
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)

        # Timing variables:
        self.sim_freq = 100.0
        self.render_freq = 30.0

        self.spin_time = time.time()
        self.spin_time_prev = self.spin_time

        self.render_time = time.time()
        self.render_time_prev = self.spin_time

        self.dt = 1.0 / self.sim_freq  # ideal, to be computed in loop

        # Kinematics and dynamics objects:
        self.robot = kin.RobotParams()
        self.robot_state = kin.RobotState(self.robot)

        self.sim_time = 0.0
        self.kin = kin.Kinematics(self.robot)
        self.kin_des = kin.Kinematics(self.robot)
        self.dyn = dyn.RecursiveNewtonEuler(self.robot, self.robot_state, self.kin)

        self.integrator_type = robot_defs.INTEGRATOR_TYPE_NONE

        self.controller_type = robot_defs.CONTROLLER_TYPE_PD

        self.kin.initialize(tf.HomogeneousTransform(),
                            np.zeros(6),
                            np.zeros(6),
                            self.robot_state.joint_state,
                            self.dt)
        self.ik = ik.InverseKinematics(self.robot, self.robot_state.joint_state)
        self.initial_endeff_pos = self.kin.h_tf_links[-1].t()
        self.initial_endeff_ori = self.kin.h_tf_links[-1].R()

        # Camera spherical coordinates:
        self.eye_r = 3.0
        self.eye_th = 0.3*np.pi
        self.eye_phi = 0.5

        self.eye_pos = np.array([self.eye_r*np.sin(self.eye_th)*np.cos(self.eye_phi),
                                 self.eye_r*np.sin(self.eye_th)*np.sin(self.eye_phi),
                                 self.eye_r*np.cos(self.eye_th)])
        self.center_pos = np.array([0.0, 0.0, 0.0])

        self.ground_graphics = GroundGraphics(length=10.0, width=10.0)
        self.robot_graphics = RobotGraphics()

        # General class for encapsulating graphics options, eg set draw
        # in wireframe, draw joint axes etc:
        self.graphics_options = GraphicsOptions(self.robot.n_dofs)

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            if event.key() == QtCore.Qt.Key_W:
                self.eye_r -= 0.1
                self.update_view()

            elif event.key() == QtCore.Qt.Key_S:
                self.eye_r += 0.1
                self.update_view()

            elif event.key() == QtCore.Qt.Key_Down:
                self.eye_th += 0.01
                self.update_view()

            elif event.key() == QtCore.Qt.Key_Up:
                self.eye_th -= 0.01
                self.update_view()

            elif event.key() == QtCore.Qt.Key_Left:
                self.eye_phi -= 0.05
                self.update_view()

            elif event.key() == QtCore.Qt.Key_Right:
                self.eye_phi += 0.05
                self.update_view()

            elif event.key() == QtCore.Qt.Key_T:
                self.graphics_options.toggle_draw_wireframe()

            elif event.key() == QtCore.Qt.Key_V:
                self.graphics_options.toggle_draw_motion_vectors()

            elif event.key() == QtCore.Qt.Key_0:
                self.graphics_options.toggle_draw_all_joint_frames()

            elif event.key() == QtCore.Qt.Key_1:
                self.graphics_options.toggle_draw_joint_frame(1)

            elif event.key() == QtCore.Qt.Key_2:
                self.graphics_options.toggle_draw_joint_frame(2)

            elif event.key() == QtCore.Qt.Key_3:
                self.graphics_options.toggle_draw_joint_frame(3)

            elif event.key() == QtCore.Qt.Key_4:
                self.graphics_options.toggle_draw_joint_frame(4)

            elif event.key() == QtCore.Qt.Key_5:
                self.graphics_options.toggle_draw_joint_frame(5)

            elif event.key() == QtCore.Qt.Key_6:
                self.graphics_options.toggle_draw_joint_frame(6)

            elif event.key() == QtCore.Qt.Key_7:
                self.graphics_options.toggle_draw_joint_frame(7)

            elif event.key() == QtCore.Qt.Key_8:
                self.graphics_options.toggle_draw_joint_frame(8)

            elif event.key() == QtCore.Qt.Key_9:
                self.graphics_options.toggle_draw_joint_frame(9)

            elif event.key() == QtCore.Qt.Key_K:
                self.controller_type = robot_defs.CONTROLLER_TYPE_PD

    def update_view(self):
        self.eye_pos = np.array([self.eye_r*np.sin(self.eye_th)*np.cos(self.eye_phi),
                                 self.eye_r*np.sin(self.eye_th)*np.sin(self.eye_phi),
                                 self.eye_r*np.cos(self.eye_th)])
        look_vec = (self.center_pos - self.eye_pos) / np.linalg.norm(self.center_pos - self.eye_pos)
        up_vec = np.array([0.0, 0.0, 1.0])
        right_vec = np.cross(look_vec, up_vec)
        glLoadIdentity()
        GLU.gluLookAt(*np.concatenate((self.eye_pos, self.center_pos, up_vec)))

    def initializeGL(self):

        # Convenience function, calls glClearColor under the hood.
        # QColor is specified as RGB ints (0-255).  Specify this clear
        # color once and call glClear(GL_COLOR_BUFFER_BIT) before each
        # round of rendering (in paintGL):
        self.qglClearColor(QtGui.QColor(100, 100, 100)) # a grey background

        # Initialize the cube vertices:
        self.initGeometry()

        # Enable the depth buffer:
        glEnable(GL_DEPTH_TEST)

    def resizeGL(self, width, height):

        # Prevent the window height from being set to zero:
        if height == 0: height = 1

        # Set the affine transform converting 'display' to 'screen' coords:.  By using
        # the same width and height passed to the resizeGL function, we resize objects
        # to the new window size.
        glViewport(0, 0, width, height)

        # Set the target matrix stack to the projection matrix stack:
        glMatrixMode(GL_PROJECTION)

        # Replace the current matrix on the stack with the identity (homogeneous tf):
        glLoadIdentity()

        # Set up a perspective projection matrix:
        fov = 45.0 # field of view angle in y-direction (degrees)
        aspect = width / float(height) # aspect ratio, determines field of view in x-direction
        zNear = 0.1 # distance from viewer to near clipping plane (+ve)
        zFar = 100.0 # distance from viewer to far clipping plane (+ve)
        GLU.gluPerspective(fov, aspect, zNear, zFar)

        # Set the target matrix stack to the modelview matrix stack:
        glMatrixMode(GL_MODELVIEW)

        # Create the initial view using the initial eye position:
        self.update_view()

    def paintGL(self):

        # Clear depth and color buffers in preparations for new rendering:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        self.ground_graphics.render()
        self.robot_graphics.render(self.kin, self.graphics_options)

        # Draw the desired state using transparency:
        self.graphics_options.set_use_alpha(True)
        self.robot_graphics.render(self.kin_des, self.graphics_options)
        self.graphics_options.set_use_alpha(False)

        # glPushMatrix()
        # self.renderText(0.0, 0.0, 0.0, 'text!')
        # glPopMatrix()

    def initGeometry(self):
        pass

    def spin(self):
        self.spin_time_prev = self.spin_time
        self.spin_time = time.time()
        self.dt = self.spin_time - self.spin_time_prev
        self.sim_time += self.dt
        self.fps = 1.0 / self.dt

        # Compute joint trajectories:
        # amp = 0.5
        # freq = 0.1
        # for i in range(self.robot.n_dofs):
        #     self.robot_state.joint_state_des[i].th = amp*np.sin(2.0*np.pi*freq*self.sim_time)
        #     self.robot_state.joint_state_des[i].thd = 2.0*np.pi*freq*amp*np.cos(2.0*np.pi*freq*self.sim_time)
        #     self.robot_state.joint_state_des[i].thdd = -2.0*np.pi*freq*2.0*np.pi*freq*amp*np.sin(2.0*np.pi*freq*self.sim_time)

        # Update kinematics from the current joint state:
        self.kin.update(tf.HomogeneousTransform(),
                        np.zeros(6),
                        np.zeros(6),
                        self.robot_state.joint_state,
                        self.dt)

        # Update desired kinematics:
        self.kin_des.update(tf.HomogeneousTransform(),
                            np.zeros(6),
                            np.zeros(6),
                            self.robot_state.joint_state_des,
                            self.dt)

        # Compute joint state from differential IK:
        amp = 0.1
        freq = 0.2
        self.robot_state.joint_state_des = self.ik.update(self.kin_des,
                                                          self.initial_endeff_pos +
                                                          np.array([amp*np.cos(2.0*np.pi*freq*self.sim_time), 0.0, amp*np.sin(2.0*np.pi*freq*self.sim_time)]),
                                                          self.initial_endeff_ori,
                                                          np.array([-2.0*np.pi*freq*amp*np.sin(2.0*np.pi*freq*self.sim_time), 0.0, 2.0*np.pi*freq*amp*np.cos(2.0*np.pi*freq*self.sim_time)]),
                                                          np.zeros(3),
                                                          self.dt)

        # Update dynamics quantities before calling class methods:
        self.dyn.update(self.robot_state.joint_state)
        
        if self.controller_type == robot_defs.CONTROLLER_TYPE_PD:
            # PD controller:
            for i in range(self.robot.n_dofs):
                self.robot_state.joint_state_des[i].u = 0.0*(
                    self.robot_state.joint_state_des[i].th - self.robot_state.joint_state[i].th) + \
                    0.0*(self.robot_state.joint_state_des[i].thd - self.robot_state.joint_state[i].thd)

        elif self.controller_type == robot_defs.CONTROLLER_TYPE_GRAV_COMP_PD:
            # Gravity Compensation + PD Controller:
            tau_grav_comp = self.dyn.compute_grav_comp(self.robot_state.joint_state_des)
            for i in range(self.robot.n_dofs):
                self.robot_state.joint_state_des[i].u = tau_grav_comp[i] + 5000.0*(
                    self.robot_state.joint_state_des[i].th - self.robot_state.joint_state[i].th) + \
                    200.0*(self.robot_state.joint_state_des[i].thd - self.robot_state.joint_state[i].thd)

                
        # Inverse dynamics (using qddot) + PD Controller:
        elif self.controller_type == robot_defs.CONTROLLER_TYPE_INVDYN_PD:
            tau_invdyn = self.dyn.compute_joint_torques(np.zeros(6),
                                                        np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0]),
                                                        self.robot_state.joint_state_des)
            for i in range(self.robot.n_dofs):
                self.robot_state.joint_state_des[i].u = tau_invdyn [i] + 5000.0*(
                    self.robot_state.joint_state_des[i].th - self.robot_state.joint_state[i].th) + \
                    200.0*(self.robot_state.joint_state_des[i].thd - self.robot_state.joint_state[i].thd)

        elif self.controller_type == robot_defs.CONTROLLER_TYPE_NONE:
            for i in range(self.robot.n_dofs):
                self.robot_state.joint_state_des[i].u = 0.0

        else:
            print('WARNING >> Invalid controller type.')
            for i in range(self.robot.n_dofs):
                self.robot_state.joint_state_des[i].u = 0.0

        # Check for contacts with the ground and apply reaction forces if necessary:
        for i, joint in enumerate(self.robot_state.joint_state):
            if self.kin.h_tf[i].t()[2] < 0.0:
                joint.fext = np.array([0.0,
                                       0.0,
                                       (robot_defs.FLOOR_STIFF*self.kin.h_tf[i].t()[2] +
                                        -robot_defs.FLOOR_DAMP*self.kin.link_vel[i][2])])
            else:
                joint.fext = np.zeros(3)

        self.robot_state.joint_state = self.integrate_dynamics(self.robot_state.joint_state,
                                                               self.robot_state.joint_state_des, self.dt)

        self.parent.statusBar().showMessage('sim_freq: '+str(self.fps))

        self.render_time = time.time()
        if((self.render_time-self.render_time_prev) >= (1.0/self.render_freq)):
            self.updateGL()
            self.render_time_prev = self.render_time

    def integrate_dynamics(self, joint_state, joint_state_des, dt):
        """ Integrates the dynamics forward in joint space by dt. """

        # Create a deep copy of the joint state to modify for integration:
        jstate = copy.deepcopy(joint_state)

        # Update the joint state by computing the joint-space dynamics and integrating:
        if self.integrator_type == robot_defs.INTEGRATOR_TYPE_EULER:
            qddot = self.dyn.compute_qddot(joint_state_des)
            for i in range(self.robot.n_dofs):
                jstate[i].thdd = qddot[i]
                jstate[i].thd = joint_state[i].thd + self.dt*jstate[i].thdd
                jstate[i].th = jstate[i].th + self.dt*jstate[i].thd

        elif self.integrator_type == robot_defs.INTEGRATOR_TYPE_RK4:
            k1 = np.zeros(2*self.robot.n_dofs)
            k2 = np.zeros(2*self.robot.n_dofs)
            k3 = np.zeros(2*self.robot.n_dofs)
            k4 = np.zeros(2*self.robot.n_dofs)

            # k1 = h * f(tn, yn):
            k1[:self.robot.n_dofs] = dt * np.array([j.thd for j in jstate])
            k1[self.robot.n_dofs:] = dt * self.dyn.compute_qddot(joint_state_des)

            # k2 = h * f(tn + h/2, yn + k1/2):
            for i in range(self.robot.n_dofs):
                jstate[i].th = joint_state[i].th + 0.5 * k1[i]
                jstate[i].thd = joint_state[i].thd + 0.5 * k1[i+self.robot.n_dofs]
            k2[:self.robot.n_dofs] = dt * np.array([j.thd for j in jstate])
            k2[self.robot.n_dofs:] = dt * self.dyn.compute_qddot(joint_state_des)

            # k3 = h * f(tn + h/2, yn + k2/2):
            for i in range(self.robot.n_dofs):
                jstate[i].th = joint_state[i].th + 0.5 * k2[i]
                jstate[i].thd = joint_state[i].thd + 0.5 * k2[i+self.robot.n_dofs]
            k3[:self.robot.n_dofs] = dt * np.array([j.thd for j in jstate])
            k3[self.robot.n_dofs:] = dt * self.dyn.compute_qddot(joint_state_des)

            # k4 = h * f(tn + h/2, yn + k3):
            for i in range(self.robot.n_dofs):
                jstate[i].th = joint_state[i].th + k3[i]
                jstate[i].thd = joint_state[i].thd + k3[i+self.robot.n_dofs]
            k4[:self.robot.n_dofs] = dt * np.array([j.thd for j in jstate])
            k4[self.robot.n_dofs:] = dt * self.dyn.compute_qddot(joint_state_des)

            # Finally, compute the joint state from RK4 intermediate variables:
            # y_{n+1} = y_{n} + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
            for i in range(self.robot.n_dofs):
                jstate[i].th = joint_state[i].th + \
                               (1.0 / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
                jstate[i].thd = joint_state[i].thd  + \
                               (1.0 / 6.0) * (k1[i + self.robot.n_dofs] + 2*k2[i + self.robot.n_dofs] + \
                                          2*k3[i + self.robot.n_dofs] + k4[i + self.robot.n_dofs])
                jstate[i].thdd = (1.0 / dt) * k1[i + self.robot.n_dofs]

        elif self.integrator_type == robot_defs.INTEGRATOR_TYPE_NONE:
            pass  # returns the joint_state copy unmodified

        return jstate

class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.resize(300, 300)
        self.setWindowTitle('Robot Simulator')

        self.initActions()
        self.initMenus()

        self.glWidget = GLWidget(self)

        self.setCentralWidget(self.glWidget)
        self.glWidget.setFocus()

        # Create the dockable sliders widget:
        self.slider_dock = QtGui.QDockWidget('Joint Angles', self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.slider_dock)

        self.slider_multi_widget = QtGui.QWidget()
        self.slider_layout = QtGui.QVBoxLayout()
        self.slider_multi_widget.setLayout(self.slider_layout)

        # Create the sliders and set their range:
        self.joint_sliders = [QtGui.QSlider(QtCore.Qt.Horizontal)
                              for i in range(self.glWidget.robot_state.robot.n_dofs)]
        [self.joint_sliders[i].setRange(-100, 100) for i in range(self.glWidget.robot_state.robot.n_dofs)]

        # Set initial values based on current state and connect the sliders:
        [self.joint_sliders[i].setValue((200/(2*np.pi))*self.glWidget.robot_state.joint_state[i].th) for
         i in range(self.glWidget.robot_state.robot.n_dofs)]
        [self.joint_sliders[i].valueChanged.connect(
            lambda val, i=i: self.glWidget.robot_state.joint_state[i].set_th((2*np.pi/200) * val)) for
         i in range(self.glWidget.robot_state.robot.n_dofs)]

        # # Set initial values based on current state and connect the sliders:
        # [self.joint_sliders[i].setValue((200/(2*np.pi))*self.glWidget.robot_state.joint_state_des[i].th) for
        #  i in range(self.glWidget.robot_state.robot.n_dofs)]
        # [self.joint_sliders[i].valueChanged.connect(
        #     lambda val, i=i: self.glWidget.robot_state.joint_state_des[i].set_th((2*np.pi/200) * val)) for
        #  i in range(self.glWidget.robot_state.robot.n_dofs)]

        [self.slider_layout.addWidget(slider) for slider in self.joint_sliders]
        self.slider_dock.setWidget(self.slider_multi_widget)

        timer = QtCore.QTimer(self)
        timer.setInterval(1000.0*(1.0/self.glWidget.sim_freq))
        QtCore.QObject.connect(timer, QtCore.SIGNAL('timeout()'), self.glWidget.spin)
        timer.start()

    def initActions(self):
        self.exitAction = QtGui.QAction('Quit', self)
        self.exitAction.setShortcut('Ctrl+Q')
        self.exitAction.setStatusTip('Exit application')
        self.connect(self.exitAction, QtCore.SIGNAL('triggered()'), self.close)

    def initMenus(self):
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(self.exitAction)

    def close(self):
        QtGui.qApp.quit()


if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
