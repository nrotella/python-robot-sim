from PyQt4 import QtCore      # core Qt functionality
from PyQt4 import QtGui       # extends QtCore with GUI functionality
from PyQt4 import QtOpenGL    # provides QGLWidget, a special OpenGL QWidget

import OpenGL.GL as gl        # python wrapping of OpenGL
from OpenGL import GLU        # OpenGL Utility Library, extends OpenGL functionality

import sys                    # we'll need this later to run our Qt application

from OpenGL.arrays import vbo    # used to store VBO data
import numpy as np               # general matrix/array math


class GLWidget(QtOpenGL.QGLWidget):
    """
    This class defines the Qt OpenGL widget, which is the main object for performing all openGL
    graphics programming.  The functions initializeGL, resizeGL, paintGL must be defined.

    """

    def __init__(self, parent=None):
        """ Initialize the Qt OpenGL Widget.

        Initialize the Qt OpenGL widget.

        Args:
            (None)

        Returns:
           (None)

        """

        # Store reference to and initialize the parent class
        self.parent = parent
        QtOpenGL.QGLWidget.__init__(self, parent)

    def initializeGL(self):
        """ Initializes OpenGL functionality and geometry.

        Virtual function provided by QGLWidget, called once at the beginning of application.
        OpenGL and geometry initialization is performed here.

        Args:
            (None)

        Returns:
           (None)

        """

        self.qglClearColor(QtGui.QColor(0, 0, 255))    # initialize the screen to blue
        gl.glEnable(gl.GL_DEPTH_TEST)                  # enable depth testing

        # Initialize the user-specified geometry
        self.initGeometry()

        # Initialize the cube rotation parameters to zero
        self.rotX = 0.0
        self.rotY = 0.0
        self.rotZ = 0.0

    def resizeGL(self, width, height):
        """ Defines behavior of OpenGL window when resized.

        Virtual function provided by QGLWidget, called once at the beginning of application
        to set up the OpenGL viewing volume and then called each time the window is resized
        by the user.

        Args:
            width (int): Screen width in pixels.
            height (int): Screen height in pixels.

        Returns:
            (None)

        """

        # Create the viewport, using the full window size
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = width / float(height)

        # Define the viewing volume (frustrum)
        GLU.gluPerspective(45.0, aspect, 1.0, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        """ Defines behavior of OpenGL window when resized.

        Virtual function provided by QGLWidget, called from QGLWidget method updateGL.
        All user rendering code should be defined here.

        Args:
            (None)

        Returns:
            (None)

        """

        # Start from a blank slate each render by clearing buffers
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Create the transform to be applied to the rendered cube
        gl.glPushMatrix()    # push the current matrix to the current stack

        gl.glTranslate(0.0, 0.0, -50.0)    # translate cube to specified depth
        gl.glScale(20.0, 20.0, 20.0)       # scale cube
        gl.glRotate(self.rotX, 1.0, 0.0, 0.0)    # rotate about X axis
        gl.glRotate(self.rotY, 0.0, 1.0, 0.0)    # rotate about Y axis
        gl.glRotate(self.rotZ, 0.0, 0.0, 1.0)    # rotate about Z axis
        gl.glTranslate(-0.5, -0.5, -0.5)         # translate cube center to origin

        # Peform the actual rendering of the cube
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        gl.glVertexPointer(3, gl.GL_FLOAT, 0, self.vertVBO)
        gl.glColorPointer(3, gl.GL_FLOAT, 0, self.colorVBO)

        gl.glDrawElements(gl.GL_QUADS, len(self.cubeIdxArray), gl.GL_UNSIGNED_INT, self.cubeIdxArray)

        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisableClientState(gl.GL_COLOR_ARRAY)

        gl.glPopMatrix()    # restore the previous modelview matrix

    def initGeometry(self):
        """ Initializes cube geometry (vertices, colors, face indices) using VBOs and arrays. """

        # Create cube vertex array and store in a VBO
        self.cubeVtxArray = np.array(
                [[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0]])
        self.vertVBO = vbo.VBO(np.reshape(self.cubeVtxArray,
                                          (1, -1)).astype(np.float32))
        self.vertVBO.bind()

        # Create cube color array and store in a VBO
        self.cubeClrArray = np.array(
                [[0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [1.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [1.0, 0.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [0.0, 1.0, 1.0 ]])
        self.colorVBO = vbo.VBO(np.reshape(self.cubeClrArray,
                                           (1, -1)).astype(np.float32))
        self.colorVBO.bind()

        # Create the cube face index array, specifying faces in terms of order sets of indices
        self.cubeIdxArray = np.array(
                [0, 1, 2, 3,
                 3, 2, 6, 7,
                 1, 0, 4, 5,
                 2, 1, 5, 6,
                 0, 3, 7, 4,
                 7, 6, 5, 4 ])

    def setRotX(self, val):
        """ Callback for X rotation slider. """
        self.rotX = val

    def setRotY(self, val):
        """ Callback for Y rotation slider. """
        self.rotY = val

    def setRotZ(self, val):
        """ Callback for Z rotation slider. """
        self.rotZ = val


class MainWindow(QtGui.QMainWindow):
    """
    This class defines the Qt main window for the application, to which we add Qt widgets for
    OpenGL graphics, user input, etc.

    """

    def __init__(self):
        """ Initialize the Qt MainWindow.

        Initializes the Qt main window by setting up the window itself, creating the GLWidget,
        adding GUI elements and creating a timed rendering loop.

        Args:
            (None)

        Returns:
           (None)

        """

        QtGui.QMainWindow.__init__(self)    # call the init for the parent class

        # Set up the main window
        self.resize(300, 300)
        self.setWindowTitle('Hello OpenGL App')

        # Create the Qt OpenGL widget and initialize GUI elements for MainWindow
        self.glWidget = GLWidget(self)
        self.initGUI()

        # Create a timer and connect its signal to the QGLWidget update function
        timer = QtCore.QTimer(self)
        timer.setInterval(20)   # period, in milliseconds
        timer.timeout.connect(self.glWidget.updateGL)
        timer.start()

    def initGUI(self):
        """ Initialize the Qt GUI elements for the main window.

        Initializes the Qt main window GUI elements.  Sets up a central widget with a vertical
        layout and adds the GLWidget followed by user input elements (sliders).

        Args:
            (None)

        Returns:
           (None)

        """

        # Create the central widget for the window and set its layout
        central_widget = QtGui.QWidget()
        gui_layout = QtGui.QVBoxLayout()
        central_widget.setLayout(gui_layout)

        self.setCentralWidget(central_widget)

        # Add the GLWidget to the layout
        gui_layout.addWidget(self.glWidget)

        # Create the sliders for cube rotation and connect their simple callback functions
        sliderX = QtGui.QSlider(QtCore.Qt.Horizontal)
        sliderX.valueChanged.connect(lambda val: self.glWidget.setRotX(val))

        sliderY = QtGui.QSlider(QtCore.Qt.Horizontal)
        sliderY.valueChanged.connect(lambda val: self.glWidget.setRotY(val))

        sliderZ = QtGui.QSlider(QtCore.Qt.Horizontal)
        sliderZ.valueChanged.connect(lambda val: self.glWidget.setRotZ(val))

        # Add the sliders to the layout below the OpenGL widget
        gui_layout.addWidget(sliderX)
        gui_layout.addWidget(sliderY)
        gui_layout.addWidget(sliderZ)


if __name__ == '__main__':

    # Run the Qt application
    app = QtGui.QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec_())
