#!/usr/bin/env/python

import numpy as np
from scipy.stats import special_ortho_group

class HomogeneousTransform(object):
    """ 
    Class implementing a three-dimensional homogeneous transformation.

    This class implements a homogeneous transformation, which is the combination of a rotation R 
    and a translation t stored as a 4x4 matrix of the form: 
    
    T = [R11 R12 R13 t1x
         R21 R22 R23 t2
         R31 R32 R33 t3
         0   0   0    1]

    Transforms can operate directly on homogeneous vectors of the form [x y z 1] using matrix 
    math. Defaults to the identity transformation if no rotation/translation are specified. The
    transformation can be accessed as either a rotation/translation or matrix via attributes which
    are kept in sync as the transformation is updated

    Attributes:
        matrix (4x4 numpy array): The homogeneous transformation as a matrix.

    """

    def __init__(self, rotation=None, translation=None, matrix=None):
        """ 
	Initialize a homogeneous transformation.

        "Overloaded" constructor which initializes a transformation object from either a 4x4 
        matrix OR the combination of a rotation matrix and translation vector. If all three 
        inputs are None, the default is the identity transformation.

   	Args:			
            rotation (3x3 numpy array): Rotation matrix.
            translation (3x1 numpy array): Translation vector.
            matrix (4x4 numpy array): Homogeneous transformation matrix.
 
	"""
        if matrix is not None:
            self._mat = matrix

        else:
            if rotation is None:
                rotation = np.identity(3)

            if translation is None:
                translation = np.zeros((3,1))
            
            self.set(rotation, translation)

            
    def __mul__(self, other):
        """
        Multiplies HomogeneousTransform objects using their underlying matrices.

        Args:
            other (HomogeneousTransform): Right-hand-side transform to multiply.

        Returns:
            (HomogeneousTransform): Resultant composed homogeneous transformation.
    
        """
        return HomogeneousTransform(matrix=self._mat.dot(other.mat))

        
    @property
    def mat(self):
        return self._mat


    @mat.setter
    def mat(self, value):
        self._mat = value

    
    def set(self, rotation, translation):
        """ Set the transformation's rotation and translation.

        Args:
            rotation (3x3 numpy array): Rotation matrix.
            translation (3x1 numpy array): Translation vector.

        Returns:
            (None)
        
        """
        self._mat = np.block([
            [rotation, translation.reshape(3,1)],
            [np.zeros((1,3)), 1.0]
        ])
        
    def inv(self):
        """ Returns the inverse of the homogeneous transformation.

        Args:
            (None)

        Returns:
            (HomogeneousTransform): Inverse homogeneous transformation.

        """
        R = self._mat[:3,:3].T
        t = -(self._mat[:3,:3].T).dot(self._mat[:3,3])
        return HomogeneousTransform(rotation=R, translation=t)

    
    def R(self):
        """ Returns the rotation portion of the transformation.

        Args:
            (None)

        Returns:
            (3x3 numpy array): Rotation matrix.
                
        """
        return self._mat[:3,:3]

    
    def Rx(self):
        """ Returns the x-axis of the rotation portion of the transformation.

        Args:
            (None)

        Returns:
            (3x1 numpy array): Rotation matrix x-axis vector.
                
        """
        return self._mat[:3,0]

    
    def Ry(self):
        """ Returns the y-axis of the rotation portion of the transformation.

        Args:
            (None)

        Returns:
            (3x1 numpy array): Rotation matrix y-axis vector.
                
        """
        return self._mat[:3,1]

    
    def Rz(self):
        """ Returns the z-axis of the rotation portion of the transformation.

        Args:
            (None)

        Returns:
            (3x1 numpy array): Rotation matrix z-axis vector.
                
        """
        return self._mat[:3,2]

    
    def t(self):
        """ Returns the translation portion of the transformation.

        Args:
            (None)

        Returns:
            (3x1 numpy array): Translation vector.
                
        """
        return self._mat[:3,-1]

    
    def tx(self):
        """ Returns the x component of the translation portion of the transformation.

        Args:
            (None)

        Returns:
            (float): Translation vector.
                
        """
        return self._mat[0,-1]

    
    def ty(self):
        """ Returns the y component of the translation portion of the transformation.

        Args:
            (None)

        Returns:
            (float): Translation vector.
                
        """
        return self._mat[1,-1]

    
    def tz(self):
        """ Returns the z component of the translation portion of the transformation.

        Args:
            (None)

        Returns:
            (float): Translation vector.
                
        """
        return self._mat[2,-1]

    
if __name__ == '__main__':
    print('main')
