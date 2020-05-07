#!/usr/bin/env/python

import pytest

from geometry import HomogeneousTransform
from scipy.stats import special_ortho_group
import numpy as np

def test_homogeneous_transform():
    """
    Test inversion and multiplication of HomogeneousTransform objects.
    
    Test that a random HomogeneousTransform object T1 returns the 4x4 identity matrix when
    multiplied by its inverse transform, T1.inv().

    """
    R = special_ortho_group.rvs(3)
    t = np.random.rand(3,1)
    T1 = HomogeneousTransform(R, t)

    result = T1.inv() * T1

    np.testing.assert_array_almost_equal(result.mat, np.identity(4))
