#!/usr/bin/env/python

# Enumeration of joint types:
JTYPE_REVOLUTE = 0
JTYPE_PRISMATIC = 1

# Enumeration of integration methods:
INTEGRATOR_TYPE_NONE = 0
INTEGRATOR_TYPE_EULER = 1
INTEGRATOR_TYPE_RK4 = 2

# Enumeration of controller_types:
CONTROLLER_TYPE_NONE = 0
CONTROLLER_TYPE_PD = 1
CONTROLLER_TYPE_GRAV_COMP_PD = 2
CONTROLLER_TYPE_INVDYN_PD = 3

# Joint viscous damping:
VISCOUS_DAMP_COEFF = 0.1

# Joint limit mechanical stiffness:
JOINT_LIM_STIFF = 10000.0

# Floor mechanical stiffness:
FLOOR_STIFF = 100000.0
FLOOR_DAMP = 1000.0
