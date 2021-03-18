# This script is not meant to be executed alone. It will be called by other
# optimizer modules to evaluate the objective function for 
# different parameter values

import numpy as np
import cmath as cm

# Function to evaluate the objective function and return the value for given values of parameters and design variables
def objective_function_1(parameters, design_variables):
  # design variables = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
  # parameters = [omega, m3]
  # Assigning the design variables
  k1 = design_variables[0]
  k2 = design_variables[1]
  k4 = design_variables[2]
  k5 = design_variables[3]
  c1 = design_variables[4]
  c2 = design_variables[5]
  c4 = design_variables[6]
  c5 = design_variables[7]
  b1 = design_variables[8]
  b2 = design_variables[9]
  w1 = design_variables[10]
  w2 = design_variables[11]
  # Assigning the parameters
  omega = parameters[0]
  m3 = 100

  # Evaluating the important expressions
  exp7 = complex(k1 + k2 + k4 + k5 - m3*omega, omega*(c1 + c2 + c4 + c5))
  exp6 = complex(k1*b1 - k2*b2 + k4*b1 - k5*b2, omega*(c1*b1 - c2*b2 + c4*b1 - c5*b2))
  exp5 = complex(k1*w2 + k2*w2 - k4*w1 - k5*w1, omega*(c1*w2 + c2*w2 - c4*w1 - c5*w1))
  exp4 = complex(k1*b1*b1 + k2*b2*b2 + k4*b1*b1 + k5*b2*b2 - (m3*omega*omega*(b1*b1 - b1*b2 + b2*b2))/3, omega*(c1*b1*b1 + c2*b2*b2 + c4*b1*b1 + c5*b2*b2)) - (exp6*exp6)/exp7
  exp3 = complex((k1*b1*w2 - k2*b2*w2 - k4*b1*w1 + k5*b2*w1), omega*(c1*b1*w2 - c2*b2*w2 - c4*b1*w1 + c5*b2*w1)) - (exp5*exp6)/exp7
  exp2 = complex(k1*w2*w2 + k2*w2*w2 + k4*w1*w1 + k5*w1*w1 - (m3*omega*omega*w1*w1 -w1*w2 + w2*w2)/3, omega*(c1*w2*w2 + c2*w2*w2 + c4*w1*w1 + c5*w1*w1)) - (exp5*exp5/exp7) - (exp3*exp3)/exp4
  exp1 = complex(0,-(1/omega)*(k1*w2 + k2*w2 - k4*w1 - k5*w1) + (exp3/exp4)*(k1*b1 - k2*b2 + k4*b1 - k5*b2 - (exp6/(exp7*omega))*(k1 + k2 + k4 + k5)) + (exp5/(exp7*omega))*(k1 + k2 + k4 + k5))
  exp0 = -(1/exp7)*(complex(0, (1/omega)*(k1 + k2 + k4 + k5)) - (exp6/exp4)*(complex(0, (1/omega)*(k1*b1 - k2*b2 + k4*b1 - k5*b2)) + (exp1*exp3)/exp2 - (exp6/(omega*exp7))*complex(0, (k1 + k2 + k4 + k5))) + (exp1*exp5)/exp2)
  return abs(exp0)

# Function to evaluate the objective function and return the value for given values of parameters and design variables
def objective_function_2(parameters, design_variables):
  # design variables = [omega, m3]
  # parameters = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
  # Assigning the design variables
  omega = design_variables[0]
  m3 = 100

  # Assigning the parameters
  k1 = parameters[0]
  k2 = parameters[1]
  k4 = parameters[2]
  k5 = parameters[3]
  c1 = parameters[4]
  c2 = parameters[5]
  c4 = parameters[6]
  c5 = parameters[7]
  b1 = parameters[8]
  b2 = parameters[9]
  w1 = parameters[10]
  w2 = parameters[11]

  # Evaluating the important expressions
  exp7 = complex(k1 + k2 + k4 + k5 - m3*omega, omega*(c1 + c2 + c4 + c5))
  exp6 = complex(k1*b1 - k2*b2 + k4*b1 - k5*b2, omega*(c1*b1 - c2*b2 + c4*b1 - c5*b2))
  exp5 = complex(k1*w2 + k2*w2 - k4*w1 - k5*w1, omega*(c1*w2 + c2*w2 - c4*w1 - c5*w1))
  exp4 = complex(k1*b1*b1 + k2*b2*b2 + k4*b1*b1 + k5*b2*b2 - (m3*omega*omega*(b1*b1 - b1*b2 + b2*b2))/3, omega*(c1*b1*b1 + c2*b2*b2 + c4*b1*b1 + c5*b2*b2)) - (exp6*exp6)/exp7
  exp3 = complex((k1*b1*w2 - k2*b2*w2 - k4*b1*w1 + k5*b2*w1), omega*(c1*b1*w2 - c2*b2*w2 - c4*b1*w1 + c5*b2*w1)) - (exp5*exp6)/exp7
  exp2 = complex(k1*w2*w2 + k2*w2*w2 + k4*w1*w1 + k5*w1*w1 - (m3*omega*omega*w1*w1 -w1*w2 + w2*w2)/3, omega*(c1*w2*w2 + c2*w2*w2 + c4*w1*w1 + c5*w1*w1)) - (exp5*exp5/exp7) - (exp3*exp3)/exp4
  exp1 = complex(0,-(1/omega)*(k1*w2 + k2*w2 - k4*w1 - k5*w1) + (exp3/exp4)*(k1*b1 - k2*b2 + k4*b1 - k5*b2 - (exp6/(exp7*omega))*(k1 + k2 + k4 + k5)) + (exp5/(exp7*omega))*(k1 + k2 + k4 + k5))
  exp0 = -(1/exp7)*(complex(0, (1/omega)*(k1 + k2 + k4 + k5)) - (exp6/exp4)*(complex(0, (1/omega)*(k1*b1 - k2*b2 + k4*b1 - k5*b2)) + (exp1*exp3)/exp2 - (exp6/(omega*exp7))*complex(0, (k1 + k2 + k4 + k5))) + (exp1*exp5)/exp2)
  return abs(exp0)
