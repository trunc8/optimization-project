# This script is not meant to be executed alone. It will be called by other
# optimizer modules to evaluate the objective function for 
# different parameter values

import numpy as np
import cmath as cm

# Function to evaluate the objective function and return the value for given values of parameters and design variables
def objective_function_1(parameters, design_variables, bounds):
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
  m3 = 625

  # Evaluating the important expressions
  try:
    exp7 = complex(k1 + k2 + k4 + k5 - m3*omega, omega*(c1 + c2 + c4 + c5))
    exp6 = complex(k1*b1 - k2*b2 + k4*b1 - k5*b2, omega*(c1*b1 - c2*b2 + c4*b1 - c5*b2))
    exp5 = complex(k1*w2 + k2*w2 - k4*w1 - k5*w1, omega*(c1*w2 + c2*w2 - c4*w1 - c5*w1))
    exp4 = complex(k1*b1*b1 + k2*b2*b2 + k4*b1*b1 + k5*b2*b2 - (m3*omega*omega*(b1*b1 - b1*b2 + b2*b2))/3, omega*(c1*b1*b1 + c2*b2*b2 + c4*b1*b1 + c5*b2*b2)) - (exp6*exp6)/exp7
    exp3 = complex((k1*b1*w2 - k2*b2*w2 - k4*b1*w1 + k5*b2*w1), omega*(c1*b1*w2 - c2*b2*w2 - c4*b1*w1 + c5*b2*w1)) - (exp5*exp6)/exp7
    exp2 = complex(k1*w2*w2 + k2*w2*w2 + k4*w1*w1 + k5*w1*w1 - (m3*omega*omega*w1*w1 -w1*w2 + w2*w2)/3, omega*(c1*w2*w2 + c2*w2*w2 + c4*w1*w1 + c5*w1*w1)) - (exp5*exp5/exp7) - (exp3*exp3)/exp4
    exp1 = complex(0,-(1/omega)*(k1*w2 + k2*w2 - k4*w1 - k5*w1) + (exp3/exp4)*(k1*b1 - k2*b2 + k4*b1 - k5*b2 - (exp6/(exp7*omega))*(k1 + k2 + k4 + k5)) + (exp5/(exp7*omega))*(k1 + k2 + k4 + k5))
    exp0 = -(1/exp7)*(complex(0, (1/omega)*(k1 + k2 + k4 + k5)) - (exp6/exp4)*(complex(0, (1/omega)*(k1*b1 - k2*b2 + k4*b1 - k5*b2)) + (exp1*exp3)/exp2 - (exp6/(omega*exp7))*complex(0, (k1 + k2 + k4 + k5))) + (exp1*exp5)/exp2)
  except ZeroDivisionError as error:
    # Penalize ZeroDivisionErrors.
    return 1e6
  except Exception as exception:
    # Output unexpected Exceptions.
    print(f"Exception occurred: {exception}")

  # Calculating the penalties
  penalties = [0 for _ in range(12)]
  if (k1 > bounds[0][1]):
    penalties[0] = 1e4*((k1 - bounds[0][1])**2)
  if (k1 < bounds[0][0]):
    penalties[0] = 1e4*((bounds[0][0] - k1)**2)
  if (k2 > bounds[1][1]):
    penalties[1] = 1e4*((k2 - bounds[1][1])**2)
  if (k2 < bounds[1][0]):
    penalties[1] = 1e4*((bounds[1][0] - k2)**2)
  if (k4 > bounds[2][1]):
    penalties[2] = 1e4*((k4 - bounds[2][1])**2)
  if (k4 < bounds[2][0]):
    penalties[2] = 1e4*((bounds[2][0] - k4)**2)
  if (k5 > bounds[3][1]):
    penalties[3] = 1e4*((k5 - bounds[3][1])**2)
  if (k5 < bounds[3][0]):
    penalties[3] = 1e4*((bounds[3][0] - k5)**2)
  if (c1 > bounds[4][1]):
    penalties[4] = 1e4*((c1 - bounds[4][1])**2)
  if (c1 < bounds[4][0]):
    penalties[4] = 1e4*((bounds[4][0] - c1)**2)
  if (c2 > bounds[5][1]):
    penalties[5] = 1e4*((c2 - bounds[5][1])**2)
  if (c2 < bounds[5][0]):
    penalties[5] = 1e4*((bounds[5][0] - c2)**2)
  if (c4 > bounds[6][1]):
    penalties[6] = 1e4*((c4 - bounds[6][1])**2)
  if (c4 < bounds[6][0]):
    penalties[6] = 1e4*((bounds[6][0] - c4)**2)
  if (c5 > bounds[7][1]):
    penalties[7] = 1e4*((c5 - bounds[7][1])**2)
  if (c5 < bounds[7][0]):
    penalties[7] = 1e4*((bounds[7][0] - c5)**2)
  if (b1 > bounds[8][1]):
    penalties[8] = 1e4*((b1 - bounds[8][1])**2)
  if (b1 < bounds[8][0]):
    penalties[8] = 1e4*((bounds[8][0] - b1)**2)
  if (b2 > bounds[9][1]):
    penalties[9] = 1e4*((b2 - bounds[9][1])**2)
  if (b2 < bounds[9][0]):
    penalties[9] = 1e4*((bounds[9][0] - b2)**2)
  if (w1 > bounds[10][1]):
    penalties[10] = 1e4*((w1 - bounds[10][1])**2)
  if (w1 < bounds[10][0]):
    penalties[10] = 1e4*((bounds[10][0] - w1)**2)
  if (w2 > bounds[11][1]):
    penalties[11] = 1e4*((w2 - bounds[11][1])**2)
  if (w2 < bounds[11][0]):
    penalties[11] = 1e4*((bounds[11][0] - w2)**2)

  return abs(exp0) + sum(penalties)

# Function to evaluate the objective function and return the value for given values of parameters and design variables
def objective_function_2(parameters, design_variables):
  # design variables = [omega, m3]
  # parameters = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
  # Assigning the design variables
  omega = design_variables[0]
  m3 = 625

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
  try:
    exp7 = complex(k1 + k2 + k4 + k5 - m3*omega, omega*(c1 + c2 + c4 + c5))
    exp6 = complex(k1*b1 - k2*b2 + k4*b1 - k5*b2, omega*(c1*b1 - c2*b2 + c4*b1 - c5*b2))
    exp5 = complex(k1*w2 + k2*w2 - k4*w1 - k5*w1, omega*(c1*w2 + c2*w2 - c4*w1 - c5*w1))
    exp4 = complex(k1*b1*b1 + k2*b2*b2 + k4*b1*b1 + k5*b2*b2 - (m3*omega*omega*(b1*b1 - b1*b2 + b2*b2))/3, omega*(c1*b1*b1 + c2*b2*b2 + c4*b1*b1 + c5*b2*b2)) - (exp6*exp6)/exp7
    exp3 = complex((k1*b1*w2 - k2*b2*w2 - k4*b1*w1 + k5*b2*w1), omega*(c1*b1*w2 - c2*b2*w2 - c4*b1*w1 + c5*b2*w1)) - (exp5*exp6)/exp7
    exp2 = complex(k1*w2*w2 + k2*w2*w2 + k4*w1*w1 + k5*w1*w1 - (m3*omega*omega*w1*w1 -w1*w2 + w2*w2)/3, omega*(c1*w2*w2 + c2*w2*w2 + c4*w1*w1 + c5*w1*w1)) - (exp5*exp5/exp7) - (exp3*exp3)/exp4
    exp1 = complex(0,-(1/omega)*(k1*w2 + k2*w2 - k4*w1 - k5*w1) + (exp3/exp4)*(k1*b1 - k2*b2 + k4*b1 - k5*b2 - (exp6/(exp7*omega))*(k1 + k2 + k4 + k5)) + (exp5/(exp7*omega))*(k1 + k2 + k4 + k5))
    exp0 = -(1/exp7)*(complex(0, (1/omega)*(k1 + k2 + k4 + k5)) - (exp6/exp4)*(complex(0, (1/omega)*(k1*b1 - k2*b2 + k4*b1 - k5*b2)) + (exp1*exp3)/exp2 - (exp6/(omega*exp7))*complex(0, (k1 + k2 + k4 + k5))) + (exp1*exp5)/exp2)
  except ZeroDivisionError as error:
    # Penalize ZeroDivisionErrors.
    return 1e6
  except Exception as exception:
    # Output unexpected Exceptions.
    print(f"Exception occurred: {exception}")
  
  # Calculating the penalty
  penalty = 0
  if omega > 100:
    penalty = 1e4*((omega - 100)**2)
  if omega < -100:
    penalty = 1e4*((-100 - omega)**2)
  
  return abs(exp0) - penalty
  