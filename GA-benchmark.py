import numpy as np
import cmath as cm
from scipy.optimize import differential_evolution

parameters = []

def objective1(design_variables):
  # design variables = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
  # parameters = [omega]
  global parameters

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
  # Negative sign so that we achieve minimization
  return -abs(exp0)


def objective2(design_variables):
  # design variables = [omega]
  # parameters = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
  global parameters

  # Assigning the design variable
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


def benchmark_cars():
  global parameters
  T = 10 # Number of iterations
  params = [10]  # Initialising the parameters to the objective function
  # np.random.seed(0)
  best_values = np.zeros(13)
  print('\n')

  l1 = [(20121,30180)]*4
  l2 = [(640,960)]*4
  l3 = [(0,2)]*2
  l4 = [(0,5)]*2
  l1.extend(l2)
  l1.extend(l3)
  l1.extend(l4)
  bounds1 = l1

  bounds2 = [[-10,10]]

  # Initialize with value of omega
  parameters = [10]

  for t in range(T):
    print('------------- Start of Iteration {} ------------- \n'.format(t+1))
    # Minimization over the 12 design variables relating to the car's build
    result = differential_evolution(objective1, bounds1, seed=0)
    parameters = result.x
    best_values[:12] = parameters

    # Maximization over frequency space using the output of previous minimization as parameters
    result = differential_evolution(objective2, bounds2, seed=0)
    parameters = result.x
    best_values[12] = parameters

    print("Optimal value of k1 after iteration {} = {} \n".format(t+1, best_values[0]))
    print("Optimal value of k2 after iteration {} = {} \n".format(t+1, best_values[1]))
    print("Optimal value of k4 after iteration {} = {} \n".format(t+1, best_values[2]))
    print("Optimal value of k5 after iteration {} = {} \n".format(t+1, best_values[3]))
    print("Optimal value of c1 after iteration {} = {} \n".format(t+1, best_values[4]))
    print("Optimal value of c2 after iteration {} = {} \n".format(t+1, best_values[5]))
    print("Optimal value of c4 after iteration {} = {} \n".format(t+1, best_values[6]))
    print("Optimal value of c5 after iteration {} = {} \n".format(t+1, best_values[7]))
    print("Optimal value of b1 after iteration {} = {} \n".format(t+1, best_values[8]))
    print("Optimal value of b2 after iteration {} = {} \n".format(t+1, best_values[9]))
    print("Optimal value of w1 after iteration {} = {} \n".format(t+1, best_values[10]))
    print("Optimal value of w2 after iteration {} = {} \n".format(t+1, best_values[11]))
    print('\n -------------- End of Iteration {} -------------- \n \n \n'.format(t+1))

  print("Optimal value of k1 = {} \n".format(best_values[0]))
  print("Optimal value of k2 = {} \n".format(best_values[1]))
  print("Optimal value of k4 = {} \n".format(best_values[2]))
  print("Optimal value of k5 = {} \n".format(best_values[3]))
  print("Optimal value of c1 = {} \n".format(best_values[4]))
  print("Optimal value of c2 = {} \n".format(best_values[5]))
  print("Optimal value of c4 = {} \n".format(best_values[6]))
  print("Optimal value of c5 = {} \n".format(best_values[7]))
  print("Optimal value of b1 = {} \n".format(best_values[8]))
  print("Optimal value of b2 = {} \n".format(best_values[9]))
  print("Optimal value of w1 = {} \n".format(best_values[10]))
  print("Optimal value of w2 = {} \n".format(best_values[11]))
  print("Optimal value of omega = {} \n".format(best_values[12]))

benchmark_cars()