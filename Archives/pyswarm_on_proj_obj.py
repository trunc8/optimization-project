#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
import cmath as cm
import numpy as np
# from scipy.optimize import differential_evolution
import timeit

from pyswarm import pso

from objective_function import objective_function_1, objective_function_2

parameters1, parameters2 = [], []
bounds1 = []

def evalObjective1(design_variables):
  # design variables = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
  # parameters1 = [omega]
  global parameters1, bounds1
  return -objective_function_1(parameters1, design_variables, bounds1)

def evalObjective2(design_variables):
  # design variables = [omega]
  # parameters2 = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
  global parameters2
  return objective_function_2(parameters2, design_variables)


def benchmarkPSOAlgorithm():
  global parameters1, parameters2, bounds1
  T = 20 # Number of iterations

  # Setting bounds for the design variables as required by scipy
  # l1= [[20121,30180],[20121,30180],[20121,30180],[20121,30180],[640,960],[640,960],[640,960],[640,960],[0,100],[0,100],[0,100],[0,100]]
  l1 = [[20121,30180]]*4
  l2 = [[640,960]]*4
  l3 = [[0,100]]*2
  l4 = [[0,100]]*2
  l1.extend(l2)
  l1.extend(l3)
  l1.extend(l4)
  bounds1 = l1 # For [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
  bounds2 = [[-100,100]] # For [omega]

  # best_values = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2,omega]
  # We optimize over best_values[0:12] and over best_values[12]
  # alternatingly. Initializing all of them as 0
  best_values = np.zeros(13)


  # Initialising the parameters to the objective function  
  parameters1 = [10] # Initialize with value of omega
  parameters2 = [] # Empty list as we will feed value from first optimization
  

  for t in range(T):
    print('------------- Start of Iteration {} ------------- \n'.format(t+1))

    # Minimization over the 12 design variables relating to the car's build
    lb = [bound[0] for bound in bounds1]
    ub = [bound[1] for bound in bounds1]
    xopt, fopt = pso(evalObjective1, lb, ub, swarmsize = 25, omega = 0.7, phip=2, phig=2, maxiter=30000)
    parameters2 = xopt
    best_values[:12] = parameters2

    # result = differential_evolution(evalObjective1, bounds1, seed=0)
    # parameters2 = result.x
    # best_values[:12] = parameters2

    

    # Maximization over frequency space using the output of previous minimization as parameters    
    lb = [bound[0] for bound in bounds2]
    ub = [bound[1] for bound in bounds2]
    xopt, fopt = pso(evalObjective2, lb, ub, swarmsize = 25, omega = 0.7, phip=2, phig=2, maxiter=30000, minfunc=0, minstep=0)
    parameters1 = xopt
    best_values[12] = parameters1
    # result = differential_evolution(evalObjective2, bounds2, seed=0)
    # parameters1 = result.x
    # best_values[12] = parameters1

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

  print("\n -------------- Results -------------- \n")
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

if __name__ == '__main__':
  print("\nThis script will benchmark our optimization objective against",
    "pyswarm's Particle Swarm Optimization algorithm\n")
  print("Starting timer...\n")

  start = timeit.default_timer()

  benchmarkPSOAlgorithm()

  stop = timeit.default_timer()

  print(f'Time taken: {stop - start:.3f}s') 



