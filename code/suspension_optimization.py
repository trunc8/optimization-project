#!/usr/bin/env python3

import argparse
import cmath as cm
import csv
import numpy as np
import os, sys
import random
import timeit
from tqdm import tqdm

# Objective function imports
from objective_function import objective_function_1, objective_function_2

# Import of optimization libraries and algorithms

#### Genetic Algorithm
# from genetic_algorithm import ??
from scipy.optimize import differential_evolution

#### Simulated Annealing
from simulated_annealing import simulated_annealing
from scipy.optimize import basinhopping # ??

#### Particle Swarm Optimization
from particle_swarm_optimization import Particle_swarm
from pyswarm import pso


class Suspension_Optimizer:
  def __init__(self):
    self.parameters1, self.parameters2 = None, None
    self.bounds1, self.bounds2 = None, None
    self.algorithm = ""
    self.verbose = True
    self.series_of_best_values = []

  def evalObjective1(self, design_variables):
    # design variables = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
    # parameters1 = [omega]
    return -objective_function_1(self.parameters1, design_variables, self.bounds1)

  def evalObjective2(self, design_variables):
    # design variables = [omega]
    # parameters2 = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
    return objective_function_2(self.parameters2, design_variables)


  def runOptimizer(self):
    T = 20 # Number of iterations

    # Setting bounds for the design variables as required by scipy
    # l1= [[20121,30180],[20121,30180],[20121,30180],[20121,30180],[640,960],[640,960],[640,960],[640,960],[2,3.5],[2,3.5],[0.75,1.2],[0.75,1.2]]
    l1 = [[20121,30180]]*4
    l2 = [[640,960]]*4
    l3 = [[2,3.5]]*2
    l4 = [[0.75,1.2]]*2
    l1.extend(l2)
    l1.extend(l3)
    l1.extend(l4)
    self.bounds1 = l1 # For [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
    self.bounds2 = [[-100,100]] # For [omega]

    # best_values = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2,omega]
    # We optimize over best_values[0:12] and over best_values[12]
    # alternatingly. Initializing all of them as 0
    best_values = np.zeros(13)
    obj_value = 0


    # Initialising the parameters to the objective function  
    self.parameters1 = [10] # Initialize with value of omega
    self.parameters2 = [] # Empty list as we will feed value from first optimization
    

    # for t in (tqdm(range(T)) if not self.verbose else range(T)):
    for t in tqdm(range(T), disable=self.verbose):
      if self.verbose:
        print('------------- Start of Iteration {} ------------- \n'.format(t+1))
      
      if self.algorithm == 'GA':
        pass
      
      elif self.algorithm == 'GA_benchmark':
        # Minimization over the 12 design variables relating to the car's build
        result = differential_evolution(self.evalObjective1, self.bounds1, seed=0)
        self.parameters2 = result.x
        best_values[:12] = self.parameters2

        # Maximization over frequency space using the output of previous minimization as parameters
        result = differential_evolution(self.evalObjective2, self.bounds2, seed=0)
        self.parameters1 = result.x
        best_values[12] = self.parameters1
        obj_value = result.fun
      
      elif self.algorithm == 'SA':
        n_iterations = 1000
        step_size =0.1
        temp = 90
        
        best, score, scores = simulated_annealing(self.evalObjective1, np.asarray(self.bounds1), n_iterations, step_size,temp,100,0.5)
        self.parameters2 = best
        best_values[:12] = self.parameters2
        
        # Maximization over frequency space using the output of previous minimization as parameters
        best, score, scores= simulated_annealing(self.evalObjective2, np.asarray(self.bounds2), n_iterations, step_size,temp,100,0.5)
        self.parameters1 = best
        best_values[12] = self.parameters1
        obj_value = score
      
      elif self.algorithm == 'SA_benchmark':
        pass
      
      elif self.algorithm == 'PSO':
        numdesign = 12
        xopt, fopt = Particle_swarm(self.evalObjective1, numdesign, self.bounds1, 25, 30000)
        self.parameters2 = xopt
        best_values[:12] = self.parameters2

        numdesign = 1
        xopt, fopt = Particle_swarm(self.evalObjective2, numdesign, self.bounds2, 25, 30000)
        self.parameters1 = xopt
        best_values[12] = self.parameters1[0]
        obj_value = fopt
      
      elif self.algorithm == 'PSO_benchmark':
        lb = [bound[0] for bound in self.bounds1]
        ub = [bound[1] for bound in self.bounds1]
        xopt, fopt = pso(self.evalObjective1, lb, ub, swarmsize = 25, omega = 0.7, phip=2, phig=2, maxiter=30000)
        self.parameters2 = xopt
        best_values[:12] = self.parameters2

        lb = [bound[0] for bound in self.bounds2]
        ub = [bound[1] for bound in self.bounds2]
        xopt, fopt = pso(self.evalObjective2, lb, ub, swarmsize = 25, omega = 0.7, phip=2, phig=2, maxiter=30000, minfunc=0, minstep=0)
        self.parameters1 = xopt
        best_values[12] = self.parameters1
        obj_value = fopt
      
      else:
        print("Algorithm not in list! Exiting.")
        sys.exit(0)

      # Storing time series of optimal values to write to csv file
      self.series_of_best_values.append([round(i,2) for i in (best_values)])
      self.series_of_best_values[t].append(round(obj_value,4))

      if self.verbose:
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

    with open(os.path.join(sys.path[0], f'../results/{self.algorithm}_design_variables.csv'), 'w') as file:
      writer = csv.writer(file)
      writer.writerow(["k1","k2","k4","k5","c1","c2","c4","c5","b1","b2","w1","w2","omega","Objective"])
      writer.writerows(self.series_of_best_values)

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
  # Setting random seeds as fixed for repeatability
  random.seed(0)
  np.random.seed(0)
  # Create an object of Optimizer class and give it algorithm name
  optimizer_object = Suspension_Optimizer()

  # Read algorithm name from terminal input
  parser = argparse.ArgumentParser(description='Optimize car suspension using '+
                                               'different optimization methods')
  parser.add_argument('-a', '--algorithm', help='Input optimization method', default='GA_benchmark')
  parser.add_argument('-v', action='store_true', help='Verbose output')
  args = parser.parse_args()

  if args.algorithm in ["GA", "GA_benchmark", "SA", "SA_benchmark", "PSO", "PSO_benchmark"]:
    optimizer_object.algorithm = args.algorithm
  else:
    optimizer_object.algorithm = "GA_benchmark"

  optimizer_object.verbose = args.v


  print("\nWelcome! This script will optimize our objective using ",
        f"{optimizer_object.algorithm}\n")

  print("Starting timer...\n")
  start = timeit.default_timer()

  optimizer_object.runOptimizer()

  stop = timeit.default_timer()
  print(f'Time taken by {optimizer_object.algorithm}: {stop - start:.3f}s')
  print("Thank you!")
