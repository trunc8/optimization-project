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
# from objective_function import objective_function_1, objective_function_2
from test_objectives import function1, function2, himmelblau, banana

# Import of optimization libraries and algorithms
#### Genetic Algorithm
from genetic_algorithm import genetic_algorithm
from scipy.optimize import differential_evolution

#### Simulated Annealing
from simulated_annealing import simulated_annealing
from scipy.optimize import dual_annealing

#### Particle Swarm Optimization
from particle_swarm_optimization import Particle_swarm
from pyswarm import pso


class Test_Optimizer:
  def __init__(self):
    self.f2_optimal_values = []
    self.function = None
    self.bounds = None
    self.numdesign = 0

  def testGA(self):
    result = genetic_algorithm(self.function, self.bounds, self.numdesign)
    xopt = result['x']
    fopt = result['fun']

    print("optimal point:", [round(i,3) for i in xopt])  #print best position
    print("optimal function value:", fopt)  #print objective function value at best position
    return xopt, fopt


  def testGABenchmark(self):
    result = differential_evolution(self.function, self.bounds, seed=0)
    xopt = result['x']
    fopt = result['fun']

    print("optimal point:", [round(i,3) for i in xopt])  #print best position
    print("optimal function value:", fopt)  #print objective function value at best position
    return xopt, fopt

  def testSA(self):
    n_iterations = 1000
    step_size =0.1
    temp = 90    
    best, score, scores = simulated_annealing(self.function, np.asarray(self.bounds), n_iterations, step_size, temp, 100, 0.5)
    xopt = best
    fopt = score

    print("optimal point:", [round(i,3) for i in xopt])  #print best position
    print("optimal function value:", fopt)  #print objective function value at best position
    return xopt, fopt

  def testSABenchmark(self):
    result = dual_annealing(self.function, self.bounds, seed=0)
    xopt = result['x']
    fopt = result['fun']

    print("optimal point:", [round(i,3) for i in xopt])  #print best position
    print("optimal function value:", fopt)  #print objective function value at best position
    return xopt, fopt


  def testPSO(self):
    xopt, fopt = Particle_swarm(self.function, self.numdesign, self.bounds, 25, 30000)

    print("optimal point:", [round(i,3) for i in xopt])  #print best position
    print("optimal function value:", fopt)  #print objective function value at best position
    return xopt, fopt

  def testPSOBenchmark(self):
    lb = [bound[0] for bound in self.bounds]
    ub = [bound[1] for bound in self.bounds]

    xopt, fopt = pso(self.function, lb, ub, swarmsize = 25, omega = 0.7, phip=2, phig=2, maxiter=30000, minfunc=0, minstep=0)

    print("optimal point:", [round(i,3) for i in xopt])  #print best position
    print("optimal function value:", fopt)  #print objective function value at best position
    return xopt, fopt

  def test(self):
    print('GA')
    self.testGA()
    print('GA_benchmark')
    self.testGABenchmark()
    print('SA')
    self.testSA()
    print('SA_benchmark')
    self.testSABenchmark()
    print('PSO')
    self.testPSO()
    print('PSO_benchmark')
    self.testPSOBenchmark()


  def runTests(self):
    print("\nTesting against custom function 1\n")
    self.function = function1
    self.numdesign = 10
    self.bounds = [[-100,100]]*self.numdesign
    self.test()


    print("\nTesting against custom function 2\n")
    self.function = function2
    self.numdesign = 10
    self.bounds = [[-100,100]]*self.numdesign
    self.test()

    print("\nTesting against Himmelblau function\n")
    self.function = himmelblau
    self.numdesign = 2
    self.bounds = [[-100,100]]*self.numdesign
    self.test()

    print("\nTesting against Banana function\n")
    self.function = banana
    self.numdesign = 2
    self.bounds = [[-100,100]]*self.numdesign
    self.test()



if __name__ == '__main__':
  # Setting random seeds as fixed for repeatability
  random.seed(0)
  np.random.seed(0)

  # Create an object of Optimizer class
  optimizer_object = Test_Optimizer()

  print("\nWelcome! This script will test all our optimization algorithms\n")

  print("Starting timer...\n")

  start = timeit.default_timer()
  optimizer_object.runTests()
  stop = timeit.default_timer()

  print(f'Time taken by testing: {stop - start:.3f}s')

  print("Thank you!")
