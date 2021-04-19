import cmath as cm
import numpy as np
import timeit

from objective_function import objective_function_1, objective_function_2

from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed


def simulated_annealing(objective, bounds, n_iterations, step_size, initial_temp,nrep,alpha):
    #random starting point
    solution=bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    #best_eval=objective(best)
    temp=initial_temp
    scores=list()
    iterations_per_temp=nrep
    for i in range(n_iterations):
        for j in range(iterations_per_temp):
            #select a new point from neighborhood---
            neighbor=solution + randn(len(bounds)) * step_size
            for i in range(len(bounds)):
                neighbor[i] = max(bounds[i][0], min(bounds[i][1],neighbor[i]))
            previous_cost=objective(solution)
            current_cost=objective(neighbor)
            
            # calculate delta as difference of costs
            diff=current_cost-previous_cost
            
            if diff<0:
                solution=neighbor
                scores.append(current_cost)
                continue;
            else:
                metropolis = exp(-diff / temp)
                if rand()<metropolis:
                    solution=neighbor
        temp=temp*alpha
    best_score=objective(solution)
    return [solution,best_score,scores]
