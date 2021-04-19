import cmath as cm
import numpy as np
import timeit

from objective_function import objective_function_1, objective_function_2

from numpy import asarray
from numpy import exp
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot

parameters1, parameters2 = [], []
bounds1 = []

# def evalObjective1(design_variables):
#   # design variables = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
#   # parameters1 = [omega]
#   global parameters1,bounds1
#   return -objective_function_1(parameters1, design_variables,bounds1)


# def evalObjective2(design_variables):
#   # design variables = [omega]
#   # parameters2 = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
#   global parameters2
#   return objective_function_2(parameters2, design_variables)

  

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
            
            # calculate delta=current_cost - previous_cost
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


# def run_SA():
#     global parameters1,parameters2,bounds1
#     T=20 # no of iterations
#     # Setting bounds for the design variables as required 
#     # l1= [[20121,30180],[20121,30180],[20121,30180],[20121,30180],[640,960],[640,960],[640,960],[640,960],[0,100],[0,100],[0,100],[0,100]]
#     l1 = [[20121,30180]]*4
#     l2 = [[640,960]]*4
#     l3 = [[0,100]]*2
#     l4 = [[0,100]]*2
#     l1.extend(l2)
#     l1.extend(l3)
#     l1.extend(l4)
    
#     bounds1=asarray(l1) # For [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2]
#     bounds2= asarray([[-100,100]]) # For [omega]
    
#     # best_values = [k1,k2,k4,k5,c1,c2,c4,c5,b1,b2,w1,w2,omega]
#     # We optimize over best_values[0:12] and over best_values[12]
#     # alternatingly. Initializing all of them as 0
    
#     best_values = np.zeros(13)
    
#     parameters1=[10] # Initialize with value of omega
#     parameters2 = [] # Empty list as we will feed value from first optimization
    
#     for t in range(T):
#         print('------------- Start of Iteration {} ------------- \n'.format(t+1))
#         # Minimization over the 12 design variables relating to the car's build
#         n_iterations = 1000
#         step_size =0.1
#         temp = 90
        
        
#         best, score, scores = simulated_annealing(evalObjective1, bounds1, n_iterations, step_size,temp,100,0.5)
#         parameters2 = best
#         best_values[:12] = parameters2
        
#         # Maximization over frequency space using the output of previous minimization as parameters
#         best,score,scores= simulated_annealing(evalObjective2, bounds2,n_iterations, step_size,temp,100,0.5)
#         parameters1 = best
#         best_values[12] = parameters1
        
#         print("Optimal value of k1 after iteration {} = {} \n".format(t+1, best_values[0]))
#         print("Optimal value of k2 after iteration {} = {} \n".format(t+1, best_values[1]))
#         print("Optimal value of k4 after iteration {} = {} \n".format(t+1, best_values[2]))
#         print("Optimal value of k5 after iteration {} = {} \n".format(t+1, best_values[3]))
#         print("Optimal value of c1 after iteration {} = {} \n".format(t+1, best_values[4]))
#         print("Optimal value of c2 after iteration {} = {} \n".format(t+1, best_values[5]))
#         print("Optimal value of c4 after iteration {} = {} \n".format(t+1, best_values[6]))
#         print("Optimal value of c5 after iteration {} = {} \n".format(t+1, best_values[7]))
#         print("Optimal value of b1 after iteration {} = {} \n".format(t+1, best_values[8]))
#         print("Optimal value of b2 after iteration {} = {} \n".format(t+1, best_values[9]))
#         print("Optimal value of w1 after iteration {} = {} \n".format(t+1, best_values[10]))
#         print("Optimal value of w2 after iteration {} = {} \n".format(t+1, best_values[11]))
#         print('\n -------------- End of Iteration {} -------------- \n \n \n'.format(t+1))
        
#     print("\n -------------- Results -------------- \n")
#     print("Optimal value of k1 = {} \n".format(best_values[0]))
#     print("Optimal value of k2 = {} \n".format(best_values[1]))
#     print("Optimal value of k4 = {} \n".format(best_values[2]))
#     print("Optimal value of k5 = {} \n".format(best_values[3]))
#     print("Optimal value of c1 = {} \n".format(best_values[4]))
#     print("Optimal value of c2 = {} \n".format(best_values[5]))
#     print("Optimal value of c4 = {} \n".format(best_values[6]))
#     print("Optimal value of c5 = {} \n".format(best_values[7]))
#     print("Optimal value of b1 = {} \n".format(best_values[8]))
#     print("Optimal value of b2 = {} \n".format(best_values[9]))
#     print("Optimal value of w1 = {} \n".format(best_values[10]))
#     print("Optimal value of w2 = {} \n".format(best_values[11]))
#     print("Optimal value of omega = {} \n".format(best_values[12]))


# print("Starting timer...\n")

# start = timeit.default_timer()

# run_SA()

# stop = timeit.default_timer()

# print(f'Time taken: {stop - start:.3f}s')