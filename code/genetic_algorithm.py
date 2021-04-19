#!/usr/bin/env python

import cmath as cm
import numpy as np

from objective_function import objective_function_1, objective_function_2

# Function to calculate the fitness value of each solution in the current population
def fitness_function(objective, design_variables):
  fitness_value = np.zeros(design_variables.shape)
  for i in range(len(design_variables)):
    fitness_value[i] = objective(design_variables[i])
  return fitness_value


# Function to select the best solutions in the current generation as mating parents 
# to produce the offsprings of the next generation
def selection_function(fitness_values, number_of_mating_parents, curr_population):
    parents = np.zeros((number_of_mating_parents, curr_population.shape[1]))
    for i in range(number_of_mating_parents):
        max_index = np.where(fitness_values == np.max(fitness_values))
        max_index = max_index[0][0]
        parents[i,:] = curr_population[max_index, :]
        fitness_values[max_index] = -10000000000
    return parents

# Function to generate the next generation by doing crossover between two parents
def crossover_function(parents, offspring_size):
    offspring = np.zeros(offspring_size)
    crossover_index = np.uint8(offspring_size[1]/2)
    for i in range(offspring_size[0]):
        index_of_parent1 = i % parents.shape[0]
        index_of_parent2 = (i+1) % parents.shape[0]
        offspring[i, 0:crossover_index] = parents[index_of_parent1, 0:crossover_index]
        offspring[i, crossover_index:] = parents[index_of_parent2, crossover_index:]
    return offspring

# Function to add some variations to the offspring using mutation by changing a single gene in each offspring randomly 
def mutation_function(offspring):
    for i in range(offspring.shape[0]):
        random_index = np.random.randint(offspring.shape[1])
        random_value = np.random.uniform(-50.0, 50.0)
        offspring[i, random_index] = offspring[i, random_index] + random_value
    return offspring


def genetic_algorithm(objective, bounds, number_of_design_variables):
    number_of_solutions_per_population = 8 # Number of solutions per population
    number_of_mating_parents = 4 # Number of mating parents
    population_size = (number_of_solutions_per_population, number_of_design_variables) # Population size
    curr_population = np.random.uniform(-100.0, 100.0, population_size) # Initializing the current population
    number_of_generations = 10000 # Number of generations
    for gen in range(number_of_generations):
      fitness = fitness_function(objective, curr_population)
      parents = selection_function(fitness, number_of_mating_parents, curr_population)
      offspring_crossover = crossover_function(parents, (population_size[0] - parents.shape[0], number_of_design_variables))
      offspring_mutation = mutation_function(offspring_crossover)
      curr_population[0:parents.shape[0], :] = parents
      curr_population[parents.shape[0]:, :] = offspring_mutation
      
    fitness = fitness_function(objective, curr_population)
    best_fitness_value_index = np.where(fitness == np.max(fitness))
    best_fitness_value_index = best_fitness_value_index[0][0]
    # print("Best solution : {}".format(-1*curr_population[best_fitness_value_index, :]))
    # print("Best solution fitness : {}".format(fitness[best_fitness_value_index]))
    # best_values[0:12] = -1*curr_population[best_fitness_value_index, :]
    
    result = {}
    result["fun"] = fitness[best_fitness_value_index][0]
    result["x"] = curr_population[best_fitness_value_index, :]
    return result