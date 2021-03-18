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

# Function to calculate the fitness value of each solution in the current population
def fitness_function_1(parameters, design_variables, type = 'max'):
	fitness_value = np.zeros(design_variables.shape)
	if type == 'max':
		for i in range(len(design_variables)):
			fitness_value[i] = objective_function_1(parameters, design_variables[i])
	else:
		for i in range(len(design_variables)):
			fitness_value[i] = -objective_function_1(parameters, design_variables[i])
	return fitness_value

# Function to calculate the fitness value of each solution in the current population
def fitness_function_2(parameters, design_variables, type = 'max'):
	fitness_value = np.zeros(design_variables.shape)
	if type == 'max':
		for i in range(len(design_variables)):
			fitness_value[i] = objective_function_2(parameters, design_variables[i])
	else:
		for i in range(len(design_variables)):
			fitness_value[i] = -objective_function_2(parameters, design_variables[i])
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

T = 10 # Number of iterations
params = [10]  # Initialising the parameters to the objective function
np.random.seed(0)
best_values = np.zeros(13)
print('\n')
for t in range(T):
	print('------------- Start of Iteration {} ------------- \n'.format(t+1))
	number_of_design_variables = 12  # Number of design variables to be optimized
	number_of_solutions_per_population = 8 # Number of solutions per population
	number_of_mating_parents = 4 # Number of mating parents
	population_size = (number_of_solutions_per_population, number_of_design_variables) # Population size
	curr_population = np.random.uniform(-100.0, 100.0, population_size) # Initializing the current population
	number_of_generations = 10000 # Number of generations
	for gen in range(number_of_generations):
		fitness = fitness_function_1(params, curr_population, 'min')
		parents = selection_function(fitness, number_of_mating_parents, curr_population)
		offspring_crossover = crossover_function(parents, (population_size[0] - parents.shape[0], number_of_design_variables))
		offspring_mutation = mutation_function(offspring_crossover)
		curr_population[0:parents.shape[0], :] = parents
		curr_population[parents.shape[0]:, :] = offspring_mutation
		# if (gen % 500) == 0:
		# 	print("Best result after generation {}: {}".format(gen, np.max(fitness)))
	# print('\n')
	fitness = fitness_function_1(params, curr_population, 'min')
	best_fitness_value_index = np.where(fitness == np.max(fitness))
	best_fitness_value_index = best_fitness_value_index[0][0]
	# print("Best solution : {}".format(-1*curr_population[best_fitness_value_index, :]))
	# print("Best solution fitness : {}".format(fitness[best_fitness_value_index]))
	best_values[0:12] = -1*curr_population[best_fitness_value_index, :]
	params = -1*curr_population[best_fitness_value_index, :]  
	number_of_design_variables = 1  # Number of design variables to be optimized
	number_of_solutions_per_population = 8 # Number of solutions per population
	number_of_mating_parents = 4 # Number of mating parents
	population_size = (number_of_solutions_per_population, number_of_design_variables) # Population size
	curr_population = np.random.uniform(-100.0, 100.0, population_size) # Initializing the current population
	number_of_generations = 10000 # Number of generations
	for gen in range(number_of_generations):
		fitness = fitness_function_2(params, curr_population)
		parents = selection_function(fitness, number_of_mating_parents, curr_population)
		offspring_crossover = crossover_function(parents, (population_size[0] - parents.shape[0], number_of_design_variables))
		offspring_mutation = mutation_function(offspring_crossover)
		curr_population[0:parents.shape[0], :] = parents
		curr_population[parents.shape[0]:, :] = offspring_mutation
		# if (gen % 1000) == 0:
		# 	print("Best result after generation {}: {}".format(gen, np.max(fitness)))
	print('\n')
	fitness = fitness_function_2(params, curr_population)
	best_fitness_value_index = np.where(fitness == np.max(fitness))
	best_fitness_value_index = best_fitness_value_index[0][0]
	# print("Best solution : {}".format(curr_population[best_fitness_value_index, :]))
	# print("Best solution fitness : {}".format(fitness[best_fitness_value_index]))
	best_values[12:] = -1*curr_population[best_fitness_value_index, :]
	params = curr_population[best_fitness_value_index, :]
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
print('\n \n')
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