import numpy as np 
import cmath as cm
 
# Function to evaluate the objective_function_1 function and return the value for given values of parameters and design variables
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

# Function to evaluate the objective_function_1 function and return the value for given values of parameters and design variables
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

# objective_function_1 function
#def objective_function_1(x):
#	return x[0]**2.0 - 4*x[0] + x[1]**2.0
 
# simulated annealing algorithm
def simulated_annealing(objective_function_1, parameters, bounds, n_iterations, step_size, temp):
	# generate an initial point
	optimal  = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	##optimal  = np.zeros(12)
	# evaluate the initial point
	optimal_value = objective_function_1(parameters, optimal )
	# current working solution
	curr, curr_value = optimal , optimal_value
	# run the algorithm
	for i in range(n_iterations):
		# take a step
		neighbour = curr + np.random.randn(len(bounds)) * step_size
		# evaluate neighbour point
		neighbour_value= objective_function_1(parameters, neighbour)
		# check for new optimal  solution
		if neighbour_value< optimal_value:
			# store new optimal  point
			optimal , optimal_value = neighbour, neighbour_value
			#print('>%d f(%s) = %.5f' % (i, optimal , optimal_value))
			#print('After iteration {} = {} \n'.format(i+1))
			print("Optimal value of k1 after iteration {} = {} \n".format(i+1, optimal [0]))
			print("Optimal value of k2 after iteration {} = {} \n".format(i+1, optimal [1]))
			print("Optimal value of k4 after iteration {} = {} \n".format(i+1, optimal [2]))
			print("Optimal value of k5 after iteration {} = {} \n".format(i+1, optimal [3]))
			print("Optimal value of c1 after iteration {} = {} \n".format(i+1, optimal [4]))
			print("Optimal value of c2 after iteration {} = {} \n".format(i+1, optimal [5]))
			print("Optimal value of c4 after iteration {} = {} \n".format(i+1, optimal [6]))
			print("Optimal value of c5 after iteration {} = {} \n".format(i+1, optimal [7]))
			print("Optimal value of b1 after iteration {} = {} \n".format(i+1, optimal [8]))
			print("Optimal value of b2 after iteration {} = {} \n".format(i+1, optimal [9]))
			print("Optimal value of w1 after iteration {} = {} \n".format(i+1, optimal [10]))
			print("Optimal value of w2 after iteration {} = {} \n".format(i+1, optimal [11]))
			print('\n')
		# difference between neighbour and current point evaluation
		diff = neighbour_value- curr_value
		# calculate temperature for current epoch
		t = temp / float(i + 1)
		# calculate metropolis acceptance criterion
		metropolis = np.exp(-diff / t)
		# checking if we should keep the new point and also accepting the point with some probability
		if diff < 0 or np.random.rand() < metropolis:
			# store the new current point
			curr, curr_value = neighbour, neighbour_value
	return [optimal , optimal_value]
 
# seed the pseudorandom number generator
np.random.seed(1)
bounds = np.asarray([[20120, 30180], [20120, 30180], [20120, 30180], [20120, 30180], [640, 960], [640, 960], [640, 960], [640, 960], [5, 15], [5, 15], [5, 15], [5, 15], [5, 15]])
n_iterations = 10000 # total no of iterations
step_size = 0.1  # maximum step size
temp = 10 # initial temperature
params =[10] # Initialising the parameters to the objective function
optimal , optimal_value = simulated_annealing(objective_function_1, params, bounds, n_iterations, step_size, temp)
#print('f(%s) = %f' % (optimal , score))
print("Optimal value of k1 = {} \n".format(optimal [0]))
print("Optimal value of k2 = {} \n".format(optimal [1]))
print("Optimal value of k4 = {} \n".format(optimal [2]))
print("Optimal value of k5 = {} \n".format(optimal [3]))
print("Optimal value of c1 = {} \n".format(optimal [4]))
print("Optimal value of c2 = {} \n".format(optimal [5]))
print("Optimal value of c4 = {} \n".format(optimal [6]))
print("Optimal value of c5 = {} \n".format(optimal [7]))
print("Optimal value of b1 = {} \n".format(optimal [8]))
print("Optimal value of b2 = {} \n".format(optimal [9]))
print("Optimal value of w1 = {} \n".format(optimal [10]))
print("Optimal value of w2 = {} \n".format(optimal [11]))
print("Optimal value of omega = {} \n".format(optimal [12]))

