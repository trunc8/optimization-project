#f(x)=x^2+(x-3)^2
# f(x,y)=(x+y-11)^2 +(x+y-7)^2
import numpy as np
from scipy.optimize import dual_annealing
#func = lambda x: np.abs(x*x - (x-3)**2)
#def func(x):
    #y#=np.abs((x[0]+x[1]-11)**2 +(x[0]+x[1]-7)**2)
    #return x,y
func = lambda x: (x[0]+x[1]-11)**2 +(x[0]+x[1]-7)**2
lw = [-5,-5] 
up = [5,5] 
ret = dual_annealing(func, bounds=list(zip(lw, up)), seed=1234)
print(ret.x)
#print(ret.y)
print(ret.fun)
