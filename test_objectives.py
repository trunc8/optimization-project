#Objective function
def function1(x):
    f = 0
    for i in range(len(x)):
        f += x[i]**2 + (x[i]-3)**2
    return f

def function2(x): #local min around -22, global min around 29
    f = 0
    for i in range(len(x)):
        f += (x[i]-40)*(x[i]-10)*(x[i]+10)*(x[i]+30)
    return f*1./40000

def himmelblau(x):
    # 2-dimensional x
    x1 = x[0]
    x2 = x[1]
    f = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
    return f


def banana(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5