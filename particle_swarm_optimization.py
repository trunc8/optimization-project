#!/usr/bin/env python
# coding: utf-8

# In[31]:


import math
import random
import time


# In[32]:


#Particle helper class
class Particle:
    def __init__(self,x0):
        self.n = len(x0)
        
        self.pos = x0  #current position
        self.pos_best = None  #best position
        
        self.vel = [random.uniform(-1,1) for i in range(self.n)]  #current velocity
        
        self.fval = math.inf  #current function value
        self.fval_best = math.inf  #best function value of particle
        
        #self.iters = 0

    #update current function value
    def set_fval(self,J):
        self.fval = J(self.pos)

        if self.fval < self.fval_best:  #see if we have found a new best and update in required
            self.pos_best = self.pos
            self.fval_best = self.fval

    #update current velocity
    def update_velocity(self, pos_best_g):
        w = 0.5  #inertia for current velocity
        c1 = 1 #1.2 - 0.2 * self.iters/maxiter  #cognative constant - factor for personal decision
        c2 = 2 #1.8 + 0.2 * self.iters/maxiter  #social constant - factor for herd decision

        for i in range(self.n):
            vel_cognitive = c1 * random.random() * (self.pos_best[i] - self.pos[i]) #calculate the cognitive velocity
            vel_social = c2 * random.random() * (pos_best_g[i] - self.pos[i]) #calculate the social velocity
            self.vel[i]= w * self.vel[i] + vel_cognitive + vel_social #finally update the particle velocity
            
        #self.iters += 1

    # update current particle position
    def update_position(self,bounds):
        for i in range(self.n):
            self.pos[i] = self.pos[i] + self.vel[i] #next position = current position + velocity * (time = 1)

            # adjusting for bounds
            if self.pos[i] < bounds[i][0]:
                self.pos[i] = bounds[i][0]
            if self.pos[i] > bounds[i][1]:
                self.pos[i] = bounds[i][1]


# In[33]:


#Optimiser
def Particle_swarm(J,numdesign,bounds,num_particles,maxiter):
    fval_best_g = math.inf  #best fval for group
    pos_best_g = None  #best position for group

    swarm=[]  #initialize
    for i in range(num_particles):
        x = [random.random() for i in range(numdesign)]
        swarm.append(Particle(x))

    for i in range(maxiter):  #iterate
        for j in range(num_particles):
            swarm[j].set_fval(J)
            
            if swarm[j].fval < fval_best_g: #check for best global value
                pos_best_g = list(swarm[j].pos)
                fval_best_g = float(swarm[j].fval)

            # cycle through swarm and update velocities and position
        for j in range(0,num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position(bounds)

    print("optimal point:", pos_best_g)  #print best position
    print("optimal function value:", fval_best_g)  #print objective function value at best position
    
    #return fval_best_g


# In[34]:


#Objective function
def function(x):
    f = 0
    # for i in range(len(x)):
    #     f += x[i]**2 + (x[i]-3)**2
    f = (x[0]**2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2
    return f


# In[35]:


#x0 = [99,87,14,-24,13,35,15,14,-24,13]  #starting point
bounds = [(-100,100),(-100,100)]  #bounds
numdesign = 2

start = time.time()
for _ in range(100):
Particle_swarm(function, numdesign, bounds, 25, 30000)  #run optimiser
end = time.time()

print(end-start)  #get eval time


# In[ ]:




