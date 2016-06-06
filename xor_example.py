#!/usr/bin/python
""" :author: Juan Perez
""" 

import numpy as np
rng = np.random
import matplotlib.pyplot as plt

class SimpleMLP:
    
    def __init__(self):
        self.W = rng.randn(8, 1);
        
    def compute_error(self, o, t):
        vec=o.transpose()-t;
        return np.sqrt(vec.dot(vec.transpose()))
        
    def logistic_function(self, x):
        return 1 / ( 1+np.exp(-x) )        
        
    def forward_computation(self, x):
        self.h1 = x[0]*self.W[0]+x[1]*self.W[2]
        self.h2 = x[0]*self.W[1]+x[1]*self.W[3]
        self.oh1 = self.logistic_function(self.h1)
        self.oh2 = self.logistic_function(self.h2)
        
        self.h3 = self.oh1*self.W[4]+self.oh2*self.W[6]
        self.h4 = self.oh1*self.W[5]+self.oh2*self.W[7]
        self.oh3 = self.logistic_function(self.h3)
        self.oh4 = self.logistic_function(self.h4)        
        
        return np.array([self.oh3, self.oh4])
        
    def compute_gradient(self, x, t):
        self.curr_out = self.forward_computation(x)
        grad_vector = np.zeros((8,1))
        
        grad_vector[4] = (self.oh3-t[0]) * self.oh3*(1-self.oh3) * self.oh1 
        grad_vector[5] = (self.oh4-t[1]) * self.oh4*(1-self.oh4) * self.oh1
        grad_vector[6] = (self.oh3-t[0]) * self.oh3*(1-self.oh3) * self.oh2
        grad_vector[7] = (self.oh4-t[1]) * self.oh4*(1-self.oh4) * self.oh2
                
        grad_vector[0] = x[0] * self.oh1 * (1-self.oh1) * \
                        ( (self.oh3-t[0])*self.oh3*(1-self.oh3)*self.W[4] + (self.oh4-t[1])*self.oh4*(1-self.oh4)*self.W[5] )
        grad_vector[1] = x[0] * self.oh2 * (1-self.oh2) * \
                        ( (self.oh3-t[0])*self.oh3*(1-self.oh3)*self.W[6] + (self.oh4-t[1])*self.oh4*(1-self.oh4)*self.W[7] )
        grad_vector[2] = x[1] * self.oh1 * (1-self.oh1) * \
                        ( (self.oh3-t[0])*self.oh3*(1-self.oh3)*self.W[4] + (self.oh4-t[1])*self.oh4*(1-self.oh4)*self.W[5] )
        grad_vector[3] = x[1] * self.oh2 * (1-self.oh2) * \
                        ( (self.oh3-t[0])*self.oh3*(1-self.oh3)*self.W[6] + (self.oh4-t[1])*self.oh4*(1-self.oh4)*self.W[7] )
        
        return grad_vector;
        
    def gd(self, D, epochs, epsilon):
        X = D[0]
        T = D[1]
        cost = np.zeros((epochs,1))
        for i in range(epochs):
            for s in range(X.shape[0]):
                x = X[s]
                t = T[s]
                self.W = self.W - epsilon*self.compute_gradient(x,t)
                cost[i] = cost[i] + self.compute_error(self.curr_out,t);
            cost[i] = cost[i]/X.shape[0]
            
        return cost
        
      

if __name__ == '__main__':
    xor_mlp = SimpleMLP()
    training_set = (np.array([[0, 0],[0, 1],[1, 0],[1, 1]]), np.array([[1, 0],[0, 1],[0, 1],[1, 0]]))
    c = xor_mlp.gd(training_set, 1000000, 0.1)
    
    plt.plot(c)    
    plt.show() 
    
    print xor_mlp.forward_computation(np.array([0, 0]))
    print xor_mlp.forward_computation(np.array([0, 1]))
    print xor_mlp.forward_computation(np.array([1, 0]))
    print xor_mlp.forward_computation(np.array([1, 1]))
    