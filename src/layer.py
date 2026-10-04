import numpy as np
from activations import ACTIVATION_FN,DERIVATIVE, sigmoid, sigmoid_derivative

class Layer:
    def __init__(self,num_curr,num_prev,activation_fn_name:str):
        self.num_curr=num_curr
        self.num_prev=num_prev
        self.activation_fn_name=activation_fn_name
        self.activation_fn=sigmoid if activation_fn_name=="softmax" else ACTIVATION_FN[activation_fn_name]
        self.derivative_fn=sigmoid_derivative if activation_fn_name=="softmax" else DERIVATIVE[activation_fn_name]
        self.W,self.b=self.initialize_weights()
        self.Z,self.A=None,None
        
    def initialize_weights(self):
        m,n=self.num_prev,self.num_curr
        weights=np.random.randn(self.num_prev,self.num_curr)
        bias=np.zeros((1,self.num_curr))

        # he-weight(for relu), glorot weight(for sigmoig,tanh) initialization technique
        if self.activation_fn_name=="relu": weights=weights*np.sqrt(2/m)
        else: weights=weights*np.sqrt(1/m)
            
        return weights,bias
        
    # considering dimensions of input of each layer as m*n where m=number of examples, n=dimension of each input, 
    # and dimension of weights as n*k where n=dimension of each input and k=number of neurons

    def forward(self,A_prev):
        self.Z=np.dot(A_prev,self.W)+self.b
        
        if not self.activation_fn_name=="softmax":
            self.A=self.activation_fn(self.Z)
        else:
            exp_Z=np.exp(self.Z-np.max(self.Z,axis=1,keepdims=True))
            self.A=exp_Z/np.sum(exp_Z,axis=1,keepdims=True)
        return self.Z,self.A