import numpy as np
from layer import Layer
from losses import MSE,BinaryCrossEntropy,CategoricalCrossEntropy
class ANN:
    def __init__(self,dimensions:list[int],activation_fn_list:list[str],loss_fn="mse"):
        self.dimensions=dimensions
        self.activation_fn_list=activation_fn_list
        self.layers=[]
        self.X_input=None
        self.y_actual=None

        loss_map={"mse":MSE(),"bce":BinaryCrossEntropy(),"cce":CategoricalCrossEntropy()}
        self.loss_fn=loss_map[loss_fn]

    def actual_output(self,y_actual):
        self.y_actual=y_actual

    def take_input(self,X_input):
        self.X_input=X_input
        
    def build_layers(self):
        self.layers=[]
        m,n=self.X_input.shape
        current_dimension=[n]+self.dimensions

        for i in range(1,len(current_dimension)):
            layer=Layer(current_dimension[i],current_dimension[i-1],self.activation_fn_list[i-1])
            self.layers.append(layer)

    def forward_prop(self):
        if not self.layers: self.build_layers()
            
        A_prev=self.X_input
        for layer in self.layers:
            _,A=layer.forward(A_prev)
            A_prev=A
        return A

    def train(self,batch_size,optimizer,epochs=100,learning_rate=0.01):
        m=self.X_input.shape[0]
        X_initial=self.X_input
        # creating an empty list to keep track of loss at each epoch
        loss_history=[]
        
        for epoch in range(epochs):

            indices=np.arange(m)
            np.random.shuffle(indices)
            X_shuffled=self.X_input[indices]
            y_shuffled_actual=np.array(self.y_actual)[indices]

            for b in range(0,m,batch_size):
                X_batch=X_shuffled[b:b+batch_size]
                y_batch_actual=y_shuffled_actual[b:b+batch_size]
                current_batch_size=X_batch.shape[0]

                self.X_input=X_batch
                y_batch_pred=self.forward_prop()

                dA=self.loss_fn.gradient(y_batch_actual,y_batch_pred)
                
                for i in range(len(self.layers)-1,-1,-1):
                    curr_layer=self.layers[i]
                    A_prev=self.X_input if i==0 else self.layers[i-1].A
                    
                    # basic four lines of code for each layer for autograd(automatic differentiation) to run gradient descent
                    if i==len(self.layers)-1 and curr_layer.activation_fn_name=="softmax":
                        dZ=(y_batch_pred-y_batch_actual)/current_batch_size
                    else: dZ=dA*curr_layer.derivative_fn(curr_layer.Z)
                    
                    dA_prev=np.dot(dZ,curr_layer.W.T)
                    dW=np.dot(A_prev.T,dZ)
                    db=np.sum(dZ,axis=0,keepdims=True)
                    
                    # updating dA to work for the next previous layer
                    dA=dA_prev

                    # update weights and biases
                    if optimizer:
                        curr_layer.W=optimizer.update(curr_layer.W,dW,f"W{i}")
                        curr_layer.b=optimizer.update(curr_layer.b,db,f"b{i}")
                    else: raise ValueError("No optimizer entered")

            self.X_input=X_initial
            y_pred=self.forward_prop()
            loss=self.loss_fn(self.y_actual,y_pred)
            # print(f"Loss at epoch {epoch}:{loss}")
            loss_history.append(float(loss))
        return loss_history

    # backward pass function which implements a single backward pass on this ANN, returns the gradient with respect to the first input X
    # it does no mini or full batch gradient descent, just does a single backward pass for the given input batch
    def backward_step(self,y_batch_actual,y_batch_pred,optimizer=None):
        current_batch_size=self.X_input.shape[0]
        dA=self.loss_fn.gradient(y_batch_actual,y_batch_pred)

        for i in range(len(self.layers)-1,-1,-1):
            curr_layer=self.layers[i]
            A_prev=self.X_input if i==0 else self.layers[i-1].A

            if i==len(self.layers)-1 and curr_layer.activation_fn_name=="softmax":
                dZ=(y_batch_pred-y_batch_actual)/current_batch_size
            else:
                dZ=dA*curr_layer.derivative_fn(curr_layer.Z)
            dW=np.dot(A_prev.T,dZ)
            dA_prev=np.dot(dZ,curr_layer.W.T)
            db=np.sum(dZ,axis=0,keepdims=True)
            dA=dA_prev

            # update weights and biases
            if optimizer:
                curr_layer.W=optimizer.update(curr_layer.W,dW,f"W{i}")
                curr_layer.b=optimizer.update(curr_layer.b,db,f"b{i}")
            else: raise ValueError("No Optimizer Entered")

        return dA
  