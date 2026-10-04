import numpy as np
class MSE:
    def __call__(self,y_true,y_pred):
        return np.mean((y_pred-y_true)**2)
    
    def gradient(self,y_true,y_pred):
        m=y_true.shape[0]
        return 2*(y_pred-y_true)/m

class BinaryCrossEntropy:
    def __call__(self,y_true,y_pred):
        eps=1e-15
        y_pred=np.clip(y_pred,eps,1-eps)
        return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

    def gradient(self,y_true,y_pred):
        eps=1e-15
        m=y_true.shape[0]
        y_pred=np.clip(y_pred,eps,1-eps)
        return -((y_true/y_pred)+((y_true-1)/(1-y_pred)))/m

class CategoricalCrossEntropy:
    def __call__(self,y_true,y_pred):
        m=y_true.shape[0]
        eps=1e-15
        y_pred=np.clip(y_pred,eps,1-eps)       
        return -np.sum(y_true*np.log(y_pred))/m

    def gradient(self,y_true,y_pred):
        m=y_true.shape[0]
        eps=1e-15
        y_pred=np.clip(y_pred,eps,1-eps)
        return (-y_true/y_pred)/m