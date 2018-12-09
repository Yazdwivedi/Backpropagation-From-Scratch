import numpy as np
import pandas as pd

def forward_pass(x,w,b):
    val=x.dot(w)+b
    return val

def relu(x):
    x[x<=0]=0
    return x
    

def sigmoid(x):
    return 1/(1+np.exp(-x))

def cross_entropy(y_true,y_pred):
    loss1=-y_true.T.dot(np.log(y_pred))
    loss2=-(1-y_true).T.dot(np.log(1-y_pred))
    loss=(loss1+loss2)/(len(y_true))
    return loss

def sigmoid_backprop(x,a2,a1):
    val=x-a2
    return a1.T.dot(val)/len(x),np.sum(val,axis=0,keepdims=True)/len((x))

def relu_derivative(x):
    x[x<=0]=0
    x[x>0]=1
    return x

def relu_backprop(prev,w,z,x):
    z=relu_derivative(z)
    val=prev.dot(w.T)*z
    return (x.T.dot(val))/len(x),np.sum(val,axis=0,keepdims=True)/len(x)