import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")

X=data.iloc[:,3:4].values
y=data.iloc[:,4:5].values


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

layer1=1
layer2=4
layer3=1

w1=np.random.rand(layer1,layer2)
w2=np.random.rand(layer2,layer3)
b1=np.zeros((1,layer2))
b2=np.zeros((1,layer3))

epochs=1000
i=epochs
 
from utilities import forward_pass,relu,sigmoid,cross_entropy,sigmoid_backprop,relu_backprop

#TRAINING

lr=1
while(i>0):
    z1=forward_pass(X_train,w1,b1)
    a1=relu(z1)
    z2=forward_pass(a1,w2,b2)
    out=sigmoid(z2)
    loss=cross_entropy(y_train,out)
    dw2,db2=sigmoid_backprop(out,y_train,a1)
    w2=w2-lr*dw2
    b2=b2-lr*db2
    dw1,db1=relu_backprop(db2,w2,z1,X_train)
    w1=w1-lr*dw1
    b1=b1-lr*db1
    i-=1


#TESTING
    
z1=forward_pass(X_test,w1,b1)
a1=relu(z1)
z2=forward_pass(a1,w2,b2)
out=sigmoid(z2)
loss=cross_entropy(y_test,out)
out=np.round(out)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,out)

print((cm[0][0]+cm[1][1])/len(X_test))



    
   