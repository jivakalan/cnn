# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:51:47 2019

@author: jkalan
"""


import numpy as np
from keras.datasets import cifar10

(x_train, y_train),(x_test, y_test) = cifar10.load_data()


##applies sigmoid function to input
def softmax(x):
    expA = np.exp(x)
    s = expA/ expA.sum()
    
    return s


def zero_pad(X):
    x_pad = np.pad(X,((0,0),(1,1),(1,1),(0,0)),mode ='constant',constant_values =(0,0))  
    ##constant values inserts the number specified in the padding. 
    ##the (1,1) represents which dimension you are inserting into and how wide
    print(x_pad.shape)
    
    return x_pad


def initialize_filters():
    ##initalize the weights of the filter 
    f = np.random.randn(1,3,3,3)
    return f



                  
#def conv_single_step(x_slice, f, b):
def conv_single_step(x_pad, l, f, b, h, c, num):
    
    # apply the filter to a single "slice" of the input image
    Z= np.sum(x_pad[num,h:h+3,l:l+3,c] * f)
    #add a bias
    #cast to float so scalar
    Z=Z+float(b)
    
    return Z

def conv_forward(x_pad):
    ##full forward convoluton, iterate over the image, increment by stride then save it in a new array
    Z=np.zeros([1000,32,32,3])
    for num in range(0,1000):
        for c in range(0,3):
            for h in range(0,32): ##moving down
                for l in range(0,32):#moving across
                    Z[num,h,l,c]=conv_single_step(x_pad,l,f,b, h, c, num)
    return Z

def relu(Z):
    ##takes output of conv_forward() and outputs R
    R = np.maximum(Z,0)
    return R

    
def pooling(R):
    ##avg pooling or max pooling...reduces dimensionality of the convolved output...i.e 32x32 becomes 4x4
    ##uses max pooling with 2x2 filter
    
    #initalize the pooling output matrix
    P=np.zeros([1000,16,16,3])
    for num in range(0,1000):
        for c in range(0,3):
            for h in range(0,32,2):
                for l in range(0,32,2):
    
                    h1=int(h/2)
                    l1=int(l/2)
    
                    P[num,h1,l1,c]= np.max(R[num,h:h+2,l:l+2,c])
    return P



x_pad =zero_pad(x_test)
f=initialize_filters()
b=np.random.randn(1,1,1,1)            


#conv layer 1
Z=conv_forward(x_pad)
#relu
R=relu(Z)
#Maxpool 
P=pooling(R)
#flatten
fc1 =P.reshape(1000,768)


W1= np.random.randn( 10, 768 ) *0.01
b1 = np.zeros( ( 10, 1) )

out = np.zeros([1000,10])

# for each image, run through 1 additional NN layer 
for n in range(0,1000):
    A_prev = fc1[n]
    A_prev = A_prev.reshape(1,768)
    Z= np.dot(W1, A_prev.T) + b1
    A = softmax(Z)
    out[n] = A.T


