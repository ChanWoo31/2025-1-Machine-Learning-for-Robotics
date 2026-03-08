import numpy as np
import matplotlib.pyplot as plt
import os

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def init_network():
    network= {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['W2'] = np.array([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5]])
    network['W3'] = np.array([[0.1, 0.2], [0.3, 0.4]])
    network['b1'] = np.array([0.3, 0.2, 0.1])
    network['b2'] = np.array([0.1, 0.2])
    network['b3'] = np.array([0.5, 0.1])
    return network

def forward(network, X, activation):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = X @ W1 + b1
    z1 = activation(a1)
    a2 = z1 @ W2 + b2
    z2 = activation(a2)
    a3 = z2 @ W3 + b3
    y = a3

    return y


X = np.array([1, 0.5])

A = forward(init_network(), X, sigmoid)
B = forward(init_network(), X, relu)

print(A)
print(B)





