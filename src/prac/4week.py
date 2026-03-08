import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

from PIL import Image

import pickle


def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def init_network():
    with open(os.path.dirname(__file__) + "/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def forward(network, X, activation):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = X @ W1 + b1
    z1 = activation(a1)
    a2 = z1 @ W2 + b2
    z2 = activation(a2)
    a3 = z2 @ W3 + b3
    y = softmax(a3)

    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test



# 배치처리 전
# x, t = get_data()
# print(x.shape)
# network = init_network()
# accuracy_cnt = 0
# for i in range(len(x)):
#     y = forward(network, x[i], sigmoid)
#     p = np.argmax(y)
#     if p ==t[i]:
#         accuracy_cnt += 1

# print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


#배치처리 후
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = forward(network, x_batch, sigmoid)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))







