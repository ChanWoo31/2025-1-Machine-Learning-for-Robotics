import numpy as np

x = np.array([1.0, 0.5])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def init_network():
    network = {}
    network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.3, 0.2, 0.1])
    network['w2'] = np.array([[0.1, 0.3], [0.2, 0.4], [0.3, 0.5]])
    network['b2'] = np.array([0.1, 0.2])
    network['w3'] = np.array([[0.1, 0.2], [0.3, 0.4]])
    network['b3'] = np.array([0.5, 0.1])

    return network

def forward(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

# 시그모이드드
    # a1 = np.dot(x, w1) + b1
    # z1 = sigmoid(a1)
    # a2 = np.dot(z1, w2) + b2
    # z2 = sigmoid(a2)
    # a3 = np.dot(z2, w3) + b3

# 렐루
    a1 = np.dot(x, w1) + b1
    z1 = relu(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = relu(a2)
    a3 = np.dot(z2, w3) + b3

    y = a3

    return y

y = forward(init_network(), x)

print(y)


