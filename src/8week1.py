import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataset.mnist import load_mnist



# ===== 변수 선언 =====

input_dim = 784
hidden_dim_1 = 50
hidden_dim_2 = 100
output_dim = 10

learning_rate = 0.05
epochs = 10
batch_size = 100


np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim_1) * 0.01
B1 = np.zeros((1, hidden_dim_1))
W2 = np.random.randn(hidden_dim_1, hidden_dim_2) * 0.01
B2 = np.zeros((1, hidden_dim_2))
W3 = np.random.randn(hidden_dim_2, output_dim) * 0.01
B3 = np.zeros((1, output_dim))

W1_SI = np.random.randn(input_dim, hidden_dim_1) * 0.01
B1_SI = np.zeros((1, hidden_dim_1))
W2_SI = np.random.randn(hidden_dim_1, hidden_dim_2) * 0.01
B2_SI = np.zeros((1, hidden_dim_2))
W3_SI = np.random.randn(hidden_dim_2, output_dim) * 0.01
B3_SI = np.zeros((1, output_dim))

# ===== 함수 선언 =====

def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis = 1, keepdims=True))
    return exp_x / np.sum(exp_x, axis = 1, keepdims=True)

def cross_entropy(x, t):
    return -np.sum(t * np.log(x + 1e-9)) / x.shape[0]

def accuracy_ReLU(x, y_true):
    z1 = x @ W1 + B1
    a1 = ReLU(z1)
    z2 = a1 @ W2 + B2
    a2 = ReLU(z2)
    z3 = a2 @ W3 + B3
    t = softmax(z3)
    y_pred = np.argmax(t, axis=1)
    y_true = np.argmax(y_true, axis=1)
    
    return np.mean(y_pred == y_true)

def accuracy_Sigmoid(x, y_true):
    z1 = x @ W1_SI + B1_SI
    a1 = sigmoid(z1)
    z2 = a1 @ W2_SI + B2_SI
    a2 = sigmoid(z2)
    z3 = a2 @ W3_SI + B3_SI
    t = softmax(z3)
    y_pred = np.argmax(t, axis=1)
    y_true = np.argmax(y_true, axis=1)
    
    return np.mean(y_pred == y_true)

def SGD(W, dW, learning_rate=0.2):
    W -= learning_rate * dW
    return W

def momentum(W, dW, learning_rate=0.1, alpha=0.9, vW = 0):
    vW = alpha * vW - learning_rate * dW
    W += vW
    return W

def AdaGrad(W, dW, hW, learning_rate=0.1, epsilon = 1e-7):
    hW += dW ** 2
    W -= learning_rate * dW / (np.sqrt(hW) + epsilon)
    return W, hW

def adam(param, dparam, m, v, learning_rate=0.01, beta1=0.9, beta2 = 0.999, epsilon = 1e-7, timestep = 1):
    m = beta1 * m + (1 - beta1) * dparam
    v = beta2 * v + (1 - beta2) * (dparam ** 2)
    m_hat = m / (1 - beta1 ** timestep)
    v_hat = v / (1 - beta2 ** timestep)
    param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

mW1, mW2, mW3 = 0, 0, 0
vW1, vW2, vW3 = 0, 0, 0
mb1, mb2, mb3 = 0, 0, 0
vb1, vb2, vb3 = 0, 0, 0
hW1, hW2, hW3 = 0, 0, 0
hb1, hb2, hb3 = 0, 0, 0

params = [W1, B1, W2, B2, W3, B3]
# mnist 호출

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 학습

acc_list_SGD = []
loss_list_SGD = []

acc_list_momentum = []
loss_list_momentum = []

acc_list_AdaGrad = []
loss_list_AdaGrad = []

acc_list_adam = []
loss_list_adam = []



train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size

# SGD
for epoch in range (epochs):
    perm = np.random.permutation(train_size)
    
    for i in range(iter_per_epoch):
        batch_mask = perm[i * batch_size : (i + 1) * batch_size]
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # Forward
        z1 = x_batch @ W1 + B1
        a1 = ReLU(z1)
        z2 = a1 @ W2 + B2
        a2 = ReLU(z2)
        z3 = a2 @ W3 + B3
        y = softmax(z3)
        loss = cross_entropy(y, t_batch)
        
        # Backward
        dL_dz3 = (y - t_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)
        
        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * ReLU_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * ReLU_derivative(z1)
        dL_dW1 = x_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis = 0, keepdims=True)
        
        # W1 -= learning_rate * dL_dW1
        # B1 -= learning_rate * dL_db1
        # W2 -= learning_rate * dL_dW2
        # B2 -= learning_rate * dL_db2
        # W3 -= learning_rate * dL_dW3
        # B3 -= learning_rate * dL_db3
        W1 = SGD(W1, dL_dW1, learning_rate)
        b1 = SGD(B1, dL_db1, learning_rate)
        W2 = SGD(W2, dL_dW2, learning_rate)
        b2 = SGD(B2, dL_db2, learning_rate)
        W3 = SGD(W3, dL_dW3, learning_rate)
        b3 = SGD(B3, dL_db3, learning_rate)
        
    print(f"SGD Epoch : {epoch + 1}/{epochs}, Loss : {loss:.4f}")
    
    acc = accuracy_ReLU(x_test, t_test)
    acc_list_SGD.append(acc)
    
    loss_list_SGD.append(loss)

print(" ======== ")

# momentum
for epoch in range (epochs):
    perm = np.random.permutation(train_size)
    
    for i in range(iter_per_epoch):
        batch_mask = perm[i * batch_size : (i + 1) * batch_size]
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # Forward
        z1 = x_batch @ W1 + B1
        a1 = ReLU(z1)
        z2 = a1 @ W2 + B2
        a2 = ReLU(z2)
        z3 = a2 @ W3 + B3
        y = softmax(z3)
        loss_momentum = cross_entropy(y, t_batch)
        
        # Backward
        dL_dz3 = (y - t_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)
        
        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * ReLU_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * ReLU_derivative(z1)
        dL_dW1 = x_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis = 0, keepdims=True)
        
        W1 = momentum(W1, dL_dW1, learning_rate, alpha=0.9)
        b1 = momentum(B1, dL_db1, learning_rate, alpha=0.9)
        W2 = momentum(W2, dL_dW2, learning_rate, alpha=0.9)
        b2 = momentum(B2, dL_db2, learning_rate, alpha=0.9)
        W3 = momentum(W3, dL_dW3, learning_rate, alpha=0.9)
        b3 = momentum(B3, dL_db3, learning_rate, alpha=0.9)
        
    print(f"momentum Epoch : {epoch + 1}/{epochs}, Loss : {loss_momentum:.4f}")
    
    acc_momentum = accuracy_ReLU(x_test, t_test)
    acc_list_momentum.append(acc_momentum)
    
    loss_list_momentum.append(loss_momentum)

print(" ======== ")

# adagrad
for epoch in range (epochs):
    perm = np.random.permutation(train_size)
    
    for i in range(iter_per_epoch):
        batch_mask = perm[i * batch_size : (i + 1) * batch_size]
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # Forward
        z1 = x_batch @ W1 + B1
        a1 = ReLU(z1)
        z2 = a1 @ W2 + B2
        a2 = ReLU(z2)
        z3 = a2 @ W3 + B3
        y = softmax(z3)
        loss_adagrad = cross_entropy(y, t_batch)
        
        # Backward
        dL_dz3 = (y - t_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)
        
        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * ReLU_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * ReLU_derivative(z1)
        dL_dW1 = x_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis = 0, keepdims=True)
        
        
        W1, hW1 = AdaGrad(W1, dL_dW1, hW1, learning_rate)
        b1, hb1 = AdaGrad(B1, dL_db1, hb1, learning_rate)
        W2, hW2 = AdaGrad(W2, dL_dW2, hW2, learning_rate)
        b2, hb2 = AdaGrad(B2, dL_db2, hb2, learning_rate)
        W3, hW3 = AdaGrad(W3, dL_dW3, hW3, learning_rate)
        b3, hb3 = AdaGrad(B3, dL_db3, hb3, learning_rate)

        
    print(f"adagrad Epoch : {epoch + 1}/{epochs}, Loss : {loss_adagrad:.4f}")
    
    acc_adagrad = accuracy_ReLU(x_test, t_test)
    acc_list_AdaGrad.append(acc_adagrad)
    
    loss_list_AdaGrad.append(loss_adagrad)

print(" ======== ")

# adam
for epoch in range (epochs):
    perm = np.random.permutation(train_size)
    
    for i in range(iter_per_epoch):
        batch_mask = perm[i * batch_size : (i + 1) * batch_size]
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        # Forward
        z1 = x_batch @ W1 + B1
        a1 = ReLU(z1)
        z2 = a1 @ W2 + B2
        a2 = ReLU(z2)
        z3 = a2 @ W3 + B3
        y = softmax(z3)
        loss_adam = cross_entropy(y, t_batch)
        
        # Backward
        dL_dz3 = (y - t_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)
        
        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * ReLU_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * ReLU_derivative(z1)
        dL_dW1 = x_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis = 0, keepdims=True)
        
        
        W1, mW1, vW1 = adam(W1, dL_dW1, mW1, vW1)
        b1, mb1, vb1 = adam(B1, dL_db1, mb1, vb1)
        W2, mW2, vW2 = adam(W2, dL_dW2, mW2, vW2)
        b2, mb2, vb2 = adam(B2, dL_db2, mb2, vb2)
        W3, mW3, vW3 = adam(W3, dL_dW3, mW3, vW3)
        b3, mb3, vb3 = adam(B3, dL_db3, mb3, vb3)


        
    print(f"adam Epoch : {epoch + 1}/{epochs}, Loss : {loss_adam:.4f}")
    
    acc_adam = accuracy_ReLU(x_test, t_test)
    acc_list_adam.append(acc_adam)
    
    loss_list_adam.append(loss_adam)
print(" ======== ")
# acc_ReLU = accuracy_ReLU(x_test, t_test)



x = np.linspace(0, epochs, epochs)

print(1)

# SGD
plt.figure()

plt.plot(x, acc_list_SGD, label='accuracy', color = 'blue')
plt.plot(x, loss_list_SGD, label='loss', color = 'blue')


# momentun
plt.plot(x, acc_list_momentum, label='accuracy', color = 'black')
plt.plot(x, loss_list_momentum, label='loss', color = 'black')

# adagrad
plt.plot(x, acc_list_AdaGrad, label='accuracy', color = 'red')
plt.plot(x, loss_list_AdaGrad, label='loss', color = 'red')


# adam
plt.plot(x, acc_list_adam, label='accuracy', color = 'cyan')
plt.plot(x, loss_list_adam, label='loss', color = 'cyan')

plt.xlabel('epoch')
plt.ylabel('accuracy & loss')
plt.legend()

# plt.figure()
# plt.plot(x, acc_list_Sigmoid, label='accuracy')
# plt.plot(x, loss_list_Sigmoid, label='loss')

# plt.xlabel('epoch')
# plt.ylabel('accuracy & loss')
# plt.legend()

plt.show()