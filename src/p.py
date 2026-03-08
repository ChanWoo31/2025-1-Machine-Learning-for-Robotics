import numpy as np
from dataset.mnist import load_mnist

import matplotlib.pyplot as plt

epoch_array = []
loss_array = []
acc_array = []

(x_train,t_train), (x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

input_dim = 784
hidden_dim = 50
output_dim = 10
learning_rate = 0.1
epochs = 10
batch_size = 100

np.random.seed(42)
W1 = np.random.randn(input_dim,hidden_dim) * 0.01
b1 = np.zeros((1,hidden_dim))
W2 = np.random.randn(hidden_dim,output_dim) * 0.01
b2 = np.zeros((1,output_dim))

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)

def sigmoid(x):
        return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)

    return s*(1-s)

def softmax(x):
    exp_x = np.exp(x-np.max(x,axis=1,keepdims=True))
    return exp_x / np.sum(exp_x,axis=1,keepdims=True)

def cross_entropy(preds, targets):
     return -np.sum(targets * np.log(preds + 1e-9)) / preds.shape[0]

train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size

for epoch in range(epochs):
    perm = np.random.permutation(train_size)

    for i in range(iter_per_epoch):
        batch_mask = perm[i*batch_size:(i+1)*batch_size]
        X_batch =  x_train[batch_mask]
        y_batch = t_train[batch_mask]

        z1 = X_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 +b2
        t = softmax(z2)
        loss = cross_entropy(t,y_batch)


        dL_dz2 = (t-y_batch) / batch_size
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2,axis=0,keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * relu_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1,axis=0,keepdims=True)

        W1 -= learning_rate * dL_dW1
        b1 -= learning_rate * dL_db1
        W2 -= learning_rate * dL_dW2
        b2 -= learning_rate * dL_db2
    
    print(f"epoch {epoch+1}/{epochs}, Loss : {loss:.4f}")

    epoch_array.append(epoch+1)
    loss_array.append(loss)


def accuracy(X, y_true):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 +b2
    t = softmax(z2)
    y_pred = np.argmax(t,axis=1)
    y_true = np.argmax(y_true,axis=1)

    return np.mean(y_pred == y_true)

acc = accuracy(x_test,t_test)
print(f"Test acc:{acc * 100 :.2f}%")

plt.plot(epoch_array, loss_array)
plt.show()