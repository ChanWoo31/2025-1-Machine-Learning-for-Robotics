import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x>0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(preds, targets):
    return -np.sum(targets * np.log(preds + 1e-9)) / preds.shape[0]

def accuracy(X, y_true):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    t = softmax(z3)
    y_pred = np.argmax(t, axis=1)
    y_true = np.argmax(y_true, axis=1)
    return np.mean(y_pred == y_true)

def SGD(W, dL_dW, learning_rate=0.01):
    W -= learning_rate * dL_dW
    return W

def Momentum(W, dL_dW, vW, learning_rate=0.05, Mom_coeff = 0.9):
    vW = Mom_coeff * vW - learning_rate * dL_dW
    W = W + vW
    return W, vW

def AdaGrad(W, dL_dW, hW, learning_rate = 0.01, epsilon = 1e-7):
    hW += dL_dW**2
    W -= learning_rate * dL_dW / (np.sqrt(hW) + epsilon)
    return W, hW

def Adam(param, dparam, m, v, beta1=0.9, beta2=0.999, learning_rate = 0.01, timestep = 1, epsilon = 1e-7):
    m = beta1 * m + (1 - beta1) * dparam
    v = beta2 * v + (1 - beta2) * (dparam ** 2)
    m_hat = m / (1 - beta1 ** timestep)
    v_hat = v / (1 - beta2 ** timestep)
    param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten= True, one_hot_label=True)

input_dim = 784
hidden_dim_1 = 50
hidden_dim_2 = 100
output_dim = 10
learning_rate = 0.1
epochs = 10
batch_size = 100

np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim_1) * 0.01
b1 = np.zeros((1, hidden_dim_1))
W2 = np.random.randn(hidden_dim_1, hidden_dim_2) * 0.01
b2 = np.zeros((1, hidden_dim_2))
W3 = np.random.randn(hidden_dim_2, output_dim)
b3 = np.zeros((1, output_dim))

train_size = x_train.shape[0]
iter_per_epoch = train_size // batch_size

loss_list_SGD = []
accuracy_list_SGD = []

loss_list_Momentum = []
accuracy_list_Momentum =  []

loss_list_AdaGrad = []
accuracy_list_AdaGrad = []

loss_list_Adam = []
accuracy_list_Adam = []

# SGD
for epoch in range(epochs):
    perm = np.random.permutation(train_size)

    for i in range(iter_per_epoch):
        batch_mask = perm[i * batch_size:(i+1) * batch_size]
        X_batch = x_train[batch_mask]
        y_batch = t_train[batch_mask]

        # Forward
        z1 = X_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss1 = cross_entropy(t, y_batch)
        loss_list_SGD.append(loss1)

        # Backward
        dL_dz3 = (t - y_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis = 0, keepdims=True)

        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * relu_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * relu_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Update
        W1 = SGD(W1, dL_dW1)
        b1 = SGD(b1, dL_db1)
        W2 = SGD(W2, dL_dW2)
        b2 = SGD(b2, dL_db2)
        W3 = SGD(W3, dL_dW3)
        b3 = SGD(b3, dL_db3)

    acc1 = accuracy(x_test, t_test)
    accuracy_list_SGD.append(acc1)
    print(f"Epoch {epoch + 1}/{epochs}, Loss:{loss1:.4f}, accuracy:{acc1:.2f}")
# Momentum
for epoch in range(epochs):
    perm = np.random.permutation(train_size)

    for i in range(iter_per_epoch):
        batch_mask = perm[i * batch_size:(i+1) * batch_size]
        X_batch = x_train[batch_mask]
        y_batch = t_train[batch_mask]

        # Forward
        z1 = X_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss2 = cross_entropy(t, y_batch)
        loss_list_Momentum.append(loss2)

        # Backward
        dL_dz3 = (t - y_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis = 0, keepdims=True)

        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * relu_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * relu_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Update
        W1, vW1 = Momentum(W1, dL_dW1, vW = 0)
        b1, vb1 = Momentum(b1, dL_db1, vW = 0)
        W2, vW2 = Momentum(W2, dL_dW2, vW = 0)
        b2, vb2 = Momentum(b2, dL_db2, vW = 0)
        W3, vW3 = Momentum(W3, dL_dW3, vW = 0)
        b3, vb3 = Momentum(b3, dL_db3, vW = 0)

    acc2 = accuracy(x_test, t_test)
    accuracy_list_Momentum.append(acc2)
    print(f"Epoch {epoch + 1}/{epochs}, Loss:{loss2:.4f}, accuracy:{acc2:.2f}")
# AdaGrad
for epoch in range(epochs):
    perm = np.random.permutation(train_size)

    for i in range(iter_per_epoch):
        batch_mask = perm[i * batch_size:(i+1) * batch_size]
        X_batch = x_train[batch_mask]
        y_batch = t_train[batch_mask]

        # Forward
        z1 = X_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss3 = cross_entropy(t, y_batch)
        loss_list_AdaGrad.append(loss3)

        # Backward
        dL_dz3 = (t - y_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis = 0, keepdims=True)

        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * relu_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * relu_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Update
        W1, hW1 = AdaGrad(W1, dL_dW1, hW = 0)
        b1, hb1 = AdaGrad(b1, dL_db1, hW = 0)
        W2, hW2 = AdaGrad(W2, dL_dW2, hW = 0)
        b2, hb2 = AdaGrad(b2, dL_db2, hW = 0)
        W3, hW3 = AdaGrad(W3, dL_dW3, hW = 0)
        b3, hb3 = AdaGrad(b3, dL_db3, hW = 0)

    acc3 = accuracy(x_test, t_test)
    accuracy_list_AdaGrad.append(acc3)
    print(f"Epoch {epoch + 1}/{epochs}, Loss:{loss3:.4f}, accuracy:{acc3:.2f}")

for epoch in range(epochs):
    perm = np.random.permutation(train_size)

    for i in range(iter_per_epoch):
        batch_mask = perm[i * batch_size:(i+1) * batch_size]
        X_batch = x_train[batch_mask]
        y_batch = t_train[batch_mask]

        # Forward
        z1 = X_batch @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = relu(z2)
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss4 = cross_entropy(t, y_batch)
        loss_list_Adam.append(loss4)

        # Backward
        dL_dz3 = (t - y_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis = 0, keepdims=True)

        dL_da2 = dL_dz3 @ W3.T
        dL_dz2 = dL_da2 * relu_derivative(z2)
        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dL_dz1 = dL_da1 * relu_derivative(z1)
        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Update
        W1, mW1, vW1 = Adam(W1, dL_dW1, m=0, v=0)
        b1, mb1, vb1 = Adam(b1, dL_db1, m=0, v=0)
        W2, mW2, vW2 = Adam(W2, dL_dW2, m=0, v=0)
        b2, mb2, vb2 = Adam(b2, dL_db2, m=0, v=0)
        W3, mW3, vW3 = Adam(W3, dL_dW3, m=0, v=0)
        b3, mb3, vb3 = Adam(b3, dL_db3, m=0, v=0)

    acc4 = accuracy(x_test, t_test)
    accuracy_list_Adam.append(acc4)
    print(f"Epoch {epoch + 1}/{epochs}, Loss:{loss4:.4f}, accuracy:{acc4:.2f}")

optim_names = ['SGD', 'Momentum', 'AdaGrad', 'Adam']

loss_list = [
    loss_list_SGD,
    loss_list_Momentum,
    loss_list_AdaGrad,
    loss_list_Adam
]

acc_list = [
    accuracy_list_SGD,
    accuracy_list_Momentum,
    accuracy_list_AdaGrad,
    accuracy_list_Adam
]

styles = ['r-', 'g--', 'b-.,', 'm:']

fig, axes = plt.subplots(2,1)
for name, loss_l, style in zip(optim_names, loss_list, styles):
    axes[0].plot(loss_l, style, label = name)
axes[0].set_title('Loss Curve')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)
for name, acc_l, style in zip(optim_names, acc_list, styles):
    axes[1].plot(acc_l, style, label = name)
axes[1].set_title('acc Curve')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('acc')
axes[1].legend()
axes[1].grid(True)
plt.show()



