# 3 Layer backpropagation

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
    # Layer1 정규화 + 활성화
    z1 = X @ W1 + b1
    # 추론 모드: 미니배치 mean/var 대신 running_mean/running_var 사용
    z1_norm = (z1 - running_mean1) / np.sqrt(running_var1 + epsilon)
    bn1_out  = gamma1 * z1_norm + beta1
    a1 = relu(bn1_out)

    # Layer2
    z2 = a1 @ W2 + b2
    z2_norm = (z2 - running_mean2) / np.sqrt(running_var2 + epsilon)
    bn2_out  = gamma2 * z2_norm + beta2
    a2 = relu(bn2_out)

    # Output
    z3 = a2 @ W3 + b3
    y  = softmax(z3)

    y_pred = np.argmax(y, axis=1)
    y_true = np.argmax(y_true, axis=1)
    return np.mean(y_pred == y_true)

def cross_entropy_loss(y, t, weight_decay_lambda):
    m = y.shape[0]
    data_loss = -np.sum(t * np.log(y + 1e-7)) / m
    weight_deacy_loss = (weight_decay_lambda / 2) * (np.sum(W1 **2) + np.sum(W2**2) + np.sum(W3**2))
    return data_loss + weight_deacy_loss

def update_parameters(dW1, db1, dW2, db2, dW3, db3, learning_rate, weight_decay_lambda):
    global W1, b1, W2, b2, W3, b3
    W1 -= learning_rate * (dW1 + weight_decay_lambda * W1)
    b1 -= learning_rate * db1
    W2 -= learning_rate * (dW2 + weight_decay_lambda * W2)
    b2 -= learning_rate * db2
    W3 -= learning_rate * (dW3 + weight_decay_lambda * W3)
    b3 -= learning_rate * db3


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

loss_list = []
accuracy_list = []

# 배치 정규화 파라미터
gamma1 = np.ones((1, hidden_dim_1))
gamma2 = np.ones((1, hidden_dim_2))
beta1 = np.zeros((1, hidden_dim_1))
beta2 = np.zeros((1, hidden_dim_2))
running_mean1 = np.zeros((1, hidden_dim_1))
running_mean2 = np.zeros((1, hidden_dim_2))
running_var1 = np.ones((1, hidden_dim_1))
running_var2 = np.ones((1, hidden_dim_2))
epsilon = 1e-7

for epoch in range(epochs):
    perm = np.random.permutation(train_size)

    for i in range(iter_per_epoch):
        batch_mask = perm[i * batch_size:(i+1) * batch_size]
        X_batch = x_train[batch_mask]
        y_batch = t_train[batch_mask]

        # Forward
        z1 = X_batch @ W1 + b1
        
        batch_mean1 = np.mean(z1, axis=0, keepdims=True)
        batch_var1 = np.var(z1, axis=0, keepdims=True)
        
        running_mean1 = 0.9 * running_mean1 + 0.1 * batch_mean1
        running_var1 = 0.9 * running_var1 + 0.1 * batch_var1

        z1_norm = (z1 - batch_mean1) / np.sqrt(batch_var1 + epsilon)
        bn_output1 = gamma1 * z1_norm + beta1

        a1 = relu(bn_output1)
        z2 = a1 @ W2 + b2

        batch_mean2 = np.mean(z2, axis=0, keepdims=True)
        batch_var2 = np.var(z2, axis=0, keepdims=True)

        running_mean2 = 0.9 * running_mean2 + 0.1 * batch_mean2
        running_var2 = 0.9 * running_var2 + 0.1 * batch_var2
        z2_norm = (z2 -batch_mean2) / np.sqrt(batch_var2 + epsilon)
        bn_output2 = gamma2 * z2_norm + beta2

        a2 = relu(bn_output2)
        z3 = a2 @ W3 + b3
        t = softmax(z3)
        loss = cross_entropy_loss(t, y_batch, 0)
        loss_list.append(loss)

        # Backward
        dL_dz3 = (t - y_batch) / batch_size
        dL_dW3 = a2.T @ dL_dz3
        dL_db3 = np.sum(dL_dz3, axis = 0, keepdims=True)

        dL_da2 = dL_dz3 @ W3.T
        dbn2 = dL_da2 * relu_derivative(bn_output2)

        dgamma2 = np.sum(dbn2 * z2_norm, axis=0, keepdims=True)
        dbeta2 = np.sum(dbn2, axis=0, keepdims=True)
        
        dz2_norm = dbn2 * gamma2

        dbatch_var2 = np.sum(dz2_norm * (z2 -batch_mean2) * -0.5 * (batch_var2 + epsilon) ** (-1.5), axis=0, keepdims=True)
        dbatch_mean2 = np.sum(dz2_norm * -1 / np.sqrt(batch_var2 + epsilon), axis=0, keepdims=True)

        dL_dz2 = dz2_norm / np.sqrt(batch_var2 + epsilon) + dbatch_var2 * 2 * (z2 - batch_mean2) / batch_size + dbatch_mean2 / batch_size
    

        dL_dW2 = a1.T @ dL_dz2
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = dL_dz2 @ W2.T
        dbn1 = dL_da1 * relu_derivative(bn_output1)
        dgamma1 = np.sum(dbn1 * z1_norm, axis=0, keepdims=True)
        dbeta1 = np.sum(dbn1, axis=0, keepdims=True)
        
        dz1_norm = dbn1 * gamma1

        dbatch_var1 = np.sum(dz1_norm * (z1 - batch_mean1) * -0.5 * (batch_var1 + epsilon) ** (-1.5), axis=0, keepdims=True)
        dbatch_mean1 = np.sum(dz1_norm * -1 / np.sqrt(batch_var1 + epsilon), axis=0, keepdims=True)


        dL_dz1 = dz1_norm / np.sqrt(batch_var1 + epsilon) + dbatch_var1 * 2 * (z1 - batch_mean1) / batch_size + dbatch_mean1 / batch_size 

        dL_dW1 = X_batch.T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Update
        W1 -= learning_rate * dL_dW1
        b1 -= learning_rate * dL_db1
        W2 -= learning_rate * dL_dW2
        b2 -= learning_rate * dL_db2
        W3 -= learning_rate * dL_dW3
        b3 -= learning_rate * dL_db3
        gamma1 -= learning_rate * dgamma1
        beta1  -= learning_rate * dbeta1
        gamma2 -= learning_rate * dgamma2
        beta2  -= learning_rate * dbeta2


    acc = accuracy(x_test, t_test)
    accuracy_list.append(acc)
    print(f"Epoch {epoch + 1}/{epochs}, Loss:{loss:.4f}, accuracy:{acc:.2f}")


fig, axes = plt.subplots(2,1)
axes[0].plot(loss_list, 'r-')
axes[1].plot(accuracy_list,'o-')
plt.show()



