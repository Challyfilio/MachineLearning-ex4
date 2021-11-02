import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# 前向传播函数
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


def cost(theta1, theta2, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    return J


if __name__ == '__main__':
    data = loadmat('ex4data1.mat')
    # print(data)

    X = data['X']  # (5000, 400)
    y = data['y']  # (5000, 1)
    # print(X.shape, y.shape)

    weight = loadmat('ex4weights.mat')
    theta1, theta2 = weight['Theta1'], weight['Theta2']  # (25, 401) (10, 26)
    # print(theta1.shape, theta2.shape)

    sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
    sample_images = data['X'][sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

    # print(sigmoid_gradient(0)) #0.25

    # np.random.random(size) 返回size大小的0-1随机浮点数
    # params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24
