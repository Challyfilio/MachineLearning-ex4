import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.io import loadmat
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


# 前向传播函数,返回传播结果
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    a1 = np.insert(X, 0, values=np.ones(m), axis=1)  # 插入一列1元素，偏置
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)  #
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


def costReg(theta1, theta2, input_size, hidden_size, num_labels, X, y, learning_rate):
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

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    return J


def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    J = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    return J, delta1, delta2


def backpropReg(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

    J = J / m

    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    # perform backpropagation
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term
    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


if __name__ == '__main__':
    data = loadmat('ex4data1.mat')
    # print(data)

    X = data['X']  # (5000, 400)
    y = data['y']  # (5000, 1)
    # print(X.shape, y.shape)

    weight = loadmat('ex4weights.mat')
    theta1, theta2 = weight['Theta1'], weight['Theta2']  # (25, 401) (10, 26)
    # print(theta1.shape, theta2.shape)
    '''
    sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
    sample_images = data['X'][sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(12, 12))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()
    '''
    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)  # (5000, 10)
    # print(y_onehot.shape)
    # print(y[0], y_onehot[0, :])  # [10] [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]

    input_size = 400
    hidden_size = 25
    num_labels = 10
    learning_rate = 1

    print(cost(theta1, theta2, input_size, hidden_size, num_labels, X, y_onehot, learning_rate))  # 0.2876291651613188

    print(sigmoid_gradient(0))  # 0.25

    # np.random.random(size) 返回size大小的0-1随机浮点数
    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24

    # minimize the objective function
    fmin = minimize(fun=backpropReg, x0=(params),
                    args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                    method='TNC', jac=True, options={'maxiter': 250})
    print(fmin)

    X = np.matrix(X)
    thetafinal1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    thetafinal2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    # 计算使用优化后的θ得出的预测
    a1, z2, a2, z3, h = forward_propagate(X, thetafinal1, thetafinal2)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    print(y_pred)

    print(classification_report(y, y_pred))
