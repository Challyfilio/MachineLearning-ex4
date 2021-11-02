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


if __name__ == '__main__':
    data = loadmat('ex4data1.mat')
    # print(data)

    X = data['X']  # (5000, 400)
    y = data['y']  # (5000, 1)
    # print(X.shape, y.shape)

    weight = loadmat('ex4weights.mat')
    theta1, theta2 = weight['Theta1'], weight['Theta2']  # (25, 401) (10, 26)
    # print(theta1.shape, theta2.shape)

    # print(sigmoid_gradient(0)) #0.25

    # np.random.random(size) 返回size大小的0-1随机浮点数
    # params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24
