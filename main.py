# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
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
    print(data)

    X = data['X']
    y = data['y']
    print(X.shape, y.shape)

    print(sigmoid_gradient(0))


    def sigmoid_gradient(z):
        return np.multiply(sigmoid(z), (1 - sigmoid(z)))


    # np.random.random(size) 返回size大小的0-1随机浮点数
    params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.24

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
