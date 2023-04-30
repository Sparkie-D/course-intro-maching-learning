import numpy as np
from numpy import random as rd
import time


def plain_distance_function(X):
    # 直观的距离计算实现方法
    # 首先初始化一个空的距离矩阵D
    D = np.zeros((X.shape[0], X.shape[0]))
    # 循环遍历每一个样本对
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            # 计算样本i和样本j的距离
            D[i, j] = np.sqrt(np.sum((X[i] - X[j])**2))
    return D


def distance_function(X):
    D = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            tmp = X[i] - X[j]
            D[i, j] = np.sqrt(np.dot(tmp.T, tmp))
    return D


def distance_function_two(X):
    # ||x-y||_2 = sqrt(x^Tx + y^Ty - 2x^Ty)
    D = np.zeros((X.shape[0], X.shape[0]))
    XXT = np.dot(X, X.T)
    #                      [[x1Tx1, x1Tx2, ... x1Txm]
    #               XXT =   [x2Tx1, x2Tx2, ... x2Txm]
    #                       [...     ...        ... ]
    #                       [xmTx1, xmTx2, ... xmTxm]]
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            D[i, j] = np.sqrt(XXT[i, i] + XXT[j, j] - 2 * XXT[i, j])
    return D

