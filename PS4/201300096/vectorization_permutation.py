import numpy as np
import pylab as p
from numpy import random as rd
import time

def plain_permutation_function(X, p):
    # 初始化结果矩阵, 其中每一行对应一个样本
    permuted_X = np.zeros_like(X)
    for i in range(X.shape[0]):
        # 采用循环的方式对每一个样本进行重排列
        permuted_X[i] = X[p[i]]
    return permuted_X

def permutation_function(X, p):
    trans = np.zeros_like(X)
    for i in range(X.shape[0]):
        trans[i][p[i]] = 1
    return np.dot(trans, X)


def permutation_function_two(X, p):
    return X[p, :]

