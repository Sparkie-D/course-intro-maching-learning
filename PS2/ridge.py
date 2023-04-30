from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import tkinter as tk
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

X, y = load_boston(return_X_y=True)


trainx, testx, trainy, testy = train_test_split(X, y, test_size=0.33, random_state=42)


# linear regression
def linReg(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    ones = np.ones(X_train.shape[0]) #生成一个全1的向量
    X_top = np.insert(X_train,X_train.shape[1]-1, values=ones, axis=1) #在X中加入最后一列全1
    tmp = np.dot(np.dot(np.linalg.inv((np.dot(X_top.T, X_top))), X_top.T), y_train)
    # print(tmp)
    return tmp #(X^TX)^{-1}X^Ty


def linRegMSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    w_hat = linReg(X_train, y_train)
    ones = np.ones(X_test.shape[0])  # 生成一个全1的向量
    X_test_top = np.insert(X_test, X_test.shape[1] - 1, values=ones, axis=1)  # 在X中加入最后一列全1
    y_pre = np.dot(X_test_top, w_hat)
    delta = y_pre - y_test
    return np.dot(delta.T, delta)/y_test.shape[0]  # 1/n * (y'-y)^T(y'-y)


reportLinRegMSE = lambda: linRegMSE(trainx, trainy, testx, testy)

print(reportLinRegMSE())

# ridge regression

#def ridgeReg(X_train: np.ndarray, y_train: np.ndarray, lmbd: float) -> np.ndarray:
#    m = X_train.shape[0]
#    d = X_train.shape[1]
#    XTX = np.dot(X_train.T, X_train) + 2*lmbd * np.eye(d)
#    XTone = np.dot(X_train.T, np.ones(m))
#    XToneoneTX = (1.0 / m) * np.dot(XTone, XTone.T)
#    inner = np.eye(m) - (1.0 / m) * np.ones((m, m))
#    XTandY = np.dot(np.dot(X_train.T, inner), y_train)
#    w = np.dot(np.linalg.inv(XTX - XToneoneTX), XTandY)
#    b = -(1.0 / m) * (np.dot(np.ones(m), y_train) - np.dot(np.ones(m), np.dot(X_train, w)))
#    res = np.r_[w, b]
#    print(res)
#    return res # (w; b)

def ridgeReg(X_train: np.ndarray, y_train: np.ndarray, lmbd: float) -> np.ndarray:
    m = X_train.shape[0]
    d = X_train.shape[1]
    XTone = np.dot(X_train.T, np.ones(m))
    tmp0 = np.dot(X_train.T, X_train) - (1.0 / m) * np.dot(XTone, XTone.T) + 2 * lmbd * np.eye(d)
    tmp1 = X_train.T - (1.0 / m) * np.dot(X_train.T, np.ones((m, m)))
    w = np.dot(np.linalg.inv(tmp0), np.dot(tmp1, y_train))
    b = (1.0 / m) * (np.dot(np.ones(m), y_train) - np.dot(np.ones(m), np.dot(X_train, w)))
    res = np.r_[w, b]
    # print(res)
    return res


def ridgeTestRegMSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, lmbd: float) -> float:
    m = X_test.shape[0]
    w_hat = ridgeReg(X_train, y_train, lmbd)
    b = w_hat[-1]
    w = np.delete(w_hat, -1)
    y_pre = np.dot(X_test, w) + b * np.ones(m)
    delta = y_pre - y_test
    return np.dot(delta.T, delta)/y_test.shape[0]  # 1/n * (y'-y)^T(y'-y)

def ridgeTrainRegMSE(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, lmbd: float) -> float:
    m = X_train.shape[0]
    w_hat = ridgeReg(X_train, y_train, lmbd)
    b = w_hat[-1]
    w = np.delete(w_hat, -1)
    y_pre = np.dot(X_train, w) + b * np.ones(m)
    delta = y_pre - y_train
    return np.dot(delta.T, delta)/y_train.shape[0]  # 1/n * (y'-y)^T(y'-y)

reportRidgetstRegMSE = lambda lmbd: ridgeTestRegMSE(trainx, trainy, testx, testy, lmbd)

reportRidgetrnRegMSE = lambda lmbd: ridgeTrainRegMSE(trainx, trainy, testx, testy, lmbd)

print(reportRidgetrnRegMSE(0))
print(reportRidgetstRegMSE(0))

# 绘图描述岭回归随着lambda变化的军方误差

xplot = np.arange(0, 10000, 1)
ylinplot = reportLinRegMSE() * np.ones(xplot.size)
yrigplot = []
ytrnplot = []
for t in xplot:
    tmp = reportRidgetstRegMSE(t)
    yrigplot.append(tmp)
    tmp = reportRidgetrnRegMSE(t)
    ytrnplot.append(tmp)
plt.plot(xplot, ylinplot, label = 'linear regression')
plt.plot(xplot, yrigplot, label = 'ridge regression, test  cases')
plt.plot(xplot, ytrnplot, label = 'ridge regression, train cases')
plt.title("MSE in different regression methods")
plt.xlabel('lambda')
plt.ylabel('MSE')
plt.legend()
plt.show()
