import torch
import numpy as np
import pandas as pd
import math
import torch 

from data import df1

product_name = torch.tensor(np.array(df1.product_name))
manufacturer = torch.tensor(np.array(df1.manufacturer))
price = torch.tensor(np.array(df1.price),requires_grad=True)
reviews = torch.tensor(np.array(df1.number_of_reviews))
category = torch.tensor(np.array(df1.amazon_category))
percent = torch.tensor(np.array(df1.percent))

df2 = df1

def numerical_gradient(f, x):
    # 数值微分求梯度,f为函数，x为NumPy数组，该函数对数组x的各个元素求数值微分

    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和x形状相同的数组

    for index in range(x.size):
        # 计算第index个x的偏导数
        temp_val = x[index]
        x[index] = temp_val + h
        fxh1 = f(x)
        x[index] = temp_val - h
        fxh2 = f(x)

        # 将偏导数存为梯度向量
        grad[index] = (fxh1 - fxh2) / (2 * h)
        x[index] = temp_val

    return grad


def bls(fun, x, grad, step, d_s):
    while True:
        if fun(x + step * grad) >= fun(x) + step * np.linalg.norm(grad) / 2:
            break
        step = d_s * step
    return step


class Project:
    def __init__(self, bias, step, d_r):
        self.data = df2
        self.utility = torch.tensor(np.zeros(df2.shape[0]))
        self.cons = torch.tensor(np.zeros(df2.shape[0]))
        self.vec = torch.tensor(np.zeros(df2.shape[0]))
        self.para = torch.tensor(np.zeros(df2.shape[0]))
        self.quantity = torch.tensor(np.zeros(df2.shape[0]))
        # coef[1]: v_i, coef[2]: (1-sigma_i), coef[3]: (alpha_i)
        
        self.bias = bias
        # decreasing rate
        self.stepsize = step
        self.d_r = d_r
        self.delta = None

    def log_maximum(self, coef):
        log = 0
        for i in range(0, 4):
            self.utility[i] = self.coef[0] + self.price[i] + (1 - self.coef[1]) * self.coef[2] * self.quantity[i] + \
                              self.coef[2] * (self.quantity.sum() - self.quantity[i])
        for i in range(0, 4):
            self.quantity[i] = math.exp(self.utility[i]) / (1 + np.exp(self.utility).sum())
            log += self.num[i] * math.log(self.quantity[i])
        q_0 = 1 - self.quantity.sum()
        n_0 = (q_0 * self.num.sum()) / (1 - q_0)
        log += n_0 * q_0
        return log

    def gradient_method(self):
        t = 0
        while True:
            self.delta = numerical_gradient(self.log_maximum, self.coef)
            self.stepsize = bls(self.log_maximum, self.coef, self.delta, self.stepsize, self.d_r)
            self.coef = self.coef + self.delta * self.stepsize
            if np.linalg.norm(self.delta) < self.bias:
                return self.coef
                break
            t += 1