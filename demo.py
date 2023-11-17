import pandas as pd
import numpy as np
import sympy.vector as sv
from sympy.vector import Del
import math

df = pd.read_csv("googleplaystore.csv")

df = df[['App', 'Rating', 'Reviews', 'Price', 'Genres']]
df = df.loc[df['Genres'].isin(['Strategy'])]
# choose four games for test, among them two are free
df = df.loc[
    df['App'].isin(['Bloons TD 5', 'European War 6: 1804', 'Plants vs. Zombies FREE', 'Clash Royale'])].drop_duplicates(
    subset='App').reset_index(drop=True)
df['Reviews'] = df['Reviews'].astype(float)
df['std_review'] = (df['Reviews'] - df['Reviews'].mean()) / df['Reviews'].std()
df['percent'] = df['Reviews'] / df['Reviews'].sum()


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

        self.num = df['percent'].values
        self.price = np.array([0, 1.99, 2.99, 0.99])
        self.quantity = np.zeros(4)
        self.revenue = np.zeros(4)
        self.utility = np.zeros(4)
        self.coef = np.zeros(3)
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


if __name__ == '__main__':
    Pro = Project(1e-4, 0.001, 0.1)
    print(Pro.gradient_method())
