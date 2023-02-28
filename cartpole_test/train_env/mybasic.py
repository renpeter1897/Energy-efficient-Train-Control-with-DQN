import matplotlib.pyplot as plt
import numpy as np
from tool import findIndex


class train:
    def __init__(self):
        self.Fmax = 238800  # 牛
        self.Bmax = 234000
        self.FPmax = 7346000  # 瓦
        self.BPmax = 10400000
        self.b1 = 8.63
        self.b2 = 0.07295
        self.b3 = 0.00112
        self.M = 420  # 吨
        self.roh = 0.06

    def get_tracForce(self, v):  # m/s —— N
        if self.Fmax * v < self.FPmax:
            return self.Fmax
        else:
            return self.FPmax / v

    def get_brakeForce(self, v):  # m/s —— N
        if self.Bmax * v < self.BPmax:
            return self.Bmax
        else:
            return self.BPmax / v

    def get_resist(self, v):  # km/h —— N/t
        w = self.b1 + self.b2 * v + self.b3 * v * v
        return w


class railway:
    def __init__(self):
        self.slpos = [0, 5000, 53000, 59000]
        self.slval = [180, 300, 180, 0]
        self.D = self.slpos[-1] - self.slpos[0]
        self.gradpos = [0, 1500, 3000, 4500]
        self.gradval = [0, 0, 0, 0]

    def get_grad(self, pos):  # m —— N/kN
        grad = 0
        for i in range(0, len(self.gradpos)):
            if pos < self.gradpos[i]:
                grad = self.gradval[i - 1]
                break
        return grad

    def get_SL(self, pos):  # m —— m/s
        i = findIndex(self.slpos, pos)
        Vlimit = self.slval[i]
        return Vlimit


def FV_curve():
    V = []
    F = []
    Fmax = train().Fmax
    FPmax = train().FPmax
    for v in range(83):
        V.append(v)
        if Fmax * v <= FPmax:
            F.append(Fmax)
        else:
            F.append(FPmax / v)
    print(V)
    print(F)
    plt.plot(V, F)
    plt.show()


def BV_curve():
    V = []
    B = []
    Bmax = train().Bmax
    BPmax = train().BPmax
    for v in range(100):
        V.append(v)
        if Bmax * v <= BPmax:
            B.append(Bmax)
        else:
            B.append(BPmax / v)
    plt.plot(V, B)
    plt.show()


def SL_Grad_curve(slpos, slval, gradpos, gradval):
    n = len(slpos)
    V = np.zeros(2 * n - 2)
    P = np.zeros(2 * n - 2)
    P[0] = slpos[0]
    V[0] = slval[0]
    for i in range(1, n - 1):
        P[2 * i - 1] = slpos[i]
        V[2 * i - 1] = slval[i - 1]
        P[2 * i] = slpos[i]
        V[2 * i] = slval[i]
    P[2 * n - 3] = slpos[n - 1]
    V[2 * n - 3] = slval[n - 2]
    plt.plot(P, V)
    n = len(gradpos)
    x = np.zeros(2 * n)
    y = np.zeros(2 * n)
    x[0] = gradpos[0]
    y[0] = gradval[0]
    # 中间的点都重复两次
    for i in range(1, n):
        x[2 * i - 1] = gradpos[i]
        y[2 * i - 1] = gradval[i - 1]
        x[2 * i] = gradpos[i]
        y[2 * i] = gradval[i]
    x[2 * n - 1] = slpos[-1]
    y[2 * n - 1] = gradval[n - 1]
    plt.plot(x, y)