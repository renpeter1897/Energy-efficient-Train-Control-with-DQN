from mybasic import train
from mybasic import railway
from mybasic import SL_Grad_curve
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class run:
    def __init__(self, pos, ds, v, t):
        self.train = train()
        self.railway = railway()
        self.M = self.train.M  # 吨
        self.ds = ds
        self.pos = pos
        self.v = v
        self.t = t
        # self.w0 = self.train.get_resist(self.v * 3.6) * self.M  # 牛
        # self.w1 = self.railway.get_grad(self.pos) * self.M * 9.8  # 牛

    def refresh(self, F, B, ds, mode='normal'):
        self.w0 = self.train.get_resist(self.v * 3.6) * self.M  # 牛  # 获得该速度下产生的阻力
        self.w1 = self.railway.get_grad(self.pos) * self.M * 9.8  # 牛  # 获得该位置产生的阻力
        if mode == 'normal':
            Ftol = F - B - self.w0 - self.w1
            pos = self.pos + ds
        elif mode == 'curve':  # 画最短时间曲线时才开启该模式
            Ftol = B + self.w0 + self.w1
            pos = self.pos - ds
        acc = Ftol / (self.M * 1000)  # m/s^2

        if 2 * acc * ds + self.v ** 2 > 0:  # vt^2-v0^2=2as 判断更新后速度是否会小于0
            v = math.sqrt(2 * acc * ds + self.v ** 2)  # m/s
            t = self.t + ds / ((v + self.v) / 2)  # s
        else:
            v = 0
            pos_cha = -self.v ** 2 / (2 * acc)  # 当速度为0时，位移不再是100m，要重新计算
            pos = self.pos + pos_cha
            t = self.t + pos_cha / ((v + self.v) / 2)

        self.pos = pos
        self.v = v
        self.t = t
        return self.v, self.pos, self.t


def final_state(serror):
    M = train().M
    B = train().Bmax
    w0 = train().get_resist(0) * M
    w1 = railway().get_grad(59000) * M * 9.8
    Ftol = B + w0 + w1
    acc = Ftol / (M * 1000)
    v0 = math.sqrt(2 * acc * serror)
    t0 = serror / (v0 / 2)
    return v0, t0


#  计算最短时间曲线
def MintimeCurve(seroor=1, ds=200):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    v0, t0 = final_state(seroor)
    slpos = railway().slpos
    slval = railway().slval
    gradpos = railway().gradpos
    gradval = railway().gradval
    FV, FPos = [0], [0]
    BV, BPos = [v0], [59000]
    V, T = [0], [0]

    for i in range(1, len(slpos)):
        start = slpos[i - 1]
        end = slpos[i]
        pos = start
        t = 0
        if i == 1:
            v = 0
        else:
            v = slval[i - 2] / 3.6
        state = run(pos, ds, v, t)
        while True:
            F = train().get_tracForce(v)
            v, pos, t = state.refresh(F, 0, ds)
            if v >= slval[i - 1] / 3.6:
                for j in range((end - pos) // ds + 1):
                    FV.append(slval[i - 1])
                    FPos.append(pos)
                    pos += ds
            else:
                FV.append(v * 3.6)
                FPos.append(pos)
            if pos >= end:
                break

    for i in range(len(slpos) - 1, 0, -1):
        start = slpos[i - 1]
        end = slpos[i]
        pos = end + 1
        if i == len(slpos) - 1:
            v = v0
        else:
            v = slval[i] / 3.6
        state = run(pos, ds, v, 0)
        while True:
            B = train().get_brakeForce(v)
            v, pos, t = state.refresh(0, B, ds, mode='curve')
            if v >= slval[i - 1] / 3.6:
                for j in range((pos - start) // ds + 1):
                    BV.append(slval[i - 1])
                    BPos.append(pos)
                    pos -= ds
            else:
                BV.append(v * 3.6)
                BPos.append(pos)
            if pos <= start:
                break
    BV = BV[::-1]
    Pos = FPos
    SL_Grad_curve(slpos, slval, gradpos, gradval)
    time = 0
    for i in range(1, len(Pos)):
        v = min(FV[i], BV[i])
        dt = ds / ((v / 3.6 + V[-1] / 3.6) / 2)
        time += dt
        T.append(time)
        V.append(v)
    plt.figure()
    SL_Grad_curve(slpos, slval, gradpos, gradval)
    plt.plot(FPos, V, label='速度距离曲线')
    plt.xlabel('位移（米）')
    plt.ylabel('速度（千米/小时）')
    plt.legend()
    plt.savefig('mintime.svg', format='svg')
    plt.show()
    Pos.append(slpos[-1] + seroor)
    T.append(T[-1] + t0)
    V.append(0)
    T = pd.DataFrame(T)
    Pos = pd.DataFrame(Pos)
    V = pd.DataFrame(V)
    df = pd.concat([Pos, T, V], axis=1)
    df.columns = ['positions', 'runtime', 'speed']
    df.to_csv('Mintime-200m1.csv')


if __name__ == '__main__':
    MintimeCurve()
