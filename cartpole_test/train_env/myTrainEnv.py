import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mybasic
from mybasic import train
from mybasic import railway
from myRunningModel import run
import gym
from gym import spaces
from gym.utils import seeding


class TrainEnv(gym.Env):

    def __init__(self):
        df = pd.read_csv(r'train_env/Mintime-200m.csv')
        SpeedLimit = np.array(df.iloc[:, 3])
        TimeLimit = np.array(df.iloc[:, 2])
        self.ds = 200  # 运行步长为200m，在该200m的位移中将火车运行视为匀加速
        self.wspeed = 1
        self.wstop = 1
        self.wenergy = 0.2
        self.wtime = 0.2
        self.train = train()
        self.railway = railway()
        self.maxDistance = self.railway.D  # 火车运行距离
        self.maxTime = 1080  # 整条线路的计划运行时间
        self.maxSpeed = np.max(self.railway.slval) / 3.6  # 最高限速
        self.action_space = spaces.Discrete(4)  # 离散动作空间
        self.low = np.array([-10, -10, -10, -10])
        self.high = np.array([10, 10, 10, 10])
        self.observation_space = spaces.Box(self.low, self.high)
        self.dSpeedLimit = np.append(SpeedLimit, 0)
        self.dTimeLimit = TimeLimit
        self.done = False
        self.reason = None
        self.statelist = []
        self.rewardlist = []
        self.stepinfo = []

    def reset(self, seed=None, return_info=None, opitions=None):
        self.info = {"P": 0,
                     "V": 0,
                     "T": 1080,
                     "R": None,
                     "E": 0}
        self.ptest = [0]
        self.vtest = [0]
        self.rewardlist = []
        self.statelist = []
        self.stepinfo = []
        self.reason = None
        self.EC = 0
        self.ds = 200
        self.done = False
        self.timestep = 0
        self.state = run(self.railway.slpos[0], self.ds, 0, 0)
        while self.state.pos <= 1000:
            obs, _, _, info = self.step(0)
            self.obs = obs
            self.ptest.append(self.state.pos)
            self.vtest.append(self.state.v * 3.6)
        return self.obs

    def step(self, action):

        if self.state.pos == self.maxDistance:
            self.ds = 1

        Reward = 0
        Radd = 0
        Rstop = 0
        Rtime = 0
        Renergy = 0
        Rspeed = 0

        F = self.train.get_tracForce(self.state.v)  # 根据当前速度获得最大牵引力
        W = self.train.get_resist(self.state.v * 3.6) * self.train.M \
            + self.railway.get_grad(self.state.pos) * self.train.M * 9.8  # 根据当前速度和位置获得阻力
        B = self.train.get_brakeForce(self.state.v)  # 根据当前速度获得最大制动力

        if action == 0:  # 以最大牵引力运行
            lastpos = self.state.pos  # 上一个状态的位置
            dv, dpos, dtime = self.state.refresh(F, 0, self.ds)  # 获得当前状态的速度、位置、时间
            pos_cha = dpos - lastpos  # 计算位移大小
            dec = ((F / 1000) * pos_cha) / 3600  # 计算转移过程产生的能耗(kwh）
            dec_r = 1  # 归一化后作为能耗奖励，产生的能耗/该状态转移可能产生的最大能耗

        elif action == 1:  # 以与阻力相等的牵引力运行，即匀速运行
            lastpos = self.state.pos
            dv, dpos, dtime = self.state.refresh(W, 0, self.ds)
            pos_cha = dpos - lastpos
            dec = ((W / 1000) * pos_cha) / 3600
            dec_r = W / F

        elif action == 2:  # 既不牵引也不制动，惰行
            dv, dpos, dtime = self.state.refresh(0, 0, self.ds)
            dec = 0
            dec_r = 0

        elif action == 3:  # 以最大制动运行
            dv, dpos, dtime = self.state.refresh(0, B, self.ds)
            dec = 0
            dec_r = 0

        # 终止条件与奖励
        self.EC += dec
        self.timestep += 1
        dspeedlimit = self.dSpeedLimit[self.timestep] / 3.6  # 计算当前位置的限速大小
        Serror = abs(dpos - self.maxDistance)  # 计算距离终点还有多少距离
        Terror = abs(dtime - self.maxTime)

        if self.state.v == 0:
            self.reason = 'FR'
            self.done = True
        elif np.float32(dv) > np.float32(dspeedlimit):
            self.reason = 'OS'
            self.done = True
            Rspeed -= 5

        if self.done:
            if Serror > 1:
                if Serror > 200:
                    Rstop -= 5 + 25 * (Serror / self.maxDistance)
                elif Serror <= 200:
                    Rstop -= 5 - 50 * (1 - Serror / 200)
            elif Serror <= 1 and Terror < 1:
                Radd += 50
                self.reason = 'SR'

        tmin = self.dTimeLimit[-1] - self.dTimeLimit[self.timestep]  # 计算当前位置的最短运行时间
        tres = self.maxTime - self.state.t  # 还剩多少时间
        Rtime += (tres - tmin) / (self.maxTime - self.dTimeLimit[-1]) if (tres - tmin) < 0 else 0  # 剩余时间如果小于最短运行时间，则给予惩罚

        Renergy += 1 - dec_r  # 能耗越低奖励越高

        Rstop *= self.wstop
        Rtime *= self.wtime
        Renergy *= self.wenergy
        Rspeed *= self.wspeed
        Reward += Radd + Rstop + Rtime + Renergy + Rspeed

        next_limit = self.dSpeedLimit[self.timestep + 1] / 3.6
        self.obs = np.array([self.state.pos / self.maxDistance,  # 对状态进行归一化
                             self.state.v / self.maxSpeed,
                             tres / self.maxTime,
                             next_limit / self.maxSpeed])

        self.info = {"P": self.state.pos,
                     "V": self.state.v,
                     "T": self.state.t,
                     "R": self.reason,
                     "E": self.EC}

        self.statelist = [self.state.pos, self.state.v, self.state.t, action]
        self.rewardlist = [Rstop, Rtime, Renergy, Radd, Rspeed]
        self.stepinfo.append(self.statelist + self.rewardlist)

        return self.obs, Reward, self.done, self.info

    def render(self, mode='huamn'):
        self.ptest.append(self.state.pos)
        self.vtest.append(self.state.v * 3.6)
        if self.done:
            print(self.info)

    def plot_vs_curve(self):
        plt.figure()
        mybasic.SL_Grad_curve(self.railway.slpos, self.railway.slval,
                              self.railway.gradpos, self.railway.gradval)
        plt.plot(self.ptest, self.vtest)
        plt.show()
        plt.clf()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]