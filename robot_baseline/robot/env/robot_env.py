import os
import random
import sys
import time

import gym
import numpy as np
import pybullet as p
from robot.resources.core.op3 import OP3
from robot.resources.walker import Walker
from gym.utils import seeding
from robot.resources.save_reward import save

class Op3Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Op3Env, self).__init__()

        '''
        # 動作空間
        # 第一個維度:
        # 第二個維度:
        # 第二個維度:
        '''

        # self.action_space = gym.spaces.box.Box(
        #     low=np.array([-0.02, -0.02, -0.02, -0.02, -0.02, -0.02], dtype=np.float32),
        #     high=np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32))

        self.action_space = gym.spaces.box.Box(
                low=np.array([-0.02, -0.02, -0.02], dtype=np.float32),
                high=np.array([0.02, 0.02, 0.02], dtype=np.float32))

        '''
        # 觀察空間
        # 
        '''
        space = {
            'accx': gym.spaces.box.Box(low=-20, high=20, shape=(128,)),
            'accy': gym.spaces.box.Box(low=-20, high=20, shape=(128,)),
            'accz': gym.spaces.box.Box(low=-20, high=20, shape=(128,)),
            'torque': gym.spaces.box.Box(low=-15, high=15, shape=(4, ))
        }

        # self.observation_space = gym.spaces.box.Box(
        #     low=np.append(np.full(128, -20), np.append(np.full(128, -20), np.append(np.full(128, -20), [-15, -15, -15, -15]))),
        #     high=np.append(np.full(128, 20), np.append(np.full(128, 20), np.append(np.full(128, 20), [15, 15, 15, 15]))))

        self.observation_space = gym.spaces.dict.Dict(space)

        self.np_random, _ = gym.utils.seeding.np_random()

        # 選擇連結方式
        # self.client = p.connect(p.DIRECT)
        self.client = p.connect(p.GUI)
        # 加速訓練
        p.setTimeStep(1/240, self.client)
        # 初始化所有東西
        self.OP3 = None
        self.done = False
        self.n_step = 0
        self.reward = 0


    def step(self, action):
        self.n_step += 1
        self.OP3.apply_action(action, self.n_step)
        time.sleep(0.2)
        # print("State = ",self.state)
        OP3_ob = self.OP3.get_obsevervation()
        self.state = self.OP3.get_state()
        self.reward = self.reward_fun()
        self.cube_ran()
        # save(self.reward)
        info = {}
        return OP3_ob, self.reward, self.done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @property
    def reset(self):
        # 初始化所有東西
        p.resetSimulation()
        self.OP3 = Walker(self.client)
        time.sleep(0.00001)
        self.OP3.reset()
        self.done = False
        # OP3_ob = self.OP3.get_observation()
        self.state = self.OP3.get_state()
        OP3_ob = self.OP3.get_obsevervation()

        return OP3_ob

    def render(self, mode='human'):
        pass

    def close(self):
        print("close")
        self.OP3.stop()
        self.OP3.stop_run = True
        p.disconnect(self.client)

    def reward_fun(self):
        robot = self.OP3.get_position()
        dis = abs(np.linalg.norm(robot - [0, 0, 1.5]))
        reward = 0

        if dis > 3:
            reward = dis + 5
            self.OP3.close()
            self.done = True
        elif robot[0] < 0:
            reward -= 1

        else:
            reward = dis

        if self.OP3.is_fallen():
            reward = reward - 10
            self.OP3.close()
            self.done = True

        return reward

    def cube_ran(self):
        if self.n_step % 2048 == 0:
            self.OP3.chane_cube_weight()
            print("-----------------change weight-------------------")
        if self.n_step % 2048 == 0:
            self.OP3.rand_ramp()
            print("------------------change ramp--------------------")
