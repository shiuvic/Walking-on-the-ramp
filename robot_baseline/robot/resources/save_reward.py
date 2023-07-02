import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('tkagg')
rewardarr = []
# plt.figure(figsize=(6, 4))
#     plt.ylabel('reward')
#     plt.xlabel('total_timesteps')


import os

SAC_reward_path = os.path.join(os.path.dirname(__file__), 'SAC_reward.npy')
SAC_finish_path = os.path.join(os.path.dirname(__file__), 'SAC_finish.npy')
# print(reward_path)
# reward_path = '/Walking-on-the-ramp/robot_baseline/robot/resources/SAC_reward.npy'

def save(reward):
    # rewardarr.append(reward)
    # plt.plot(rewardarr)
    # plt.draw()
    # plt.pause(1./48.)
    # plt.clf()
    z = reward
    arr = np.load(SAC_finish_path)
    arr = np.append(arr, z)
    np.save(SAC_finish_path, arr)
    if z > 15:
        save_done(z)

def step_C(step):
    step_path = os.path.join(os.path.dirname(__file__), 'step.npy')
    x = step
    arr = np.load(step_path)
    arr = np.append(arr, x)
    np.save(step_path, arr)

def clear():
    finish_path = os.path.join(os.path.dirname(__file__), 'finish.npy')
    a = []
    np.save(finish_path,a)
    np.save(SAC_reward_path, a)
    np.save(finish_path, a)

def smooth(y, box):

    ans = []
    for i in range(0, 200000-box, 1):
        y_sum = sum(y[i:i+box])
        ans.append(y_sum/box)
    # np.save("./smooth-SAC-30000", ans)
    return ans

def load():
    data1 = np.load(SAC_reward_path)
    # smoo  = smooth(data1, 30000)
    # smoo = np.load("./smooth-SAC-10000.npy")
    # plt.plot(smoo)
    plt.plot(data1)
    plt.show()

def save_done(reward):
    z = reward
    arr = np.load(SAC_finish_path)
    arr = np.append(arr, z)
    np.save(SAC_finish_path, arr)


def load_done():
    data1 = np.load(SAC_finish_path)
    plt.plot(data1)
    plt.show()

def load_data():
    data1 = np.load("/Walking-on-the-ramp/robot_baseline/logs/1005_ramp_slop_rand/reward.npy")
    plt.plot(data1)
    plt.show()

def show_step():
    step_path = os.path.join(os.path.dirname(__file__), 'step.npy')
    data1 = np.load(step_path)
    plt.plot(data1)
    plt.show()

# clear()

load()

# load_data()