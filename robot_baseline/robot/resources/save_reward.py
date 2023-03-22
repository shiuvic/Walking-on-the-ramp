import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
rewardarr = []
# plt.figure(figsize=(6, 4))
#     plt.ylabel('reward')
#     plt.xlabel('total_timesteps')


import os

reward_path = os.path.join(os.path.dirname(__file__), 'SAC_reward.npy')
finish = os.path.join(os.path.dirname(__file__), 'SAC_finish.npy')
# print(reward_path)
# reward_path = '/Walking-on-the-ramp/robot_baseline/robot/resources/SAC_reward.npy'
def save(reward):
    # rewardarr.append(reward)
    # plt.plot(rewardarr)
    # plt.draw()
    # plt.pause(1./48.)
    # plt.clf()
    z = reward
    arr = np.load(reward_path)
    arr = np.append(arr, z)
    np.save(reward_path, arr)
    if z > 15:
        save_done(z)

def step_C(step):
    x = step
    arr = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/step.npy')
    arr = np.append(arr, x)
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/step.npy', arr)

def clear():
    a = []
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/finish.npy',a)
    np.save(reward_path, a)
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/step.npy', a)

def smooth(y, box):

    ans = []
    for i in range(0, 300000-box, 1):
        y_sum = sum(y[i:i+box])
        ans.append(y_sum/box)
    np.save("./smooth-SAC-10000", ans)
    return ans

def load():
    data1 = np.load(reward_path)
    # smoo  = smooth(data1, 10000)
    # smoo = np.load("./smooth-SAC-10000.npy")
    # plt.plot(smoo)
    plt.plot(data1)
    plt.show()

def save_done(reward):
    z = reward
    arr = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/SAC_finish.npy')
    arr = np.append(arr, z)
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/SAC_finish.npy', arr)


def load_done():
    data1 = np.load("/Walking-on-the-ramp/robot_baseline/robot/resources/SAC_finish.npy")
    plt.plot(data1)
    plt.show()

def load_data():
    data1 = np.load("/Walking-on-the-ramp/robot_baseline/logs/1005_ramp_slop_rand/reward.npy")
    plt.plot(data1)
    plt.show()

def show_step():
    data1 = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/step.npy')
    plt.plot(data1)
    plt.show()

clear()

# load()

# load_data()