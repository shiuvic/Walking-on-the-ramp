# t = 1. / 500.
# pos, ang = p.getBasePositionAndOrientation(self.robot)
# ang = p.getEulerFromQuaternion(ang)
# # 角度轉成方向
# ori = (math.cos(ang[2]), math.sin(ang[2]))
# # 取得xy座標
# pos = pos[:2]
# # 取得速度
# pre_vel = np.zeros(3)
# new_vel = np.zeros(3)
# pre_vel[0] = p.getBaseVelocity(self.robot)[0][0]
# pre_vel[1] = p.getBaseVelocity(self.robot)[0][1]
# pre_vel[2] = p.getBaseVelocity(self.robot)[0][2]
# # 串街 位置, 方向, 加速度 當作 observation
# time.sleep(t)
# new_vel[0] = p.getBaseVelocity(self.robot)[0][0]
# new_vel[1] = p.getBaseVelocity(self.robot)[0][1]
# new_vel[2] = p.getBaseVelocity(self.robot)[0][2]
# acc = np.zeros(3)
# acc[0] = (new_vel[0] - pre_vel[0]) / t
# acc[1] = (new_vel[1] - pre_vel[1]) / t
# acc[2] = (new_vel[2] - pre_vel[2]) / t
# # observation = (pos + ori + pre_vel + new_vel )
# # observation = pre_vel
# # print("%.6f   %.6f   %.6f"%(pre_vel, new_vel, acc))
# # print(pre_vel, new_vel)
# # print("Acceleration: ", end=' ')
# # a = acc
# # l_sho_pitch = p.getJointState(self.robot, 12)[0]
# # l_sho_roll = p.getJointState(self.robot, 13)[0]
# # l_el = p.getJointState(self.robot, 14)[0]
# # pos = [l_sho_pitch, l_sho_roll, l_el]
# # pos*np.array([1])
# observation = acc
# return observation

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import numpy as np
accx = []
accy = []
accz = []
# plt.figure()
def show_acc(acc):
    accx.append(acc[0])
    accy.append(acc[1])
    accz.append(acc[2])
    # plt.subplot(131)
    plt.figure()
    plt.plot(accx)
    # plt.subplot(132)
    plt.figure()
    plt.plot(accy)
    # plt.subplot(133)
    plt.figure()
    plt.plot(accz)
    # plt.tight_layout()
    # plt.show()
    plt.draw()
    plt.pause(1./48.)
    plt.clf()
