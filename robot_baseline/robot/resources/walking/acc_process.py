import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('tkagg')


def ob(acc):
    save(acc)
    # accx, accy, accz = acc

    accx = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accx.npy')
    accy = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accy.npy')
    accz = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accz.npy')


    if len(accx) < 128:
        datax = np.append((np.full(shape=128-len(accx), fill_value=0)), accx)
    else:
        datax = accx[len(accx)-128:len(accx)]

    if len(accy) < 128:
        datay = np.append((np.full(shape=128-len(accy), fill_value=0)), accy)
    else:
        datay = accy[len(accy)-128:len(accy)]

    if len(accz) < 128:
        dataz = np.append((np.full(shape=128-len(accz), fill_value=0)), accz)
    else:
        dataz = accz[len(accz)-128:len(accz)]

    # plt.plot(datax)
    # plt.plot(datay)
    # plt.plot(dataz)
    # plt.draw()
    # plt.pause(1./48.)
    # plt.clf()
    data = [datax, datay, dataz]
    return data


def save(acc):
    x, y, z = acc

    arrx = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accx.npy')
    arrx = np.append(arrx, x)
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accx.npy', arrx)

    arry = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accy.npy')
    arry = np.append(arry, y)
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accy.npy', arry)

    arrz = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accz.npy')
    arrz = np.append(arrz, z)
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accz.npy', arrz)

def clear():
    x = []
    y = []
    z = []
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accx.npy', x)
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accy.npy', y)
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accz.npy', z)

def plot_data():

    accx = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accx.npy')
    accy = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accy.npy')
    accz = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/walking/accz.npy')
    plt.plot(accx, label='x')
    plt.plot(accy, label='y')
    plt.plot(accz, label='z')
    plt.legend()
    plt.draw()
    plt.pause(1./48.)
    plt.clf()

# clear()
# plot_data()
# # x = np.full(128, 5)
# # x = np.append(x, 6)
# x = []
# for i in range(256):
#     x = np.append(x, i)
# print(ob(x), len(ob(x)))
# # print(x[-128:-1])

# plot_data()