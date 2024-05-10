import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
import os

path = os.path.dirname(__file__)

def ob(acc):
    save(acc)
    # accx, accy, accz = acc

    accx = np.load(os.path.join(os.path.dirname(__file__), 'accx.npy'))
    accy = np.load(os.path.join(os.path.dirname(__file__), 'accy.npy'))
    accz = np.load(os.path.join(os.path.dirname(__file__), 'accz.npy'))


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

    # plt.plot(datax, label='x')
    # plt.plot(datay, label='y')
    # plt.plot(dataz, label='z')
    # # plt.xlabel('step', fontsize=20)7
    # # plt.legend(loc="upper left",
    # #            fontsize=20,
    # #            )
    # # plt.show()
    # plt.draw()
    # plt.pause(1./48.)
    # plt.clf()
    data = [datax, datay, dataz]
    return data


def save(acc):
    x, y, z = acc
    arrx = np.load(os.path.join(os.path.dirname(__file__), 'accx.npy'))
    arry = np.load(os.path.join(os.path.dirname(__file__), 'accy.npy'))
    arrz = np.load(os.path.join(os.path.dirname(__file__), 'accz.npy'))

    arrx = np.append(arrx, x)
    np.save(os.path.join(os.path.dirname(__file__),'accx.npy'), arrx)

    arry = np.append(arry, y)
    np.save(os.path.join(os.path.dirname(__file__),'accy.npy'), arry)

    arrz = np.append(arrz, z)
    np.save(os.path.join(os.path.dirname(__file__), 'accz.npy'), arrz)

def clear():
    x = []
    y = []
    z = []
    np.save(os.path.join(os.path.dirname(__file__), 'accx.npy'), x)
    np.save(os.path.join(os.path.dirname(__file__), 'accy.npy'), y)
    np.save(os.path.join(os.path.dirname(__file__), 'accz.npy'), z)

def plot_data():

    accx = np.load(os.path.join(os.path.dirname(__file__), 'accx.npy'))
    accy = np.load(os.path.join(os.path.dirname(__file__), 'accy.npy'))
    accz = np.load(os.path.join(os.path.dirname(__file__), 'accz.npy'))
    print(len(accx))
    # data = ((np.full(shape=128, fill_value=0)))
    num = 2
    datax = np.append((np.full(shape=128 - num, fill_value=0)), accx[0:num])
    datay = np.append((np.full(shape=128 - num, fill_value=0)), accy[0:num])
    dataz = np.append((np.full(shape=128 - num, fill_value=0)), accz[0:num])



    # print(len(data))


    plt.plot(datax, label='x')
    plt.plot(datay, label='y')
    plt.plot(dataz, label='z')
    plt.xlabel('step', fontsize=20)
    plt.ylabel('$Acceleration (m/s^2)$', fontsize=20)
    plt.legend(loc="upper left",
               fontsize=20,
               )

    plt.show()
    # plt.draw()
    # plt.pause(1./48.)
    # plt.clf()




# clear()
# pass
# # x = np.full(128, 5)
# # x = np.append(x, 6)
# x = []
# for i in range(256):
#     x = np.append(x, i)
# print(ob(x), len(ob(x)))
# print(x[-128:-1])

# plot_data()