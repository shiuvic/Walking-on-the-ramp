import random
import math
import numpy as np


def clean_ramp():
    # a = [40]
    a = [25]
    # a = [30]
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/ramp.npy', a)

def load():
    x = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/ramp.npy')
    print(180/x[0])
    return x

clean_ramp()

# load()
