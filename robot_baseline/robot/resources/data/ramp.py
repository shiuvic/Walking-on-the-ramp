import random
import math
import numpy as np
import os
path = os.path.join(os.path.dirname(__file__), 'ramp.npy')

def clean_ramp():
    ##神秘數字##
    # a = [180]
    a = [40] #-0.07853982
    # a = [25]   #-0.12853982
    # a = [30]
    # a = [10]

    np.save(path, a)

def load():
    x = np.load(path)
    print("ramp degree:", 180/x[0])
    return x
def save(angle):
    np.save(path, angle)

clean_ramp()
#
# load()