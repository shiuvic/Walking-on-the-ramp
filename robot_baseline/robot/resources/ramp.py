import random
import math
import numpy as np


def clean_ramp():
    a = [40]
    # a = [25]
    # a = [30]
    np.save('ramp.npy', a)

def load():
    x = np.load('ramp.npy')
    print(180/x[0])
    return x

clean_ramp()

# load()
