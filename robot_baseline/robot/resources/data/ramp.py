import random
import math
import numpy as np
import os
path = os.path.join(os.path.dirname(__file__), 'ramp.npy')

def clean_ramp():
    ##神秘數字##
    # a = [180]
    a = [40]
    # a = [25]
    # a = [30]
    np.save(path, a)

def load():
    x = np.load(path)
    print(180/x[0])
    return x

clean_ramp()

# load()
