import numpy as np
import os
path = os.path.join(os.path.dirname(__file__), 'cube_weight.npy')
def save_weight(x):
    a = [x]
    np.save(path, a)

def clean_weight():
    # a = [0.8]
    a = [0.4]
    np.save(path, a)

def load():
    x = np.load(path)
    print(x)

clean_weight()

# load()