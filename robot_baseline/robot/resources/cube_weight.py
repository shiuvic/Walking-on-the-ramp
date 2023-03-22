import numpy as np

def save_weight(x):
    a = [x]
    np.save('cube_weight.npy', a)

def clean_weight():
    a = [0.8]
    # a = [0.6]
    np.save('cube_weight.npy', a)

def load():
    x = np.load('cube_weight.npy')
    print(x)

clean_weight()

# load()