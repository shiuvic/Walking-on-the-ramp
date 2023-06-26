import numpy as np

def save_weight(x):
    a = [x]
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/cube_weight.npy', a)

def clean_weight():
    # a = [0.8]
    a = [0.4]
    np.save('/Walking-on-the-ramp/robot_baseline/robot/resources/cube_weight.npy', a)

def load():
    x = np.load('/Walking-on-the-ramp/robot_baseline/robot/resources/cube_weight.npy')
    print(x)

clean_weight()

# load()