import numpy

from robot_baseline import robot
import gym
def register(id, entry_point, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point,
    )
register(
    id='robot-v0',
    entry_point='robot.env:Op3Env'
)
