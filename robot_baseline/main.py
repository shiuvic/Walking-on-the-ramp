import gym
import os

import numpy
import numpy as np
from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.type_aliases import TensorDict
from robot_baseline import robot
from stable_baselines3.ppo.policies import CnnPolicy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Parallel environments
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CustomCNN, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            print(key, subspace)
            if key != 'torque':
                print(subspace.shape)
                extractors[key] = nn.Sequential(
                                                nn.LazyConv1d(1, 3, stride=1),
                                                nn.MaxPool1d(3),
                                                nn.MaxPool1d(2),
                                                )
                total_concat_size += 21
            else:
                extractors[key] = nn.Sequential(nn.Flatten(),
                                                nn.Linear(4, 16),
                                                nn.ReLU()
                                                )
                total_concat_size += 16

                # if key == 'accx':
                #     extractors[key] = nn.Sequential(
                #                                     nn.LazyLinear(64),
                #                                     nn.ReLU(),
                #                                     nn.Linear(64, 16),
                #                                     nn.ReLU(),
                #                                     # nn.ReLU()
                #                                     )
                #     total_concat_size += 16
                #
                # elif key == 'accy':
                #     extractors[key] = nn.Sequential(
                #                                     nn.LazyLinear(64),
                #                                     nn.ReLU(),
                #                                     nn.Linear(64, 32),
                #                                     nn.ReLU(),
                #                                     nn.Linear(32, 8),
                #                                     nn.ReLU()
                #                                     )
                #     total_concat_size += 8
                #
                # elif key == 'accz':
                #     extractors[key] = nn.Sequential(
                #                                     nn.Linear(128, 64),
                #                                     nn.ReLU(),
                #                                     nn.Linear(64, 8),
                #                                     nn.ReLU()
                #                                     )
                #     total_concat_size += 8
                #
                # elif key == 'torque':
                #     extractors[key] = nn.Sequential(
                #                                     nn.LazyLinear(16),
                #                                     nn.ReLU(),
                #                                     nn.Linear(16, 8),
                #                                     nn.ReLU()
                #                                     )
                #     total_concat_size += 8

        self.flattened_tensor = nn.Flatten()

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            print(extractor(observations[key]).shape)

        return th.cat(encoded_tensor_list, dim=1)

class Custom(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super(Custom, self).__init__(observation_space, features_dim)

        self.cnn1 = nn.LazyConv1d(out_channels=4, kernel_size=3, padding=1)
        self.cnn11 = nn.LazyConv1d(out_channels=2, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.cnn2 = nn.LazyConv1d(out_channels=4, kernel_size=3, padding=1)
        self.cnn21 = nn.LazyConv1d(out_channels=2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.cnn3 = nn.LazyConv1d(out_channels=4, kernel_size=3, padding=1)
        self.cnn31 = nn.LazyConv1d(out_channels=2, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.cnn4 = nn.LazyConv1d(out_channels=4, kernel_size=3, padding=1)
        self.cnn41 = nn.LazyConv1d(out_channels=2, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.fc1 = nn.LazyLinear(200)
        self.re1 = nn.ReLU()
        self.fc2 = nn.Linear(200, 100)
        self.re2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)
        self.re3 = nn.ReLU()
        self.fc4 = nn.Linear(10, 128)
        self.re4 = nn.ReLU()

    def forward(self, observations: TensorDict) -> th.Tensor:
        # torque = observations['torque'].view(-1, 4)
        # print('observations', observations)
        accx = observations['accx'].view(-1, 1, 128)
        accy = observations['accy'].view(-1, 1, 128)
        accz = observations['accz'].view(-1, 1, 128)
        # ori = observations['ori'].view(-1, 1, 3)
        # print(accx.shape)
        # print(accy.shape)
        # print(accz.shape)
        # print(torque.shape)

        accx_out = self.relu1(self.cnn11(self.cnn1(accx)))
        accy_out = self.relu2(self.cnn21(self.cnn2(accy)))
        accz_out = self.relu3(self.cnn31(self.cnn3(accz)))
        # torque_out = self.relu4(self.cnn4(torque))
        # ori_out = self.relu3(self.cnn31(self.cnn3(ori)))

        # print(accx_out.shape)
        # print(accy_out.shape)
        # print(accz_out.shape)
        # print(torque_out.shape)
        # print(ori_out.shape)

        extractors_out = th.cat((accx_out, accy_out, accz_out), dim=0).view(-1, 128*6)
        # print(f'\033[37m exteactor_out shape = {extractors_out.shape}')
        # extractors_out = th.cat((extractors_out, ori_out.view(-1)), dim=0).view(-1, 129*6)
        # print(f'\033[37m exteactor_out shape = {extractors_out.shape}')
        # extractors_out = extractors_out.view(-1, 129*6)
        # print('\033[37m', extractors_out.shape, '\033[0m extractors output shape')

        # extractors_out = th.cat((extractors_out, torque), dim=1)
        # print('\033[37m', extractors_out.shape, '\033[0m extractors output shape')

        extractors_out = self.re1(self.fc1(extractors_out))
        extractors_out = self.re2(self.fc2(extractors_out))
        extractors_out = self.re3(self.fc3(extractors_out))
        extractors_out = self.re4(self.fc4(extractors_out))

        # print(extractors_out.shape, 'extractors output shape')

        return extractors_out

policy_kwargs = dict(
    features_extractor_class=Custom
)
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='rl_model')

#------------------------------------------PPO---------------------------------------------#

# env = make_vec_env('robot-v0', n_envs=1)
#
# # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256, dict(pi=[64, 32], vf=[64, 32])])
#
# model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# # model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
#
# model.learn(total_timesteps=300000, callback=checkpoint_callback)
# model.save("ppo2_robot")
# model = env.close()
#
# # del model # remove to demonstrate saving and loading
#
# # model = PPO.load("ppo2_robot")
# # model = PPO.load("./logs/rl_model_290000_steps.zip")
# # model = PPO.load("./logs/random/rl_model_100000_steps")
# # model = PPO.load("./logs/1005_ramp_slop_rand/ppo2_robot")
# # obs = env.reset()
# # while True:
# #     action, _states = model.predict(obs)
# #     obs, rewards, dones, info = env.step(action)

#------------------------------------------PPO---------------------------------------------#

#------------------------------------------SAC---------------------------------------------#
from stable_baselines3 import SAC, DQN

# env = gym.make("Pendulum-v1")
env = make_vec_env('robot-v0', n_envs=1)

model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

model.learn(total_timesteps=300000, log_interval=4, callback=checkpoint_callback)
model.save("sac_robot")
model = env.close()

# del model # remove to demonstrate saving and loading

# model = SAC.load("sac_robot.zip")
# print(model.policy)
# print('>>>>>>>><<<<<<<<<<<<<<<<<<',model.policy)

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)


#------------------------------------------SAC---------------------------------------------#
