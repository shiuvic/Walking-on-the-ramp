import gym
import os

import numpy
from stable_baselines3 import PPO
import torch as th
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
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

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CustomCNN, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            # print(key, subspace)
            if is_image_space(subspace):
                # extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                # if key != 'torque':
                #     extractors[key] = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=0),
                #                                     nn.ReLU())
                #
                #
                #     # extractors[key] = nn.Sequential(nn.Flatten(),
                #     #                                 nn.Linear(128, 64),
                #     #                                 nn.ReLU(),
                #     #                                 nn.Linear(64, 32))
                #     total_concat_size += 1*381
                # else:
                #     # extractors[key] = nn.Sequential(nn.Flatten(),
                #     #                                 nn.Linear(4, 16),
                #     #                                 nn.ReLU(),
                #     #                                 nn.Linear(16, 64),
                #     #                                 nn.Linear(64, 32))
                #     extractors[key] = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=2, kernel_size=2, padding=0),
                #                                     nn.ReLU())
                #     total_concat_size += 4*64

                if key == 'accx':
                    extractors[key] = nn.Sequential(nn.Flatten(),
                                                    nn.LazyLinear(64),
                                                    nn.ReLU(),
                                                    nn.Linear(64, 16),
                                                    nn.ReLU(),
                                                    # nn.ReLU()
                                                    )
                    total_concat_size += 16

                elif key == 'accy':
                    extractors[key] = nn.Sequential(nn.Flatten(),
                                                    nn.LazyLinear(64),
                                                    nn.ReLU(),
                                                    nn.Linear(64, 32),
                                                    nn.ReLU(),
                                                    nn.Linear(32, 8),
                                                    nn.ReLU()
                                                    )
                    total_concat_size += 8

                elif key == 'accz':
                    extractors[key] = nn.Sequential(nn.Flatten(),
                                                    nn.Linear(128, 64),
                                                    nn.ReLU(),
                                                    nn.Linear(64, 8),
                                                    nn.ReLU()
                                                    )
                    total_concat_size += 8

                elif key == 'torque':
                    extractors[key] = nn.Sequential(nn.Flatten(),
                                                    nn.LazyLinear(16),
                                                    nn.ReLU(),
                                                    nn.Linear(16, 8),
                                                    nn.ReLU()
                                                    )
                    total_concat_size += 8

        self.flattened_tensor = nn.Flatten()

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # print(encoded_tensor_list)

        return th.cat(encoded_tensor_list, dim=1)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        print('hello world')
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):

                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            else:
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self,observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # print(encoded_tensor_list)
        return th.cat(encoded_tensor_list, dim=1)



policy_kwargs = dict(
    features_extractor_class=CustomCNN
)

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./logs/', name_prefix='rl_model')

###########################################PPO##############################################

# env = make_vec_env('robot-v0', n_envs=1)
#
# # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256, dict(pi=[64, 32], vf=[64, 32])])
#
# model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, n_steps=64)
# # model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
#
# # model.learn(total_timesteps=200000, callback=checkpoint_callback)
# # model.save("ppo2_robot")
# # model = env.close()
#
#
#
#
# del model # remove to demonstrate saving and loading
#
# model = PPO.load("ppo2_robot")
# # model = PPO.load("./logs/rl_model_290000_steps.zip")
# # model = PPO.load("./logs/random/rl_model_100000_steps")
# # model = PPO.load("./logs/1005_ramp_slop_rand/ppo2_robot")
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)

###########################################PPO##############################################

###########################################SAC##############################################
from stable_baselines3 import SAC

# env = gym.make("Pendulum-v1")
env = make_vec_env('robot-v0', n_envs=1)

model = SAC("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=300000, log_interval=4, callback=checkpoint_callback)
# model.save("sac_robot")
# model = env.close()

# del model # remove to demonstrate saving and loading
#
# model = SAC.load("sac_robot")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)

###########################################SAC##############################################