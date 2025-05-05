# This script uses the framework provided by Bernd Schrooten in Deep Reinforcement Learning with Gymnasium published by Packt Publishing
# Code generation assistant: DataCamp AI's Python code generation tool
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import numpy as np
import gymnasium as gym 

#for i in gym.envs.registry.keys():
#    print(i)

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.n)
# print(env.action_space.sample())

observation, info = env.reset()
print(observation)
print(info)