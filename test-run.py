import math
import random
from collections import deque, namedtuple
from itertools import count

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image


def policy(observation):
    return [0, 1, 0]


env = gym.make("CarRacing-v2", render_mode="human")
observation, info = env.reset(seed=420)
for _ in range(1000):
    action = policy(observation)  # User-defined policy function
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("terminated")
        observation, info = env.reset()
env.close()
