import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MySafetyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.array([0.5], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        obs = np.array([0.5], dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
