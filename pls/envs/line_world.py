from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LineWorldSafetyEnv(gym.Env):
    """Small 1D safety environment for shield debugging.

    The agent moves on a 1D line with terminal unsafe states at both edges.
    """

    metadata = {"render_modes": []}

    def __init__(self, length: int = 7, max_steps: int = 50):
        if length < 3:
            raise ValueError("length must be >= 3")

        self.length = int(length)
        self.max_steps = int(max_steps)

        self.action_space = spaces.Discrete(3)  # 0=left, 1=stay, 2=right
        self.observation_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        self.position = self.length // 2
        self.step_count = 0

    def _obs(self) -> np.ndarray:
        norm = self.position / float(self.length - 1)
        return np.array([norm], dtype=np.float32)

    def _is_unsafe(self) -> bool:
        return self.position == 0 or self.position == self.length - 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Reset to the center position and return observation + info."""
        super().reset(seed=seed)
        self.position = self.length // 2
        self.step_count = 0
        return self._obs(), {"position": self.position}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply action and return Gymnasium step tuple.

        Returns:
            observation, reward, terminated, truncated, info
        """
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action: {action}")

        self.step_count += 1
        if action == 0:
            self.position = max(0, self.position - 1)
        elif action == 2:
            self.position = min(self.length - 1, self.position + 1)

        terminated = self._is_unsafe()
        truncated = self.step_count >= self.max_steps and not terminated

        reward = -1.0 if terminated else 0.1

        info = {
            "position": self.position,
            "unsafe": terminated,
        }
        return self._obs(), reward, terminated, truncated, info
