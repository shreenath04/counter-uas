import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env import CounterUASEnv

class CounterUASGymEnv(gym.Env):
    def __init__(self, num_friendly=5, num_hostile=5, grid_size=50):
        super().__init__()
        self.env = CounterUASEnv(
            grid_size=grid_size,
            num_friendly=num_friendly,
            num_hostile=num_hostile
        )
        
        
        state_size = (num_friendly*3) + (num_friendly * 3) + (num_hostile * 7)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([27] * num_friendly)
        
        self.num_friendly = num_friendly
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.env.reset()
        obs = self._build_obs(state)
        return obs, {}
    
    def step(self, actions):
        actions = list(actions)
        state, rewards, done = self.env.step(actions)
        obs = self._build_obs(state)
        total_reward = float(np.sum(rewards))
        return obs, total_reward, done, False, {
            'intercepted': sum(1 for a in self.env.hostile_alive if not a) - self.env.breaches,
            'breaches': self.env.breaches,
            'friendlies_lost': sum(1 for a in self.env.friendly_alive if not a)
        }
    
    def _build_obs(self, state):
        obs_parts = []
        for i in range(self.num_friendly):
            obs_parts.extend(self.env.friendly_drones[i])
        obs_parts.extend(state)
        return np.array(obs_parts, dtype=np.float32)