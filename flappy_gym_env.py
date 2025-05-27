import gym
from gym import spaces
import numpy as np
from flappy_env import FlappyBirdEnv  

class FlappyBirdGym(gym.Env):
    def __init__(self):
        super(FlappyBirdGym, self).__init__()
        self.env = FlappyBirdEnv()
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 = không nhảy, 1 = nhảy

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done, info

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        pygame.quit()
