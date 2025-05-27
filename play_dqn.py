from flappy_gym_env import FlappyBirdGym
from stable_baselines3 import DQN
import time

env = FlappyBirdGym()
model = DQN.load("dqn_flappybird")

obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    time.sleep(1/30)
    if done:
        obs = env.reset()
