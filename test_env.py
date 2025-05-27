from flappy_env import FlappyBirdEnv
import random

env = FlappyBirdEnv()
state = env.reset()

while True:
    action = random.choice([0, 1])  # 0: do nothing, 1: jump
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        state = env.reset()
