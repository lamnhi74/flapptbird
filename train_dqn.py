from flappy_gym_env import FlappyBirdGym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# Khởi tạo môi trường
env = DummyVecEnv([lambda: FlappyBirdGym()])

# Khởi tạo mô hình DQN
model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-3, buffer_size=50000, learning_starts=1000, batch_size=32)

# Huấn luyện mô hình
model.learn(total_timesteps=100_000)

# Lưu mô hình
model.save("dqn_flappybird")
rewards = []
obs = env.reset()
for episode in range(500):
    done = False
    total_reward = 0
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    rewards.append(total_reward)

# Vẽ biểu đồ bằng matplotlib
import matplotlib.pyplot as plt

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode - DQN")
plt.grid()
plt.show()
