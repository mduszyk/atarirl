import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
    if reward > 0:
        print('reward', reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
