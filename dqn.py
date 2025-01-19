import logging
import random
from collections import deque
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import nn

import gymnasium as gym
import ale_py


def preprocess(frames):
    # TODO
    return frames[-1]


def qnet(num_actions):
    # input: 4 x 84 x 84
    return nn.Sequential(
        nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, padding=0, stride=4),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=0, stride=2),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=3136, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=num_actions),
    )


def copy_weights(src, dst):
    # for s, d in zip(src.parameters(), dst.parameters()):
    #     d.data = s.data.clone()
    dst.load_state_dict(src.state_dict())


def eps_greedy(actions_values, eps):
    if random.random() < eps:
        return random.randint(0, len(actions_values) - 1)
    return torch.argmax(actions_values)


def sample_batch(buffer, batch_size):
    s1_batch = []
    a_batch = []
    r_batch = []
    s2_batch = []
    for i in np.random.randint(0, len(buffer), (batch_size,)):
        s1, a, r, s2 = buffer[i]
        s1_batch.append(s1)
        a_batch.append(a)
        r_batch.append(r)
        s2_batch.append(s2)
    s1_batch = torch.stack(s1_batch, dim=0)
    a_batch = torch.tensor(a_batch)
    r_batch = torch.tensor(r_batch)
    s2_batch = torch.stack(s2_batch, dim=0)
    return s1_batch, a_batch, r_batch, s2_batch


def dqn(env, q1, q2, params, sgd_step):
    buffer = deque(maxlen=params.buffer_size)
    step = 0

    x, info = env.reset(seed=13)
    for episode in range(params.num_episodes):
        frames = deque([x], maxlen=params.frames_per_state)
        s1 = preprocess(frames)

        for t in range(params.max_episode_time):
            # eps annealed linearly from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter
            eps = max(-9e-7 * step + 1., .1)
            a = eps_greedy(q1(s1), eps)

            x, r, terminated, truncated, info = env.step(a)
            frames.append(x)
            s2 = preprocess(frames)
            transition = (s1, a, r, s2)
            buffer.append(transition)
            s1 = s2

            if len(buffer) == params.buffer_size:
                batch = sample_batch(buffer, params.batch_size)
                sgd_step(q1, q2, batch)

            step += 1
            if step % params.target_update_freq == 0:
                copy_weights(q1, q2)

            if terminated or truncated:
                break

        x, info = env.reset()


def dqn_step(q1, q2, batch, opt, params):
    s1_batch, a_batch, r_batch, s2_batch = batch
    batch_size = s1_batch.shape[0]
    with torch.no_grad():
        a_max = torch.argmax(q2(s2_batch), dim=1)
        target = r_batch + params.gamma * q2(s2_batch)[torch.arange(batch_size), a_max]
    opt.zero_grad()
    output = q1(s1_batch)[torch.arange(batch_size), a_batch]
    loss = torch.mean((target - output) ** 2)
    loss.backward()
    opt.step()


def double_dqn_step(q1, q2, batch, opt, params):
    s1_batch, a_batch, r_batch, s2_batch = batch
    batch_size = s1_batch.shape[0]
    with torch.no_grad():
        a_max = torch.argmax(q1(s2_batch), dim=1)
        target = r_batch + params.gamma * q2(s2_batch)[torch.arange(batch_size), a_max]
    opt.zero_grad()
    output = q1(s1_batch)[torch.arange(batch_size), a_batch]
    loss = torch.mean((target - output) ** 2)
    loss.backward()
    opt.step()


@dataclass
class Params:
    # M in the paper
    num_episodes = 10000
    # T in the paper
    max_episode_time = 1000
    # C in the paper
    target_update_freq = 100
    # N in the paper
    buffer_size = 500
    # m in the paper
    frames_per_state = 4
    gamma = .99
    lr = 1e-3
    batch_size = 32


def main():
    logging.basicConfig(
        format='%(asctime)s %(module)s %(levelname)s %(message)s',
        level=logging.INFO, handlers=[logging.StreamHandler()], force=True)

    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    num_actions = env.action_space.n

    q1 = qnet(num_actions)
    q2 = qnet(num_actions)

    params = Params()

    opt = torch.optim.RMSprop(q1.parameters(), lr=params.lr)
    sgd_step = partial(double_dqn_step, opt=opt, params=params)
    dqn(env, q1, q2, params, sgd_step)


if __name__ == '__main__':
    main()
