import logging
import random
from collections import deque
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import gymnasium as gym
import ale_py


def preprocess(frames):
    # Max of pixel values for each channel between current and previous frames
    frames = [torch.maximum(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    frames = torch.stack(frames)

    # Extract the Y channel, luminance, H x W x 3 -> H x W x 1, Y = 0.299 * R + 0.587 * G + 0.114 * B
    # This reflects how humans perceive brightness.
    weights = torch.tensor([0.299, 0.587, 0.114], device=frames[0].device).view(1, 1, 1, 3)
    frames = torch.sum(frames * weights, dim=-1, keepdim=True).permute(0, 3, 1, 2)

    # Resize to 84 x 84
    frames = F.interpolate(frames, size=(84, 84), mode='bilinear')
    # remove previous channel dim and add batch dim
    frames = frames.squeeze(1).unsqueeze(0)

    # Scale values to [0, 1]
    return frames / 255.


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


def sample_batch(buffer, batch_size, device):
    s0_batch = []
    a_batch = []
    r_batch = []
    s1_batch = []
    for i in np.random.randint(0, len(buffer), (batch_size,)):
        s0, a, r, s1 = buffer[i]
        s0_batch.append(s0)
        a_batch.append(a)
        r_batch.append(r)
        s1_batch.append(s1)
    s0_batch = torch.concat(s0_batch, dim=0).to(device)
    a_batch = torch.tensor(a_batch, device=device)
    r_batch = torch.tensor(r_batch, device=device)
    s1_batch = torch.concat(s1_batch, dim=0).to(device)
    return s0_batch, a_batch, r_batch, s1_batch


def dqn(env, q0, q1, params, sgd_step, device):
    buffer = deque(maxlen=params.buffer_size)
    step = 0

    x, info = env.reset(seed=13)
    for episode in range(params.num_episodes):
        frames = [torch.tensor(x, device=device)] * (params.frames_per_state + 1)
        frames = deque(frames, maxlen=len(frames))
        s0 = preprocess(frames)

        for t in range(params.max_episode_time):
            # eps annealed linearly from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter
            eps = max(-9e-7 * step + 1., .1)
            a = eps_greedy(q0(s0.to(device)), eps)

            x, r, terminated, truncated, info = env.step(a)
            episode_end = terminated or truncated

            frames.append(torch.tensor(x, device=device))
            s1 = preprocess(frames)
            transition = (s0.cpu(), a, r, s1.cpu())
            buffer.append(transition)
            s0 = s1

            if len(buffer) < params.buffer_start_size:
                if step % params.log_freq == 0:
                    logging.info('Episode: %d, t: %d, step: %d, buffer: %d', episode, t, step, len(buffer))
            else:
                batch = sample_batch(buffer, params.batch_size, device)
                l = sgd_step(q0, q1, batch, episode_end)
                if step % params.log_freq == 0:
                    logging.info('Episode: %d, t: %d, step: %d, loss: %e', episode, t, step, l)

            step += 1
            if step % params.target_update_freq == 0:
                copy_weights(q0, q1)

            if episode_end:
                break

        x, info = env.reset()


def dqn_step(q0, q1, batch, episode_end, opt, params):
    s0_batch, a_batch, r_batch, s1_batch = batch
    batch_size = s0_batch.shape[0]
    if episode_end:
        target = r_batch
    else:
        with torch.no_grad():
            actions_values = q1(s1_batch)
            a_max = torch.argmax(actions_values, dim=1)
            target = r_batch + params.gamma * actions_values[torch.arange(batch_size), a_max]
    opt.zero_grad()
    output = q0(s0_batch)[torch.arange(batch_size), a_batch]
    loss = torch.mean((target - output) ** 2)
    loss.backward()
    opt.step()
    return loss.item()


def double_dqn_step(q0, q1, batch, episode_end, opt, params):
    s0_batch, a_batch, r_batch, s1_batch = batch
    batch_size = s0_batch.shape[0]
    if episode_end:
        target = r_batch
    else:
        with torch.no_grad():
            a_max = torch.argmax(q0(s1_batch), dim=1)
            target = r_batch + params.gamma * q1(s1_batch)[torch.arange(batch_size), a_max]
    opt.zero_grad()
    output = q0(s0_batch)[torch.arange(batch_size), a_batch]
    loss = torch.mean((target - output) ** 2)
    loss.backward()
    opt.step()
    return loss.item()


@dataclass
class Params:
    # M in the paper
    num_episodes = 1000
    # T in the paper
    max_episode_time = 100_000
    # C in the paper
    target_update_freq = 10_000
    # N in the paper
    buffer_size = 1000_000
    buffer_start_size = 50_000
    # m in the paper
    frames_per_state = 4
    gamma = .99
    lr = .00025
    batch_size = 32
    log_freq = 500


def main():
    logging.basicConfig(
        format='%(asctime)s %(module)s %(levelname)s %(message)s',
        level=logging.INFO, handlers=[logging.StreamHandler()], force=True)

    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    num_actions = env.action_space.n

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: %s', device)

    q0 = qnet(num_actions).to(device)
    q1 = qnet(num_actions).to(device)

    params = Params()

    opt = torch.optim.RMSprop(q0.parameters(), lr=params.lr)
    sgd_step = partial(double_dqn_step, opt=opt, params=params)
    dqn(env, q0, q1, params, sgd_step, device)


if __name__ == '__main__':
    main()
