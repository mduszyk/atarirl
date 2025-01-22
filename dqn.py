import logging
import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from functools import partial

import blosc
import numpy as np
import torch
from torch import nn

import gymnasium as gym
import ale_py

import mlflow

from preprocess import PreprocessWrapper


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


def eps_greedy(eps, q0, s0, num_actions, device):
    if random.random() < eps:
        return random.randint(0, num_actions - 1)
    with torch.no_grad():
        actions_values = q0(s0.to(device))
        return torch.argmax(actions_values)


# eps annealed linearly from 1.0 to 0.1 over the first million frames, and fixed at 0.1 thereafter
def next_epsilon(step):
    return max(-9e-7 * step + 1., .1)


def compress(x):
    return blosc.compress(x.cpu().numpy().tobytes(), typesize=x.itemsize)


def decompress(b, shape, device):
    b = blosc.decompress(b)
    return torch.tensor(np.frombuffer(b, dtype=np.float32).reshape(shape), device=device)


def sample_batch(buffer, batch_size, device):
    s0_batch = []
    a_batch = []
    r_batch = []
    s1_batch = []
    for i in np.random.randint(0, len(buffer), (batch_size,)):
        s, a, r = buffer[i]
        s = decompress(s, (1, 5, 84, 84), device)
        s0 = s[:, :-1, :, :]
        s1 = s[:, 1:, :, :]
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
    num_actions = env.action_space.n
    replay_buffer = deque(maxlen=params.buffer_size)
    step = 0

    x, info = env.reset(seed=13)
    for episode in range(1, params.num_episodes + 1):
        episode_end = False
        payoff = 0

        frames = [x]
        for _ in range(params.frames_per_state - 1):
            a = env.action_space.sample()
            x, r, terminated, truncated, info = env.step(a)
            episode_end = terminated or truncated
            if episode_end: break
            frames.append(x)
            payoff += r
        if episode_end: continue
        s0 = torch.concat(frames, dim=1)

        for t in range(params.max_episode_time):
            t0 = time.time()

            eps = 1
            if len(replay_buffer) >= params.buffer_start_size:
                eps = next_epsilon(step - params.buffer_start_size)
            mlflow.log_metric('eps', eps, step=step)
            a = eps_greedy(eps, q0, s0, num_actions, device)

            x, r, terminated, truncated, info = env.step(a)
            episode_end = terminated or truncated
            payoff += r

            t1 = time.time()
            r = np.clip(r, -1, 1)
            s1 = torch.concat((s0, x), dim=1)
            s = compress(s1)
            transition = (s, a, r)
            replay_buffer.append(transition)
            s0 = s1[:, 1:, :, :]

            mlflow.log_metric('buffer', len(replay_buffer), step=step)
            if len(replay_buffer) < params.buffer_start_size:
                if step % params.log_freq == 0:
                    logging.info('Episode: %d, t: %d, step: %d, eps: %f, buffer: %d',
                                 episode, t, step, eps, len(replay_buffer))
            else:
                batch = sample_batch(replay_buffer, params.batch_size, device)
                l = sgd_step(q0, q1, batch, episode_end)
                mlflow.log_metric("loss", l, step=step)
                if step % params.log_freq == 0:
                    logging.info('Episode: %d, t: %d, step: %d, eps: %f, loss: %e',
                                 episode, t, step, eps, l)

            step += 1
            if step % params.target_update_freq == 0:
                copy_weights(q0, q1)

            t2 = time.time()
            mlflow.log_metric('sgd_over_envstep_time', (t2 - t1) / (t1 - t0), step=step)

            if episode_end: break

        x, info = env.reset()

        mlflow.log_metric('payoff', payoff, step=step)
        if episode % params.model_log_freq == 0:
            mlflow.pytorch.log_model(q0, f'q0_episode_{episode}')


def dqn_sgd_step(q0, q1, batch, episode_end, opt, params, target_fn):
    s0_batch, a_batch, r_batch, s1_batch = batch
    m = s0_batch.shape[0]
    target = target_fn(q0, q1, batch, episode_end, params)
    opt.zero_grad()
    output = q0(s0_batch)[torch.arange(m), a_batch]
    loss = torch.mean((target - output) ** 2)
    loss.backward()
    opt.step()
    return loss.item()


def dqn_target(q0, q1, batch, episode_end, params):
    s0_batch, a_batch, r_batch, s1_batch = batch
    if episode_end:
        return r_batch
    m = s0_batch.shape[0]
    with torch.no_grad():
        actions_values = q1(s1_batch)
        a_max = torch.argmax(actions_values, dim=1)
        return r_batch + params.gamma * actions_values[torch.arange(m), a_max]


def double_dqn_target(q0, q1, batch, episode_end, params):
    s0_batch, a_batch, r_batch, s1_batch = batch
    if episode_end:
        return r_batch
    m = s0_batch.shape[0]
    with torch.no_grad():
        a_max = torch.argmax(q0(s1_batch), dim=1)
        return r_batch + params.gamma * q1(s1_batch)[torch.arange(m), a_max]


@dataclass(frozen=True)
class Params:
    # M in the paper
    num_episodes = 100_000
    # T in the paper
    max_episode_time = 100_000
    # C in the paper
    target_update_freq = 10_000
    # N in the paper
    # buffer_size = 1_000_000
    buffer_size = 100_000
    # buffer_start_size = 50_000
    buffer_start_size = 30_000
    # buffer_start_size = 10
    # m in the paper
    frames_per_state = 4
    gamma = .99
    lr = .00025
    batch_size = 32
    log_freq = 500
    env_id = "ALE/Breakout-v5"
    model_log_freq = 500
    skip_frames = 4


def main():
    mlflow.set_experiment('dqn')

    logging.basicConfig(
        format='%(asctime)s %(module)s %(levelname)s %(message)s',
        level=logging.INFO, handlers=[logging.StreamHandler()], force=True)

    params = Params()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: %s', device)

    gym.register_envs(ale_py)
    env = gym.make(params.env_id, render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = PreprocessWrapper(env, params.skip_frames, device)
    num_actions = env.action_space.n

    q0 = qnet(num_actions).to(device)
    q1 = qnet(num_actions).to(device)

    opt = torch.optim.RMSprop(q0.parameters(), lr=params.lr)
    sgd_step = partial(dqn_sgd_step, opt=opt, params=params, target_fn=double_dqn_target)

    with mlflow.start_run(run_name=params.env_id):
        # TODO why asdict(params) does not work ?
        params_dict = dict(filter(lambda x: not x[0].startswith('__'), Params.__dict__.items()))
        mlflow.log_params(params_dict)
        dqn(env, q0, q1, params, sgd_step, device)

    mlflow.pytorch.log_model(q0, "q0")

if __name__ == '__main__':
    main()
