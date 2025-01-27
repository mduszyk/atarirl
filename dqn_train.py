import importlib
import json
import logging
import random
import os
from collections import deque

import blosc
import numpy as np
import torch
from torch import nn

import gymnasium as gym
import ale_py

import mlflow

from preprocess import PreprocessWrapper
from utils import load_params


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
        q0.eval()
        actions_values = q0(s0.to(device))[0]
        q0.train()
        return torch.argmax(actions_values).item()


# eps annealed linearly from eps_start to eps_end
# over the first decay_time frames, and fixed at eps_end thereafter
def next_epsilon(step, eps_start, eps_end, decay_time):
    if step < 0:
        return 1
    elif step > decay_time:
        return .1
    else:
        return (eps_end - eps_start) * step / decay_time + eps_start


def compress(x):
    return blosc.compress(x.numpy().tobytes(), typesize=x.itemsize)


def decompress(b, shape):
    b = blosc.decompress(b)
    x = np.frombuffer(b, dtype=np.float32).reshape(shape)
    return torch.tensor(x)


def sample_batch(buffer, batch_size, compression, device):
    s0_batch = []
    s1_batch = []
    a_batch = []
    r_batch = []
    for i in np.random.randint(0, len(buffer), (batch_size,)):
        s, a, r = buffer[i]
        if compression:
            s = decompress(s, (1, 5, 84, 84))
        s = s.to(device, non_blocking=True)
        s0 = s[:, :-1, :, :]
        s1 = s[:, 1:, :, :]
        s0_batch.append(s0)
        a_batch.append(a)
        r_batch.append(r)
        s1_batch.append(s1)
    s0_batch = torch.concat(s0_batch, dim=0)
    s1_batch = torch.concat(s1_batch, dim=0)
    a_batch = torch.tensor(a_batch).to(device, non_blocking=True)
    r_batch = torch.tensor(r_batch).to(device, non_blocking=True)
    return s0_batch, a_batch, r_batch, s1_batch


def dqn(env, q0, q1, params, opt, target_fn, device):
    q0.train()
    q1.eval()
    step = 0
    sgd_step = 0
    target_update_step = 0

    num_actions = env.action_space.n
    replay_buffer = deque(maxlen=params['buffer_max_size'])
    x, info = env.reset(seed=13)
    for episode in range(1, params['num_episodes'] + 1):
        s0 = torch.concat([x] * params['frames_per_state'], dim=1)
        score = 0
        avg_loss = 0

        t = 0
        eps = 1
        for t in range(1, params['max_episode_time'] + 1):
            eps = next_epsilon(step - params['buffer_min_size'],
                               params['eps_start'], params['eps_end'], params['eps_decay_time'])
            a = eps_greedy(eps, q0, s0, num_actions, device)

            x, r, terminated, truncated, info = env.step(a)
            episode_end = terminated or truncated
            score += r

            s = torch.concat((s0, x), dim=1)
            s0 = s[:, 1:, :, :]
            s = s.cpu()
            if params['buffer_compression']:
                s = compress(s)
            transition = (s, a, np.clip(r, -1, 1))
            replay_buffer.append(transition)

            step += 1
            if len(replay_buffer) >= params['buffer_min_size'] and step % params['sgd_update_freq'] == 0:
                batch = sample_batch(replay_buffer, params['batch_size'], params['buffer_compression'], device)
                avg_loss += dqn_sgd_step(q0, q1, batch, episode_end, opt, target_fn, params['gamma'])
                sgd_step += 1
                if sgd_step % params['target_update_freq'] == 0:
                    copy_weights(q0, q1)
                    target_update_step += 1

            if episode_end:
                break

        x, info = env.reset()

        avg_loss /= t
        logging.info('Episode: %d, len: %d, step: %d, sgd: %d, target: %d, buf: %d, eps: %.2f, score: %.2f, loss: %e',
                     episode, t, step, sgd_step, target_update_step, len(replay_buffer), eps, score, avg_loss)
        metrics = {
            'eps': eps,
            'buffer': len(replay_buffer),
            'avg_loss': avg_loss,
            'score': score,
            'episode_length': t,
        }
        mlflow.log_metrics(metrics, step=step)
        if episode % params['model_log_freq'] == 0:
            mlflow.pytorch.log_model(q0, f'q0_episode_{episode}')


def dqn_sgd_step(q0, q1, batch, episode_end, opt, target_fn, gamma):
    s0_batch, a_batch, r_batch, s1_batch = batch
    m = s0_batch.shape[0]
    target = target_fn(q0, q1, batch, episode_end, gamma)
    opt.zero_grad()
    output = q0(s0_batch)[torch.arange(m), a_batch]
    error = torch.clip(target - output, -1, 1)
    loss = torch.mean(error ** 2)
    loss.backward()
    opt.step()
    return loss.item()


def double_dqn_target(q0, q1, batch, episode_end, gamma):
    s0_batch, a_batch, r_batch, s1_batch = batch
    if episode_end:
        return r_batch
    m = s0_batch.shape[0]
    with torch.no_grad():
        q0.eval()
        a_max = torch.argmax(q0(s1_batch), dim=1)
        q0.train()
        return r_batch + gamma * q1(s1_batch)[torch.arange(m), a_max]


def dqn_target(q0, q1, batch, episode_end, gamma):
    s0_batch, a_batch, r_batch, s1_batch = batch
    if episode_end:
        return r_batch
    m = s0_batch.shape[0]
    with torch.no_grad():
        actions_values = q1(s1_batch)
        a_max = torch.argmax(actions_values, dim=1)
        return r_batch + gamma * actions_values[torch.arange(m), a_max]


def main():
    mlflow.set_experiment('dqn')

    logging.basicConfig(
        format='%(asctime)s %(module)s %(levelname)s %(message)s',
        level=logging.INFO, handlers=[logging.StreamHandler()], force=True)

    params = load_params(
        os.environ.get('DQN_PARAMS_FILE', 'dqn_params.toml'),
        profile=os.environ.get('DQN_PARAMS_PROFILE'), env_var_prefix='DQN_')
    logging.info('Params:\n%s', json.dumps(params, indent=4))

    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    torch.use_deterministic_algorithms(True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: %s', device)

    gym.register_envs(ale_py)
    env = gym.make(params['gym_env_id'], render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = PreprocessWrapper(env, params['skip_frames'], device, noop_max=params['noop_max'])
    num_actions = env.action_space.n

    q0 = qnet(num_actions).to(device)
    q1 = qnet(num_actions).to(device)
    copy_weights(q0, q1)

    module = importlib.import_module('.'.join(params['optimizer_class'].split('.')[:-1]))
    optimizer_class = getattr(module, params['optimizer_class'].split('.')[-1])
    opt = optimizer_class(q0.parameters(), lr=params['lr'], **params['optimizer_kwargs'])

    with mlflow.start_run(run_name=params['gym_env_id']):
        mlflow.log_params(params)
        dqn(env, q0, q1, params, opt, double_dqn_target, device)
        env.close()

    mlflow.pytorch.log_model(q0, "q0")

if __name__ == '__main__':
    main()
