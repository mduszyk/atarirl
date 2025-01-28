import importlib
import json
import logging
import random
import os

import blosc
import numpy as np
import torch
from torch import nn

import gymnasium as gym
import ale_py

import mlflow

from preprocess import PreprocessWrapper
from replay_buffer import ReplayBuffer
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
def next_epsilon(step, params):
    if step < 0:
        return 1
    elif step > params.eps_decay_time:
        return .1
    else:
        return (params.eps_end - params.eps_start) * step / params.eps_decay_time + params.eps_start


def compress(x):
    x = x.to('cpu', copy=False)
    return blosc.compress(x.numpy().tobytes(), typesize=x.itemsize)


def decompress(b, shape):
    b = blosc.decompress(b)
    x = np.frombuffer(b, dtype=np.float32).reshape(shape)
    return torch.tensor(x)


def sample_batch(buffer, params, device):
    s0_batch = []
    s1_batch = []
    a_batch = []
    r_batch = []
    for i in np.random.randint(0, len(buffer), (params.batch_size,)):
        action, reward, state_transition = buffer[i]
        if params.buffer_compression:
            state_transition = [decompress(frame, (1, 1, 84, 84)) for frame in state_transition]
        state_transition = torch.concat(state_transition, dim=1).to(device, non_blocking=True)
        s0 = state_transition[:, :-1, :, :]
        s1 = state_transition[:, 1:, :, :]
        s0_batch.append(s0)
        a_batch.append(action)
        r_batch.append(reward)
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
    frame, info = env.reset(seed=params.random_seed)
    frames = [frame] * params.frames_per_state
    initial_frames = list(map(compress, frames)) if params.buffer_compression else frames
    replay_buffer = ReplayBuffer(params.buffer_max_size, initial_frames, params.frames_per_state)

    for episode in range(1, params.num_episodes + 1):
        state = torch.concat(frames, dim=1)
        score = 0
        avg_loss = 0
        t = 1
        eps = 1

        for t in range(1, params.max_episode_time + 1):
            eps = next_epsilon(step - params.buffer_min_size, params)
            action = eps_greedy(eps, q0, state, num_actions, device)

            frame, reward, terminated, truncated, info = env.step(action)
            episode_end = terminated or truncated
            score += reward

            state = torch.concat((state, frame), dim=1)[:, 1:, :, :]
            frame = frame.to('cpu', copy=False)
            frame = compress(frame) if params.buffer_compression else frame
            replay_buffer.append(action, np.clip(reward, -1, 1), frame)

            step += 1
            if len(replay_buffer) >= params.buffer_min_size and step % params.sgd_update_freq == 0:
                batch = sample_batch(replay_buffer, params, device)
                avg_loss += dqn_sgd_step(q0, q1, batch, episode_end, opt, target_fn, params.gamma)
                sgd_step += 1
                if sgd_step % params.target_update_freq == 0:
                    copy_weights(q0, q1)
                    target_update_step += 1

            if episode_end:
                break

        frame, info = env.reset()
        frames = [frame] * params.frames_per_state
        initial_frames = list(map(compress, frames)) if params.buffer_compression else frames
        replay_buffer.new_episode(initial_frames)

        avg_loss /= t
        logging.info('Episode: %d, len: %d, step: %d, sgd: %d, target: %d, buf: %d, eps: %.2f, score: %.2f, loss: %e',
                     episode, t, step, sgd_step, target_update_step, len(replay_buffer), eps, score, avg_loss)
        metrics = {
            'eps': eps,
            'buffer': len(replay_buffer),
            'loss': avg_loss,
            'score': score,
            'episode_length': t,
        }
        mlflow.log_metrics(metrics, step=step)
        if episode % params.model_log_freq == 0:
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

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.use_deterministic_algorithms(True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: %s', device)

    gym.register_envs(ale_py)
    env = gym.make(params.gym_env_id, render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = PreprocessWrapper(env, params.skip_frames, device, noop_max=params.noop_max)
    num_actions = env.action_space.n

    q0 = qnet(num_actions).to(device)
    q1 = qnet(num_actions).to(device)  # target network
    copy_weights(q0, q1)

    module = importlib.import_module('.'.join(params.optimizer_class.split('.')[:-1]))
    optimizer_class = getattr(module, params.optimizer_class.split('.')[-1])
    opt = optimizer_class(q0.parameters(), lr=params.lr, **params.optimizer_kwargs)

    target_fn = globals()[params.target]

    with mlflow.start_run(run_name=params.gym_env_id):
        mlflow.log_params(params)
        dqn(env, q0, q1, params, opt, target_fn, device)
        env.close()

    mlflow.pytorch.log_model(q0, "q0")

if __name__ == '__main__':
    main()
