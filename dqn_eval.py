import logging
import os
import random
from functools import partial

import ale_py
import cv2
import mlflow
import numpy as np
import gymnasium as gym
import torch

from preprocess import PreprocessWrapper
from utils import load_params


def dqn_agent(state, q0, num_actions, eps=.05):
    if eps > 0 and random.random() < eps:
        return random.randint(0, num_actions - 1)
    actions_values = q0(state)[0]
    return torch.argmax(actions_values).item()


@torch.no_grad()
def play(env, agent, params, video_processed=None, video=None):
    score = 0
    (x, x2), info = env.reset()
    if video_processed is not None:
        video_processed.write(x)
    if video is not None:
        video.write(x2)
    state = torch.concat([x] * params.frames_per_state, dim=1)
    prev_state = state
    for t in range(0, params.max_episode_time):
        # action = env.action_space.sample()
        if torch.all(prev_state == state):
            # fire action starts the breakout game
            action = 1
        else:
            action = agent(state)
        (x, x2), reward, terminated, truncated, info = env.step(action)
        if video_processed is not None:
            video_processed.write(x)
        if video is not None:
            video.write(x2)
        score += reward
        if t % 100 == 0:
            logging.info('t: %d, score: %f, info: %s', t, score, info)
        if terminated or truncated:
            break
        prev_state = state
        state = torch.concat((state[:, 1:, :, :], x), dim=1)
    logging.info('t: %d, score: %f, info: %s', t, score, info)
    return score


class VideoWriter:

    def __init__(self, file, fps=15, size=None):
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.file = file
        self.writer = cv2.VideoWriter(file, codec, fps, size)

    def write(self, frame):
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
            frame = frame[0][0] * 255
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)

    def release(self):
        self.writer.release()
        logging.info('Wrote video: %s', self.file)


def main():
    logging.basicConfig(
        format='%(asctime)s %(module)s %(levelname)s %(message)s',
        level=logging.INFO, handlers=[logging.StreamHandler()], force=True)

    params = load_params(os.environ.get('DQN_PARAMS_FILE', 'dqn_params.toml'),
                         profile=os.environ.get('DQN_PARAMS_PROFILE'), env_var_prefix='DQN_')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: %s', device)

    model_uri = os.environ['DQN_MODEL_URI']
    logging.info('loading: %s', model_uri)
    q0 = mlflow.pytorch.load_model(model_uri, map_location=device)
    q0.eval()

    gym.register_envs(ale_py)
    env = gym.make(params.gym_env_id, render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = PreprocessWrapper(env, params.skip_frames, device, processed_only=False)
    num_actions = env.action_space.n

    agent = partial(dqn_agent, q0=q0, num_actions=num_actions, eps=0)

    env_name = params.gym_env_id.replace('/', '-')
    model_name = model_uri.split('/')[-1]
    video_processed = VideoWriter(f'videos/{env_name}-{model_name}.gray.mp4', size=(84, 84))
    video = VideoWriter(f'videos/{env_name}-{model_name}.mp4', size=(160, 210))
    try:
        score = play(env, agent, params, video_processed, video)
        logging.info('score: %f', score)
    finally:
        video_processed.release()
        video.release()

    env.close()


if __name__ == '__main__':
    main()
