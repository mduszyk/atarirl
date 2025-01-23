import logging
import random
from functools import partial

import ale_py
import cv2
import mlflow
import numpy as np
import gymnasium as gym
import torch

from dqn import Params
from preprocess import PreprocessWrapper


def dqn_agent(state, q0):
    actions_values = q0(state)[0]
    return torch.argmax(actions_values).item()


@torch.no_grad()
def play(env, agent, params, video_processed=None, video=None):
    payoff = 0
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
        payoff += reward
        if t % 100 == 0:
            logging.info('t: %d, payoff: %f, info: %s', t, payoff, info)
        if terminated or truncated:
            break
        prev_state = state
        state = torch.concat((state[:, 1:, :, :], x), dim=1)
    logging.info('t: %d, payoff: %f, info: %s', t, payoff, info)
    return payoff


class VideoWriter:

    def __init__(self, file, fps=15, size=None):
        codec = cv2.VideoWriter_fourcc(*'mp4v')
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


def main():
    mlflow.set_tracking_uri("file:///tmp/mlflow")

    logging.basicConfig(
        format='%(asctime)s %(module)s %(levelname)s %(message)s',
        level=logging.INFO, handlers=[logging.StreamHandler()], force=True)

    params = Params()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Device: %s', device)

    # model_uri = 'runs:/35543818a44b42c8b44c331b1d48b919/q0_episode_5000'
    model_uri = 'runs:/7675e5d22c5e4a1dba628059e8e9e1c0/q0_episode_4000'
    logging.info('loading: %s', model_uri)
    q0 = mlflow.pytorch.load_model(model_uri, map_location=device)
    q0.eval()

    gym.register_envs(ale_py)
    env = gym.make(params.env_id, render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = PreprocessWrapper(env, params.skip_frames, device, processed_only=False)

    agent = partial(dqn_agent, q0=q0)

    video_processed = VideoWriter('videos/dqn_eval_processed.mp4', size=(84, 84))
    # video_processed = None
    video = VideoWriter('videos/dqn_eval.mp4', size=(160, 210))
    try:
        payoff = play(env, agent, params, video_processed, video)
        logging.info('payoff: %f', payoff)
    finally:
        env.close()
        video_processed.release()
        video.release()


if __name__ == '__main__':
    main()
