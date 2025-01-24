import torch
import torch.nn.functional as F

import gymnasium as gym


def preprocess(frame):
    # frame shape: 210 x 160 x 3

    # Extract the Y channel, luminance, H x W x 3 -> H x W x 1, Y = 0.299 * R + 0.587 * G + 0.114 * B
    # This reflects how humans perceive brightness.
    weights = torch.tensor([0.299, 0.587, 0.114], device=frame.device).view(1, 1, 1, 3)
    frame = torch.sum(frame * weights, dim=-1, keepdim=True).permute(0, 3, 1, 2)

    # Resize to 84 x 84
    frame = F.interpolate(frame, size=(84, 84), mode='bilinear')
    # remove previous channel dim and add batch dim
    frame = frame.squeeze(1).unsqueeze(0)

    # Scale values to [0, 1]
    return frame / 255.


class PreprocessWrapper(gym.Wrapper):

    def __init__(self, env, skip, device, processed_only=True):
        super().__init__(env)
        self.skip = skip
        self.device = device
        self.processed_only = processed_only

    @torch.no_grad()
    def reset(self, **kwargs):
        x, info = self.env.reset(**kwargs)
        frame = torch.tensor(x, device=self.device)
        if self.processed_only:
            return preprocess(frame), info
        return (preprocess(frame), x), info

    @torch.no_grad()
    def step(self, action):
        total_reward = 0.
        terminated = False
        truncated = False
        frame1 = None
        frame2 = None
        for i in range(self.skip):
            x, reward, terminated, truncated, info = self.env.step(action)
            frame1 = frame2
            frame2 = torch.tensor(x, device=self.device)
            total_reward += float(reward)
            if terminated or truncated:
                break
        if frame1 is None:
            frame = frame2
        elif frame2 is None:
            frame = frame1
        else:
            frame = torch.maximum(frame1, frame2)
        frame = preprocess(frame)
        if self.processed_only:
            return frame, total_reward, terminated, truncated, info
        return (frame, x), total_reward, terminated, truncated, info
