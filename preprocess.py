import random
import torch
import torch.nn.functional as F
import gymnasium as gym


def preprocess(frame):
    # frame shape: 210 x 160 x 3

    # Extract the luminance (Y channel)
    # Y = 0.299 * R + 0.587 * G + 0.114 * B
    weights = torch.tensor([0.299, 0.587, 0.114], device=frame.device).view(1, 1, 1, 3)
    frame = torch.sum(frame * weights, dim=-1, keepdim=True).permute(0, 3, 1, 2)

    # Resize to 84 x 84
    frame = F.interpolate(frame, size=(84, 84), mode='bilinear')

    # Scale values to [0, 1]
    frame /= 255

    return frame


class PreprocessWrapper(gym.Wrapper):

    def __init__(self, env, skip, device, processed_only=True, noop_max=0):
        super().__init__(env)
        self.skip = skip
        self.device = device
        self.processed_only = processed_only
        self.noop_max = noop_max

    @torch.no_grad()
    def reset(self, **kwargs):
        x, info = self.env.reset(**kwargs)
        # offset which frames the agent sees, since it only sees every 4 frames
        if self.noop_max > 0:
            for i in range(random.randint(0, self.noop_max - 1)):
                x, reward, terminated, truncated, info = self.env.step(0)
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
            frame1 = frame2
            x, reward, terminated, truncated, info = self.env.step(action)
            frame2 = x
            total_reward += float(reward)
            if terminated or truncated:
                break
        if frame1 is None:
            frame = torch.tensor(frame2).to(self.device, non_blocking=True)
        elif frame2 is None:
            frame = torch.tensor(frame1).to(self.device, non_blocking=True)
        else:
            frame = torch.maximum(
                torch.tensor(frame1).pin_memory().to(self.device, non_blocking=True),
                torch.tensor(frame2).pin_memory().to(self.device, non_blocking=True)
            )
        frame = preprocess(frame)
        if self.processed_only:
            return frame, total_reward, terminated, truncated, info
        return (frame, x), total_reward, terminated, truncated, info
