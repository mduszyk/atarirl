import numpy as np
import torch

from preprocess import preprocess


def test_preprocess():
    frames = [(np.random.rand(210, 160, 3) * 255).astype(np.uint8) for _ in range(5)]
    frames = list(map(torch.tensor, frames))
    s = preprocess(frames)
    assert s.shape == (1, 4, 84, 84)
