import torch

from preprocess import preprocess


def test_preprocess():
    frame = (torch.rand((210, 160, 3)) * 255).to(torch.uint8)
    s = preprocess(frame)
    assert s.shape == (1, 1, 84, 84)
