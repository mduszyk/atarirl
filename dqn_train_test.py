from collections import deque

import torch

from dqn_train import qnet, copy_weights, sample_batch, compress
from replay_buffer import ReplayBuffer
from utils import AttrDict


def test_qnet():
    q = qnet(4)
    x = torch.rand((8, 4, 84, 84))
    y = q(x)
    assert y.shape == (8, 4)


def test_copy_weights():
    q1 = qnet(4)
    q2 = qnet(4)
    q3 = qnet(4)
    copy_weights(q1, q2)
    assert next(q1.parameters()).data is not next(q2.parameters()).data
    copy_weights(q2, q3)
    for param1, param2 in zip(q1.parameters(), q2.parameters()):
        assert torch.equal(param1, param2)
    for param1, param2 in zip(q2.parameters(), q3.parameters()):
        assert torch.equal(param1, param2)
    for param1, param2 in zip(q1.parameters(), q3.parameters()):
        assert torch.equal(param1, param2)


def test_sample_batch():
    initial_frames = [compress(torch.rand((1, 1, 84, 84)))] * 4
    buf = ReplayBuffer(20, initial_frames, 4)
    for _ in range(15):
        frame = torch.rand((1, 1, 84, 84))
        frame = compress(frame)
        action = torch.randint(0, 4, (1,)).item()
        reward = torch.rand((1,)).item()
        transition = (action, reward, frame)
        buf.append(transition)
    params = AttrDict({'batch_size': 5, 'buffer_compression': True})
    s0_batch, a_batch, r_batch, s1_batch = sample_batch(buf, params, 'cpu')
    assert s0_batch.shape == (5, 4, 84, 84)
    assert a_batch.shape == (5,)
    assert r_batch.shape == (5,)
    assert s1_batch.shape == (5, 4, 84, 84)
