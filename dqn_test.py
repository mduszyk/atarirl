from collections import deque

import torch

from dqn import qnet, copy_weights, sample_batch


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
    copy_weights(q2, q3)
    for param1, param2 in zip(q1.parameters(), q2.parameters()):
        assert torch.equal(param1, param2)
    for param1, param2 in zip(q2.parameters(), q3.parameters()):
        assert torch.equal(param1, param2)
    for param1, param2 in zip(q1.parameters(), q3.parameters()):
        assert torch.equal(param1, param2)


def test_sample_batch():
    buf = deque()
    for _ in range(15):
        s1 = torch.rand((1, 4, 84, 84))
        s2 = torch.rand((1, 4, 84, 84))
        a = torch.randint(0, 4, (1,)).item()
        r = torch.rand((1,)).item()
        t = (s1, a, r, s2)
        buf.append(t)
    s0_batch, a_batch, r_batch, s1_batch = sample_batch(buf, 5, 'cpu')
    assert s0_batch.shape == (5, 4, 84, 84)
    assert a_batch.shape == (5,)
    assert r_batch.shape == (5,)
    assert s1_batch.shape == (5, 4, 84, 84)
