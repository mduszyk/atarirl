import torch
from dqn import qnet, copy_weights


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
