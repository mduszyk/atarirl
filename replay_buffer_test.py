from replay_buffer import StateBuffer, ReplayBuffer


def test_single_episode():
    buf = StateBuffer(max_len=10, initial_frames=[0, 1, 2], state_len=3)
    assert len(buf) == 1
    buf.append(3)
    assert len(buf) == 2
    assert buf[0] == [0, 1, 2]
    assert buf[1] == [1, 2, 3]
    for i in range(10):
        buf.append(4 + i)
    assert len(buf) == 10
    assert buf[0] == [2, 3, 4]
    assert buf[5] == [7, 8, 9]
    assert buf[9] == [11, 12, 13]


def test_multiple_episodes():
    buf = StateBuffer(max_len=10, initial_frames=[0, 1, 2], state_len=3)
    for i in range(3, 6):
        buf.append(i)
    assert len(buf) == 4
    assert buf[0] == [0, 1, 2]
    assert buf[3] == [3, 4, 5]
    buf.new_episode([6, 7, 8])
    assert len(buf) == 5
    assert buf[4] == [6, 7, 8]
    buf.append(9)
    assert len(buf) == 6
    assert buf[5] == [7, 8, 9]
    buf.new_episode([10, 11, 12])
    assert buf[6] == [10, 11, 12]
    assert len(buf) == 7
    buf.append(13)
    assert len(buf) == 8
    assert buf[7] == [11, 12, 13]
    buf.append(14)
    buf.append(15)
    buf.append(16)
    assert len(buf) == 10
    assert buf[9] == [14, 15, 16]
    assert buf[0] == [1, 2, 3]
    assert buf[1] == [2, 3, 4]
    assert buf[2] == [3, 4, 5]
    assert buf[3] == [6, 7, 8]


def test_empty_episodes():
    buf = StateBuffer(max_len=10, initial_frames=[0, 1, 2], state_len=3)
    for i in range(3, 6):
        buf.append(i)
    buf.new_episode([6, 7, 8])
    buf.new_episode([9, 10, 11])
    assert buf[2] == [2, 3, 4]
    assert buf[3] == [3, 4, 5]
    assert buf[4] == [6, 7, 8]
    assert buf[5] == [9, 10, 11]


def test_replay_buffer():
    buf = ReplayBuffer(max_len=10, initial_frames=[0] * 4, state_len=4)
    assert len(buf) == 0
    buf.append(action=1, reward=1, frame=1)
    assert len(buf) == 1
    assert buf[0] == (1, 1, [0, 0, 0, 0, 1])
    buf.append(action=2, reward=2, frame=2)
    assert len(buf) == 2
    assert buf[1] == (2, 2, [0, 0, 0, 1, 2])
    buf.new_episode(initial_frames=[10] * 4)
    assert len(buf) == 2
    buf.append(action=20, reward=20, frame=20)
    assert len(buf) == 3
    assert buf[2] == (20, 20, [10, 10, 10, 10, 20])
    buf.append(action=30, reward=30, frame=30)
    assert len(buf) == 4
    assert buf[3] == (30, 30, [10, 10, 10, 20, 30])
    assert buf[0] == (1, 1, [0, 0, 0, 0, 1])
    assert buf[1] == (2, 2, [0, 0, 0, 1, 2])
    assert buf[2] == (20, 20, [10, 10, 10, 10, 20])
