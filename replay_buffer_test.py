from replay_buffer import StateBuffer


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
